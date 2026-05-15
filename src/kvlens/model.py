"""Gemma 4 text model assembly."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from kvlens.attention import Attention
from kvlens.config import Gemma4Config
from kvlens.norm import RMSNorm
from kvlens.ple import PerLayerEmbedder, PerLayerInjection, PerLayerProjection


class GemmaMLP(nn.Module):
    """Simple gated MLP used in each decoder block."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(hidden_states), approximate="tanh")
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class GemmaBlock(nn.Module):
    """One decoder block following the Gemma 4 execution order."""

    def __init__(self, config: Gemma4Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = GemmaMLP(config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.ple_dim > 0:
            self.per_layer_injection = PerLayerInjection(
                hidden_size=config.hidden_size,
                ple_dim=config.ple_dim,
                eps=config.rms_norm_eps,
            )
            self.layer_scalar = nn.Parameter(torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        cache: object | None = None,
        metrics: object | None = None,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            positions=positions,
            cache=cache,
            metrics=metrics,
            shared_kv_states=shared_kv_states,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if per_layer_input is not None:
            hidden_states = self.per_layer_injection(hidden_states, per_layer_input)
            return hidden_states * self.layer_scalar
        return hidden_states


class GemmaModel(nn.Module):
    """Gemma 4 text decoder with tied embeddings and PLE."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # Match HF: embed_scale is stored as a buffer so it follows model dtype
        self.register_buffer(
            "embed_scale",
            torch.tensor(math.sqrt(config.hidden_size)),
            persistent=False,
        )

        if config.ple_dim > 0:
            self.embed_tokens_per_layer: PerLayerEmbedder | None = PerLayerEmbedder(
                vocab_size=config.vocab_size,
                num_layers=config.num_layers,
                ple_dim=config.ple_dim,
            )
            self.per_layer_model_projection: PerLayerProjection | None = PerLayerProjection(
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                ple_dim=config.ple_dim,
                eps=config.rms_norm_eps,
            )
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_model_projection = None

        self.layers = nn.ModuleList(
            [GemmaBlock(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens_per_layer is not None
        return self.embed_tokens_per_layer(input_ids)

    def project_per_layer_inputs(
        self,
        hidden_states: torch.Tensor,
        per_layer_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.per_layer_model_projection is not None
        return self.per_layer_model_projection.combine(hidden_states, per_layer_inputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        cache: object | None = None,
        metrics: object | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}"
            )

        batch_size, seq_len = input_ids.shape
        if seq_len == 0:
            raise ValueError("input_ids must have at least one token; got empty sequence")
        if positions is None:
            positions = torch.arange(seq_len, device=input_ids.device)
        if positions.ndim != 1 or positions.shape[0] != seq_len:
            raise ValueError(
                "positions must be a 1D tensor with length matching input_ids.shape[1], "
                f"got shape {tuple(positions.shape)} for seq_len={seq_len}"
            )
        if positions[-1].item() >= self.config.max_position_embeddings:
            raise ValueError(
                f"positions exceed max_position_embeddings={self.config.max_position_embeddings}"
            )

        hidden_states = self.embed_tokens(input_ids) * self.embed_scale
        if self.embed_tokens_per_layer is not None:
            per_layer_inputs: torch.Tensor | None = self.project_per_layer_inputs(
                hidden_states,
                self.get_per_layer_inputs(input_ids),
            )
        else:
            per_layer_inputs = None

        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        for layer_idx, layer in enumerate(self.layers):
            layer_input = (
                per_layer_inputs[:, :, layer_idx, :] if per_layer_inputs is not None else None
            )
            hidden_states = layer(
                hidden_states,
                per_layer_input=layer_input,
                positions=positions,
                cache=cache,
                metrics=metrics,
                shared_kv_states=shared_kv_states,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        softcap = self.config.final_logit_softcapping
        if softcap > 0.0:
            return softcap * torch.tanh(logits / softcap)
        return logits

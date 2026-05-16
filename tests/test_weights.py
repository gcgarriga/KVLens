"""Tests for HuggingFace weight mapping and verification."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from kvlens.model import GemmaModel
from kvlens.weights import (
    hf_to_kvlens_name,
    load_gemma2_weights,
    load_hf_weights,
    map_hf_state_dict,
)


def build_fake_hf_state_dict(model: GemmaModel) -> dict[str, torch.Tensor]:
    hf_state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name == "embed_tokens.weight":
            hf_name = "model.embed_tokens.weight"
        elif name == "embed_tokens_per_layer.embedding.weight":
            hf_name = "model.embed_tokens_per_layer.weight"
        elif name == "per_layer_model_projection.proj.weight":
            hf_name = "model.per_layer_model_projection.weight"
        elif name == "per_layer_model_projection.norm.weight":
            hf_name = "model.per_layer_projection_norm.weight"
        elif name == "norm.weight":
            hf_name = "model.norm.weight"
        elif name.startswith("layers."):
            hf_name = "model." + name
            hf_name = hf_name.replace(".per_layer_injection.gate_proj.", ".per_layer_input_gate.")
            hf_name = hf_name.replace(".per_layer_injection.proj.", ".per_layer_projection.")
            hf_name = hf_name.replace(".per_layer_injection.norm.", ".post_per_layer_input_norm.")
        else:
            hf_name = name
        hf_state[hf_name] = torch.randn_like(tensor)
    hf_state["lm_head.weight"] = hf_state["model.embed_tokens.weight"].clone()
    return hf_state


class TestWeightMapping:
    def test_maps_expected_names(self) -> None:
        assert hf_to_kvlens_name("model.embed_tokens.weight") == "embed_tokens.weight"
        assert hf_to_kvlens_name("model.embed_tokens_per_layer.weight") == (
            "embed_tokens_per_layer.embedding.weight"
        )
        assert hf_to_kvlens_name("model.layers.0.per_layer_input_gate.weight") == (
            "layers.0.per_layer_injection.gate_proj.weight"
        )

    def test_loads_and_verifies_shapes(self, tiny_config) -> None:
        model = GemmaModel(tiny_config)
        hf_state = build_fake_hf_state_dict(model)
        mapped = map_hf_state_dict(hf_state)

        load_hf_weights(model, hf_state)

        for name, tensor in mapped.items():
            torch.testing.assert_close(model.state_dict()[name], tensor)
        assert model.embed_tokens.weight is model.lm_head.weight

    def test_reports_shape_mismatch(self, tiny_config) -> None:
        model = GemmaModel(tiny_config)
        hf_state = build_fake_hf_state_dict(model)
        hf_state["model.embed_tokens.weight"] = torch.randn(1, 1)

        with pytest.raises(RuntimeError, match="shape_errors"):
            load_hf_weights(model, hf_state)

    def test_loads_from_safetensors_file(self, tiny_config, tmp_path: Path) -> None:
        model = GemmaModel(tiny_config)
        hf_state = build_fake_hf_state_dict(model)
        file_path = tmp_path / "toy.safetensors"
        save_file(hf_state, str(file_path))

        load_hf_weights(model, file_path)

        torch.testing.assert_close(
            model.state_dict()["embed_tokens.weight"],
            hf_state["model.embed_tokens.weight"],
        )

    def test_materialized_meta_buffers_match_loaded_dtype(self, tiny_config) -> None:
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                model = GemmaModel(tiny_config)
        finally:
            torch.set_default_dtype(previous_dtype)

        hf_state = {
            name: torch.randn(tensor.shape, dtype=torch.bfloat16)
            for name, tensor in build_fake_hf_state_dict(model).items()
        }

        load_hf_weights(model, hf_state)

        assert model.embed_scale.dtype == model.embed_tokens.weight.dtype
        assert model.embed_scale.dtype == torch.bfloat16


def build_fake_gemma2_state_dict(model: GemmaModel) -> dict[str, torch.Tensor]:
    """Fake HF state dict in Gemma 2 naming (no PLE keys)."""
    hf_state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name in ("lm_head.weight",):
            continue  # tied — covered by embed_tokens
        if name == "embed_tokens.weight":
            hf_state["model.embed_tokens.weight"] = torch.randn_like(tensor)
        elif name == "norm.weight":
            hf_state["model.norm.weight"] = torch.randn_like(tensor)
        elif name.startswith("layers."):
            hf_state["model." + name] = torch.randn_like(tensor)
    return hf_state


class TestGemma2WeightLoading:
    def test_load_gemma2_weights_succeeds(self, tiny_gemma2_config) -> None:
        model = GemmaModel(tiny_gemma2_config)
        hf_state = build_fake_gemma2_state_dict(model)
        load_gemma2_weights(model, hf_state)
        torch.testing.assert_close(
            model.state_dict()["embed_tokens.weight"],
            hf_state["model.embed_tokens.weight"],
        )

    def test_load_gemma2_weights_ties_lm_head(self, tiny_gemma2_config) -> None:
        model = GemmaModel(tiny_gemma2_config)
        hf_state = build_fake_gemma2_state_dict(model)
        load_gemma2_weights(model, hf_state)
        assert model.embed_tokens.weight is model.lm_head.weight
        torch.testing.assert_close(
            model.state_dict()["lm_head.weight"],
            hf_state["model.embed_tokens.weight"],
        )

    def test_load_gemma2_weights_shape_mismatch_raises(self, tiny_gemma2_config) -> None:
        model = GemmaModel(tiny_gemma2_config)
        hf_state = build_fake_gemma2_state_dict(model)
        hf_state["model.embed_tokens.weight"] = torch.randn(1, 1)
        with pytest.raises(RuntimeError, match="shape mismatch"):
            load_gemma2_weights(model, hf_state)

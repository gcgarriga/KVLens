"""Weight download and loading helpers for KVLens."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from kvlens.model import GemmaModel
from kvlens.ple import PerLayerEmbedder
from kvlens.rope import RoPE


def download_weight_snapshot(repo_id: str, *, cache_dir: str | Path | None = None) -> str:
    return snapshot_download(
        repo_id=repo_id,
        allow_patterns=[
            "*.safetensors",
            "tokenizer.model",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        cache_dir=cache_dir,
    )


_TEXT_PREFIXES = ("model.language_model.", "model.")


def _strip_text_prefix(name: str) -> str:
    """Strip the HF text-model prefix, supporting both multimodal and standalone layouts."""
    for prefix in _TEXT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def is_text_param(name: str) -> bool:
    """Return True if the HF key belongs to the text model (not audio/vision)."""
    if name.startswith("model.language_model."):
        return True
    non_text = (
        "model.audio_tower.",
        "model.vision_tower.",
        "model.multi_modal",
        "model.embed_audio",
        "model.embed_vision",
    )
    return not any(name.startswith(p) for p in non_text)


def hf_to_kvlens_name(name: str) -> str | None:
    """Map an HF parameter name to KVLens. Returns None for keys to skip."""
    if name == "lm_head.weight":
        return name

    stripped = _strip_text_prefix(name)

    if stripped == "embed_tokens.weight":
        return "embed_tokens.weight"
    if stripped == "embed_tokens_per_layer.weight":
        return "embed_tokens_per_layer.embedding.weight"
    if stripped == "per_layer_model_projection.weight":
        return "per_layer_model_projection.proj.weight"
    if stripped == "per_layer_projection_norm.weight":
        return "per_layer_model_projection.norm.weight"
    if stripped == "norm.weight":
        return "norm.weight"

    if stripped.startswith("layers."):
        mapped = stripped
        mapped = mapped.replace(".per_layer_input_gate.", ".per_layer_injection.gate_proj.")
        mapped = mapped.replace(".per_layer_projection.", ".per_layer_injection.proj.")
        mapped = mapped.replace(".post_per_layer_input_norm.", ".per_layer_injection.norm.")
        return mapped

    raise KeyError(f"unsupported HuggingFace parameter name '{name}'")


def load_safetensor_state_dict(path: str | Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.is_file():
        return load_file(path, device=device)

    state_dict: dict[str, torch.Tensor] = {}
    for file_path in sorted(path.glob("*.safetensors")):
        for name, tensor in load_file(file_path, device=device).items():
            if name in state_dict:
                raise RuntimeError(f"duplicate tensor '{name}' across safetensor shards")
            state_dict[name] = tensor
    if not state_dict:
        raise RuntimeError(f"no .safetensors files found in {path}")
    return state_dict


def _materialize_meta_buffers(model: GemmaModel, device: str) -> None:
    buffer_dtype = next(
        (parameter.dtype for parameter in model.parameters() if not parameter.is_meta),
        torch.get_default_dtype(),
    )
    for module in model.modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is None or not buffer.is_meta:
                continue
            if name == "embed_scale" and isinstance(module, GemmaModel):
                module._buffers[name] = torch.tensor(
                    math.sqrt(module.config.hidden_size),
                    dtype=buffer_dtype,
                    device=device,
                )
                continue
            if name == "scale" and isinstance(module, PerLayerEmbedder):
                module._buffers[name] = torch.tensor(
                    math.sqrt(module.ple_dim),
                    dtype=buffer_dtype,
                    device=device,
                )
                continue
            if name == "inv_freq" and isinstance(module, RoPE):
                rotary_dim = module.rotary_dim
                module._buffers[name] = 1.0 / (
                    module.theta
                    ** (
                        torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
                        / rotary_dim
                    )
                )
                continue
            raise RuntimeError(f"unsupported meta buffer '{name}' in {type(module).__name__}")


def verify_state_dict_shapes(
    model: GemmaModel,
    mapped_state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str], list[str]]:
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(mapped_state_dict))
    extra = sorted(set(mapped_state_dict) - set(model_state))
    shape_errors: list[str] = []

    for name, tensor in mapped_state_dict.items():
        if name not in model_state:
            continue
        if tuple(model_state[name].shape) != tuple(tensor.shape):
            shape_errors.append(
                f"{name}: expected {tuple(model_state[name].shape)}, got {tuple(tensor.shape)}"
            )

    return missing, extra, shape_errors


def map_hf_state_dict(
    hf_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    for hf_name, tensor in hf_state_dict.items():
        if not is_text_param(hf_name):
            continue
        kvlens_name = hf_to_kvlens_name(hf_name)
        if kvlens_name is not None:
            mapped[kvlens_name] = tensor

    if "embed_tokens.weight" in mapped:
        mapped["lm_head.weight"] = mapped["embed_tokens.weight"]
    elif "lm_head.weight" in mapped:
        mapped["embed_tokens.weight"] = mapped["lm_head.weight"]

    return mapped


def load_gemma2_weights(
    model: GemmaModel,
    source: str | Path | dict[str, torch.Tensor],
    device: str = "cpu",
) -> None:
    """Load Gemma 2 2B weights (no PLE, no KV sharing, different param names)."""
    hf_state_dict = (
        source if isinstance(source, dict) else load_safetensor_state_dict(source, device=device)
    )
    mapped: dict[str, torch.Tensor] = {}

    for hf_name, tensor in hf_state_dict.items():
        stripped = hf_name.removeprefix("model.")
        if stripped == "embed_tokens.weight":
            mapped["embed_tokens.weight"] = tensor
            mapped["lm_head.weight"] = tensor
        elif stripped == "norm.weight":
            mapped["norm.weight"] = tensor
        elif stripped.startswith("layers."):
            mapped[stripped] = tensor

    missing, extra, shape_errors = verify_state_dict_shapes(model, mapped)
    errors = []
    if missing:
        errors.append(f"missing keys: {missing}")
    if shape_errors:
        errors.append(f"shape mismatch: {shape_errors}")
    if errors:
        raise RuntimeError("Gemma 2 weight loading failed — " + "; ".join(errors))
    for key in extra:
        mapped.pop(key, None)
    model.load_state_dict(mapped, strict=True, assign=True)
    model.lm_head.weight = model.embed_tokens.weight
    _materialize_meta_buffers(model, device)


def load_hf_weights(
    model: GemmaModel,
    source: str | Path | dict[str, torch.Tensor],
    device: str = "cpu",
) -> None:
    hf_state_dict = (
        source if isinstance(source, dict) else load_safetensor_state_dict(source, device=device)
    )
    mapped_state_dict = map_hf_state_dict(hf_state_dict)
    missing, extra, shape_errors = verify_state_dict_shapes(model, mapped_state_dict)

    # Shared layers don't have K/V projections — drop those extra keys
    _shared_kv_suffixes = (".k_proj.weight", ".v_proj.weight", ".k_norm.weight", ".v_norm.weight")
    extra_real = [k for k in extra if not k.endswith(_shared_kv_suffixes)]

    if missing or extra_real or shape_errors:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra_real:
            details.append(f"extra={extra_real}")
        if shape_errors:
            details.append(f"shape_errors={shape_errors}")
        raise RuntimeError("weight loading failed: " + "; ".join(details))

    # Remove extra keys before loading
    for key in extra:
        mapped_state_dict.pop(key, None)

    model.load_state_dict(mapped_state_dict, strict=True, assign=True)
    model.lm_head.weight = model.embed_tokens.weight
    _materialize_meta_buffers(model, device)

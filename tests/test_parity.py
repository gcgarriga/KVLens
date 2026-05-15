"""Integration parity test against a real Gemma checkpoint reference.

Verifies that KVLens logits match HuggingFace transformers to within
bfloat16 precision tolerances. The remaining differences (~3.3 max)
come from accumulated bfloat16 matmul rounding over 42 decoder layers.

Set env vars to run:
  KVLENS_PARITY_WEIGHTS  — path to HF checkpoint directory
  KVLENS_PARITY_REFERENCE — path to tests/fixtures/parity_reference.pt
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from kvlens.config import Gemma4Config
from kvlens.model import GemmaModel
from kvlens.weights import load_hf_weights

# bfloat16 matmul rounding compounds ~0.03 per layer over 42 layers
MAX_DIFF_TOLERANCE = 4.0
MEAN_DIFF_TOLERANCE = 0.5
TOP_K = 5


def required_parity_paths() -> tuple[Path | None, Path | None]:
    weights_path = os.environ.get("KVLENS_PARITY_WEIGHTS")
    reference_path = os.environ.get("KVLENS_PARITY_REFERENCE")
    return (
        Path(weights_path) if weights_path else None,
        Path(reference_path) if reference_path else None,
    )


@pytest.mark.integration
def test_real_checkpoint_matches_reference_logits() -> None:
    weights_path, reference_path = required_parity_paths()
    if weights_path is None or reference_path is None:
        pytest.skip("set KVLENS_PARITY_WEIGHTS and KVLENS_PARITY_REFERENCE to run parity")

    payload = torch.load(reference_path, map_location="cpu", weights_only=True)
    input_ids = payload["input_ids"]
    reference_logits = payload["reference_logits"]

    model = GemmaModel(Gemma4Config())
    load_hf_weights(model, weights_path)
    model.to(torch.bfloat16).eval()

    with torch.no_grad():
        actual_logits = model(input_ids)

    actual_f32 = actual_logits.float()
    ref_f32 = reference_logits.float()
    diff = (actual_f32 - ref_f32).abs()

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    assert max_diff < MAX_DIFF_TOLERANCE, (
        f"Max logit diff {max_diff:.4f} exceeds tolerance {MAX_DIFF_TOLERANCE}"
    )
    assert mean_diff < MEAN_DIFF_TOLERANCE, (
        f"Mean logit diff {mean_diff:.4f} exceeds tolerance {MEAN_DIFF_TOLERANCE}"
    )

    # Top-1 agreement at all positions
    for pos in range(input_ids.shape[1]):
        actual_top1 = actual_f32[0, pos].argmax().item()
        ref_top1 = ref_f32[0, pos].argmax().item()
        assert actual_top1 == ref_top1, (
            f"Top-1 mismatch at position {pos}: actual={actual_top1} ref={ref_top1}"
        )

    # Top-K agreement at the last position (the generation-critical one)
    last = input_ids.shape[1] - 1
    actual_topk = actual_f32[0, last].topk(TOP_K).indices.tolist()
    ref_topk = ref_f32[0, last].topk(TOP_K).indices.tolist()
    assert actual_topk == ref_topk, (
        f"Top-{TOP_K} mismatch at last position: actual={actual_topk} ref={ref_topk}"
    )

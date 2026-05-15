"""Tests for instrumentation."""

from __future__ import annotations

import torch

from kvlens.instrument import MetricsCollector


class TestMetricsCollector:
    def test_records_expected_structure(self) -> None:
        collector = MetricsCollector(enabled=True)
        collector.start_step(0)
        attention = torch.tensor([[[[0.25, 0.75]]]], dtype=torch.float32)
        collector.record_attention(0, attention, cached_tokens=1, new_tokens=1)
        collector.record_cache(0, bytes_used=128, evicted=4)
        collector.record_latency("forward", 1.5)
        collector.finalize_step(generated_token=42)

        assert len(collector.steps) == 1
        step = collector.steps[0]
        assert step["step"] == 0
        assert step["generated_token"] == 42
        assert step["cache"][0]["bytes"] == 128
        assert step["cache"][0]["evicted"] == 4
        assert "entropy" in step["attention"][0]
        assert step["latency_ms"]["forward"] == 1.5

    def test_disabled_path_retains_no_metrics(self) -> None:
        collector = MetricsCollector(enabled=False)
        collector.start_step(0)
        collector.record_cache(0, bytes_used=128)
        collector.record_latency("forward", 1.0)
        collector.finalize_step()

        assert collector.steps == []
        assert collector.latest() is None

    def test_on_step_callback_fires(self) -> None:
        received: list[dict] = []
        collector = MetricsCollector(enabled=True, on_step=received.append)
        collector.start_step(0)
        collector.record_cache(0, bytes_used=64)
        collector.finalize_step(generated_token=7)

        assert len(received) == 1
        assert received[0]["generated_token"] == 7
        assert received[0]["cache"][0]["bytes"] == 64

    def test_on_step_not_called_when_disabled(self) -> None:
        received: list[dict] = []
        collector = MetricsCollector(enabled=False, on_step=received.append)
        collector.start_step(0)
        collector.finalize_step()

        assert received == []

    def test_on_step_receives_copy_not_reference(self) -> None:
        received: list[dict] = []
        collector = MetricsCollector(enabled=True, on_step=received.append)
        collector.start_step(0)
        collector.record_cache(0, bytes_used=100)
        collector.finalize_step(generated_token=1)

        # Mutating the received data should not affect collector internals
        received[0]["cache"][0]["bytes"] = 999
        assert collector.steps[0]["cache"][0]["bytes"] == 100

    def test_on_step_default_none_no_error(self) -> None:
        collector = MetricsCollector(enabled=True)
        collector.start_step(0)
        collector.finalize_step(generated_token=1)
        assert len(collector.steps) == 1

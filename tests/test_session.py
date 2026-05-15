"""Tests for the interactive session: command parsing, state management, dispatch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx
import pytest
import torch

from kvlens.config import Gemma4Config
from kvlens.generation import GenerationResult
from kvlens.session import (
    VALID_STRATEGIES,
    SessionState,
    cmd_help,
    cmd_reset,
    cmd_strategy,
    dispatch_command,
    parse_command,
    parse_input,
)

# ── parse_input ──────────────────────────────────────────────


class TestParseInput:
    def test_plain_text(self) -> None:
        assert parse_input("Hello world") == ("prompt", "Hello world")

    def test_slash_command(self) -> None:
        assert parse_input("/strategy naive") == ("command", "strategy naive")

    def test_slash_only(self) -> None:
        kind, content = parse_input("/")
        assert kind == "command"

    def test_empty_string(self) -> None:
        assert parse_input("") == ("empty", "")

    def test_whitespace_only(self) -> None:
        assert parse_input("   ") == ("empty", "")

    def test_preserves_prompt_text(self) -> None:
        assert parse_input("  hello  ") == ("prompt", "hello")

    def test_slash_with_leading_space(self) -> None:
        kind, _ = parse_input("  /help")
        assert kind == "command"


class TestParseCommand:
    def test_name_and_args(self) -> None:
        assert parse_command("strategy naive") == ("strategy", "naive")

    def test_name_only(self) -> None:
        assert parse_command("help") == ("help", "")

    def test_empty(self) -> None:
        assert parse_command("") == ("", "")

    def test_case_insensitive(self) -> None:
        name, _ = parse_command("HELP")
        assert name == "help"

    def test_multi_word_args(self) -> None:
        name, args = parse_command("compare standard,naive")
        assert name == "compare"
        assert args == "standard,naive"


# ── cmd_help ─────────────────────────────────────────────────


def test_cmd_help_lists_all_commands() -> None:
    text = cmd_help()
    assert "/strategy" in text
    assert "/reset" in text
    assert "/help" in text


# ── cmd_strategy ─────────────────────────────────────────────


@pytest.fixture
def tiny_config() -> Gemma4Config:
    return Gemma4Config.tiny()


@pytest.fixture
def fresh_state() -> SessionState:
    return SessionState()


class TestCmdStrategy:
    def test_switch_valid(self, fresh_state: SessionState, tiny_config: Gemma4Config) -> None:
        msg = cmd_strategy("quantized", fresh_state, tiny_config)
        assert "quantized" in msg
        assert fresh_state.strategy == "quantized"
        assert fresh_state.position == 0
        assert fresh_state.token_history == []
        assert fresh_state.cache is not None

    def test_switch_resets_state(
        self, fresh_state: SessionState, tiny_config: Gemma4Config
    ) -> None:
        fresh_state.position = 42
        fresh_state.turn_count = 3
        fresh_state.token_history = [1, 2, 3]
        fresh_state.last_prompt_text = "old prompt"
        cmd_strategy("quantized", fresh_state, tiny_config)
        assert fresh_state.position == 0
        assert fresh_state.turn_count == 0
        assert fresh_state.token_history == []
        assert fresh_state.last_prompt_text == ""

    def test_invalid_strategy(self, fresh_state: SessionState) -> None:
        msg = cmd_strategy("invalid", fresh_state, None)
        assert "Unknown strategy" in msg
        assert fresh_state.strategy == "standard"  # unchanged

    def test_empty_shows_current(self, fresh_state: SessionState) -> None:
        msg = cmd_strategy("", fresh_state, None)
        assert "Current strategy" in msg
        assert "standard" in msg

    def test_naive_warns(self, fresh_state: SessionState, tiny_config: Gemma4Config) -> None:
        msg = cmd_strategy("naive", fresh_state, tiny_config)
        assert "⚠" in msg or "doesn't persist" in msg
        assert fresh_state.strategy == "naive"

    def test_all_valid_strategies(
        self, fresh_state: SessionState, tiny_config: Gemma4Config
    ) -> None:
        for name in VALID_STRATEGIES:
            msg = cmd_strategy(name, fresh_state, tiny_config)
            assert fresh_state.strategy == name
            assert "reset" in msg.lower() or "switched" in msg.lower()


# ── cmd_reset ────────────────────────────────────────────────


class TestCmdReset:
    def test_resets_all_state(self, fresh_state: SessionState, tiny_config: Gemma4Config) -> None:
        fresh_state.position = 100
        fresh_state.turn_count = 5
        fresh_state.token_history = [10, 20]
        fresh_state.last_prompt_text = "hello"
        cmd_reset(fresh_state, tiny_config)
        assert fresh_state.position == 0
        assert fresh_state.turn_count == 0
        assert fresh_state.token_history == []
        assert fresh_state.last_prompt_text == ""
        assert fresh_state.cache is not None

    def test_preserves_strategy(
        self, fresh_state: SessionState, tiny_config: Gemma4Config
    ) -> None:
        fresh_state.strategy = "quantized"
        cmd_reset(fresh_state, tiny_config)
        assert fresh_state.strategy == "quantized"


# ── dispatch_command ─────────────────────────────────────────


class TestDispatchCommand:
    def test_dispatches_help(self, fresh_state: SessionState) -> None:
        msg = dispatch_command("help", fresh_state, None)
        assert "/strategy" in msg

    def test_dispatches_strategy(
        self, fresh_state: SessionState, tiny_config: Gemma4Config
    ) -> None:
        msg = dispatch_command("strategy quantized", fresh_state, tiny_config)
        assert fresh_state.strategy == "quantized"
        assert msg  # non-empty response

    def test_dispatches_reset(self, fresh_state: SessionState, tiny_config: Gemma4Config) -> None:
        fresh_state.position = 10
        msg = dispatch_command("reset", fresh_state, tiny_config)
        assert fresh_state.position == 0
        assert msg  # non-empty response

    def test_unknown_command_returns_help(self, fresh_state: SessionState) -> None:
        msg = dispatch_command("foobar", fresh_state, None)
        assert "Unknown command" in msg
        assert "/help" in msg

    def test_unknown_does_not_crash(self, fresh_state: SessionState) -> None:
        # Should not raise
        dispatch_command("", fresh_state, None)
        dispatch_command("   ", fresh_state, None)


# ── CLI integration ──────────────────────────────────────────


class TestCLIIntegration:
    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify no-args invocation prints help and exits 0."""
        from kvlens.cli import main

        result = main([])
        output = capsys.readouterr().out
        assert result == 0
        assert "usage" in output.lower()
        assert "experiment" in output
        assert "plot" in output
        assert "generate" in output

    def test_old_generate_command_still_parses(self) -> None:
        from kvlens.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["generate", "--prompt", "hello"])
        assert args.command == "generate"
        assert args.prompt == "hello"

    def test_old_info_command_still_parses(self) -> None:
        from kvlens.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["info"])
        assert args.command == "info"

    def test_dashboard_cli_flags_are_removed(self) -> None:
        from kvlens.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--port", "7685"])
        with pytest.raises(SystemExit):
            parser.parse_args(["generate", "--prompt", "hello", "--dashboard"])

    def test_dashboard_runtime_module_is_removed(self) -> None:
        assert importlib.util.find_spec("kvlens.server") is None
        assert importlib.util.find_spec("kvlens.dashboard_html") is None

    def test_removed_subcommands_reject_invocation(self) -> None:
        from kvlens.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["compare", "--prompt", "test"])
        with pytest.raises(SystemExit):
            parser.parse_args(["chat"])

    def test_plot_command_parses(self) -> None:
        from kvlens.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["plot", "--input", "results.json"])
        assert args.command == "plot"
        assert args.input == "results.json"
        assert args.output_dir == "figures"
        assert args.model_type == "gemma4"
        assert args.budget == 256

    def test_plot_command_parses_all_args(self) -> None:
        from kvlens.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "plot",
                "--input",
                "my_results.json",
                "--output-dir",
                "out/",
                "--model-type",
                "gemma2",
                "--budget",
                "128",
            ]
        )
        assert args.command == "plot"
        assert args.input == "my_results.json"
        assert args.output_dir == "out/"
        assert args.model_type == "gemma2"
        assert args.budget == 128

    def test_parse_prompts_file_rejects_empty_files(self, tmp_path: Path) -> None:
        from kvlens.cli import _parse_prompts_file

        prompts_path = tmp_path / "empty_prompts.txt"
        prompts_path.write_text(" \n\n")

        with pytest.raises(ValueError, match="No valid prompts found"):
            _parse_prompts_file(prompts_path)

    def test_info_still_works_end_to_end(self) -> None:
        from kvlens.cli import main

        result = main(["info", "--tiny"])
        assert result == 0

    def test_generate_gemma2_path_runs_end_to_end(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_gemma2_config: Gemma4Config,
        tmp_path: Path,
    ) -> None:
        from kvlens import cli

        class FakeTokenizer:
            eos_id = 1
            eot_id = 106

            def encode(self, text: str) -> list[int]:
                assert text == "hello world"
                return [2, 10, 11]

            def decode(self, ids: list[int]) -> str:
                return "decoded"

        calls: dict[str, object] = {}

        def fake_download(repo_id: str, *, cache_dir: str | Path | None = None) -> str:
            del cache_dir
            calls["repo_id"] = repo_id
            return str(tmp_path)

        def fake_from_pretrained(
            repo_id: str,
            *,
            cache_dir: str | Path | None = None,
        ) -> FakeTokenizer:
            del cache_dir
            calls["tokenizer_repo"] = repo_id
            return FakeTokenizer()

        def fake_load_gemma2_weights(model, source: str | Path, device: str = "cpu") -> None:
            calls["loader"] = "gemma2"
            calls["config"] = model.config
            calls["source"] = Path(source)
            state_dict = {
                name: torch.randn(tensor.shape, dtype=torch.float32)
                for name, tensor in model.state_dict().items()
            }
            model.load_state_dict(state_dict, strict=True, assign=True)
            model.lm_head.weight = model.embed_tokens.weight
            from kvlens.weights import _materialize_meta_buffers

            _materialize_meta_buffers(model, device)

        def fail_load_hf_weights(*args, **kwargs) -> None:
            raise AssertionError("Gemma 4 loader should not be used for --model-type gemma2")

        def fake_generate(model, input_ids, generation_config, **kwargs) -> GenerationResult:
            del kwargs
            calls["generated_config"] = model.config
            calls["generation_input_shape"] = tuple(input_ids.shape)
            calls["cache_strategy"] = generation_config.cache_strategy
            return GenerationResult(
                sequences=torch.tensor([[2, 10, 11, 1]], dtype=torch.long),
                generated_ids=[1],
                step_logits=[torch.zeros((1, tiny_gemma2_config.vocab_size))],
            )

        monkeypatch.setattr(cli, "download_weight_snapshot", fake_download)
        monkeypatch.setattr(cli.GemmaTokenizer, "from_pretrained", fake_from_pretrained)
        monkeypatch.setattr(
            cli.Gemma4Config,
            "gemma2_2b",
            classmethod(lambda cls: tiny_gemma2_config),
        )
        monkeypatch.setattr(cli, "load_gemma2_weights", fake_load_gemma2_weights)
        monkeypatch.setattr(cli, "load_hf_weights", fail_load_hf_weights)
        monkeypatch.setattr(cli, "generate", fake_generate)

        result = cli.main(
            [
                "generate",
                "--prompt",
                "hello world",
                "--repo-id",
                "google/gemma-2-2b",
                "--model-type",
                "gemma2",
                "--max-tokens",
                "1",
            ]
        )

        assert result == 0
        assert calls["repo_id"] == "google/gemma-2-2b"
        assert calls["tokenizer_repo"] == "google/gemma-2-2b"
        assert calls["loader"] == "gemma2"
        assert calls["source"] == tmp_path
        assert calls["config"] is tiny_gemma2_config
        assert calls["generated_config"] is tiny_gemma2_config
        assert calls["generation_input_shape"] == (1, 3)
        assert calls["cache_strategy"] == "standard"

    def test_load_runtime_casts_loaded_model_to_bfloat16(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_gemma2_config: Gemma4Config,
        tmp_path: Path,
    ) -> None:
        from kvlens import cli

        class FakeTokenizer:
            pass

        calls: dict[str, object] = {}

        def fake_from_dir(path: str | Path) -> FakeTokenizer:
            calls["tokenizer_path"] = Path(path)
            return FakeTokenizer()

        def fake_load_gemma2_weights(model, source: str | Path, device: str = "cpu") -> None:
            calls["loader"] = "gemma2"
            calls["source"] = Path(source)
            calls["device"] = device
            state_dict = {
                name: torch.randn(tensor.shape, dtype=torch.float32)
                for name, tensor in model.state_dict().items()
            }
            model.load_state_dict(state_dict, strict=True, assign=True)
            model.lm_head.weight = model.embed_tokens.weight
            from kvlens.weights import _materialize_meta_buffers

            _materialize_meta_buffers(model, device)

        monkeypatch.setattr(cli.GemmaTokenizer, "from_dir", fake_from_dir)
        monkeypatch.setattr(
            cli.Gemma4Config,
            "gemma2_2b",
            classmethod(lambda cls: tiny_gemma2_config),
        )
        monkeypatch.setattr(cli, "load_gemma2_weights", fake_load_gemma2_weights)

        args = cli.build_parser().parse_args(
            [
                "generate",
                "--prompt",
                "hello world",
                "--weights-path",
                str(tmp_path),
                "--model-type",
                "gemma2",
                "--device",
                "cpu",
            ]
        )

        model, tokenizer = cli.load_runtime(args)

        assert isinstance(tokenizer, FakeTokenizer)
        assert calls["tokenizer_path"] == tmp_path
        assert calls["loader"] == "gemma2"
        assert calls["source"] == tmp_path
        assert calls["device"] == "cpu"
        assert model.embed_tokens.weight is model.lm_head.weight
        assert model.embed_tokens.weight.dtype == torch.bfloat16
        assert not model.training

    def test_generate_gemma2_gated_repo_error_is_actionable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from kvlens import cli

        def fake_download(repo_id: str, *, cache_dir: str | Path | None = None) -> str:
            del repo_id, cache_dir
            raise cli.GatedRepoError(
                "Access restricted",
                response=httpx.Response(
                    401,
                    request=httpx.Request(
                        "HEAD",
                        "https://huggingface.co/google/gemma-2-2b",
                    ),
                ),
            )

        monkeypatch.setattr(cli, "download_weight_snapshot", fake_download)

        result = cli.main(
            [
                "generate",
                "--prompt",
                "hello world",
                "--repo-id",
                "google/gemma-2-2b",
                "--model-type",
                "gemma2",
                "--max-tokens",
                "1",
            ]
        )

        output = capsys.readouterr().out
        assert result == 1
        assert "Cannot access gated repo 'google/gemma-2-2b'" in output
        assert "hf auth" in output
        assert "login" in output
        assert "--weights-path" in output

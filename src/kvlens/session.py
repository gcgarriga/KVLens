"""Interactive session with slash commands for KVLens."""

from __future__ import annotations

from dataclasses import dataclass, field

from kvlens.cache import CacheProtocol, create_cache
from kvlens.config import Gemma4Config
from kvlens.model import GemmaModel
from kvlens.tokenizer import GemmaTokenizer

VALID_STRATEGIES = (
    "standard",
    "naive",
    "quantized",
    "streaming",
    "h2o",
    "snapkv",
    "pyramidkv",
    "all_layers_h2o",
    "all_layers_snapkv",
    "all_layers_streaming",
    "all_layers_pyramidkv",
)

HELP_TEXT = """\
Commands:
  /strategy <name>  Switch cache strategy
                    (standard, naive, quantized, streaming, h2o,
                     snapkv, pyramidkv, all_layers_h2o,
                     all_layers_snapkv, all_layers_streaming,
                     all_layers_pyramidkv)
  /reset            Clear cache and conversation
  /help             Show this help"""


def parse_input(text: str) -> tuple[str, str]:
    """Parse user input into (kind, content).

    Returns ("command", "name args...") for slash commands,
    or ("prompt", "text") for plain text.
    """
    stripped = text.strip()
    if not stripped:
        return ("empty", "")
    if stripped.startswith("/"):
        # Strip the leading slash and return as command
        return ("command", stripped[1:])
    return ("prompt", stripped)


def parse_command(content: str) -> tuple[str, str]:
    """Split command content into (name, args).

    >>> parse_command("strategy naive")
    ('strategy', 'naive')
    >>> parse_command("help")
    ('help', '')
    """
    parts = content.strip().split(None, 1)
    if not parts:
        return ("", "")
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    return (name, args)


@dataclass
class SessionRuntime:
    """Immutable after creation — holds model artifacts."""

    model: GemmaModel | None = None
    tokenizer: GemmaTokenizer | None = None
    config: Gemma4Config | None = None


@dataclass
class SessionState:
    """Mutable conversation state."""

    strategy: str = "standard"
    cache: CacheProtocol | None = None
    position: int = 0
    turn_count: int = 0
    token_history: list[int] = field(default_factory=list)
    last_prompt_text: str = ""
    last_prompt_is_first_turn: bool = True


def cmd_help() -> str:
    """Return help text listing all commands."""
    return HELP_TEXT


def cmd_strategy(name: str, state: SessionState, config: Gemma4Config | None) -> str:
    """Switch the active cache strategy. Resets cache and conversation state."""
    name = name.strip().lower()
    if not name:
        return f"Current strategy: {state.strategy}\n" + (
            "Usage: /strategy <name>  — options: " + ", ".join(VALID_STRATEGIES)
        )
    if name not in VALID_STRATEGIES:
        return f"Unknown strategy '{name}'. Valid options: {', '.join(VALID_STRATEGIES)}"

    state.strategy = name
    state.position = 0
    state.turn_count = 0
    state.token_history = []
    state.last_prompt_text = ""

    if config is not None:
        state.cache = create_cache(name, config)
    else:
        state.cache = None

    warning = ""
    if name == "naive":
        warning = (
            "\n⚠ Naive cache doesn't persist state across turns"
            " — useful for single-turn experiments only."
        )

    return f"Switched to '{name}' strategy. Cache reset.{warning}"


def cmd_reset(state: SessionState, config: Gemma4Config | None) -> str:
    """Clear cache and conversation state."""
    state.position = 0
    state.turn_count = 0
    state.token_history = []
    state.last_prompt_text = ""

    if config is not None:
        state.cache = create_cache(state.strategy, config)
    else:
        state.cache = None

    return "Cache cleared. Starting fresh."


def dispatch_command(
    content: str,
    state: SessionState,
    config: Gemma4Config | None,
) -> str:
    """Dispatch a slash command and return the response message."""
    name, args = parse_command(content)

    if name == "help":
        return cmd_help()
    if name == "strategy":
        return cmd_strategy(args, state, config)
    if name == "reset":
        return cmd_reset(state, config)

    return f"Unknown command '/{name}'.\n{HELP_TEXT}"

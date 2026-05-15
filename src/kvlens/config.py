"""Model and generation configuration dataclasses for Gemma 4 E4B."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class RoPEConfig:
    """Configuration for Rotary Position Embeddings."""

    theta: float = 10000.0
    partial_factor: float = 1.0


@dataclass(frozen=True)
class LayerParams:
    """Bundled parameters for one layer type (sliding or global)."""

    head_dim: int
    rope: RoPEConfig
    window_size: int | None


@dataclass(frozen=True)
class Gemma4Config:
    """Gemma 4 E4B model configuration.

    All values default to the real Gemma 4 E4B architecture.
    Use Gemma4Config.tiny() for CPU-friendly test configs.
    """

    num_layers: int = 42
    hidden_size: int = 2560
    intermediate_size: int = 10240
    num_heads: int = 8
    num_kv_heads: int = 2
    vocab_size: int = 262144
    max_position_embeddings: int = 131072

    # Per-layer type pattern: "sliding" or "global"
    # Default: every 6th layer (indices 5,11,17,23,29,35,41) is global
    layer_types: tuple[str, ...] = ()

    # Bundled params per layer type
    sliding: LayerParams = field(
        default_factory=lambda: LayerParams(
            head_dim=256,
            rope=RoPEConfig(theta=10000.0),
            window_size=512,
        )
    )
    global_: LayerParams = field(
        default_factory=lambda: LayerParams(
            head_dim=512,
            rope=RoPEConfig(theta=1_000_000.0, partial_factor=0.25),
            window_size=None,
        )
    )

    # Normalization
    rms_norm_eps: float = 1e-6

    # PLE
    ple_dim: int = 256

    # QK normalization (Gemma 4 yes, Gemma 2 no)
    qk_norm: bool = True

    # Per-attention logit softcapping (0.0 = disabled; Gemma 2 uses 50.0)
    attn_logit_softcapping: float = 0.0

    # Final logit softcapping
    final_logit_softcapping: float = 30.0

    # Shared KV: layers (num_layers - num_kv_shared_layers) .. (num_layers - 1)
    # reuse K/V from the last non-shared layer of the same type.
    num_kv_shared_layers: int = 18

    # Per-layer window_size overrides used by iso-memory budgeting. Tuple of
    # (layer_idx, window_size) pairs; window_size may be None to disable eviction.
    per_layer_overrides: tuple[tuple[int, int | None], ...] = ()

    @property
    def first_kv_shared_layer(self) -> int:
        """Index of the first layer that shares KV states."""
        return self.num_layers - self.num_kv_shared_layers

    def is_kv_shared_layer(self, layer_idx: int) -> bool:
        """True if this layer reuses K/V from an earlier layer."""
        return layer_idx >= self.first_kv_shared_layer > 0

    def kv_shared_source(self, layer_idx: int) -> int | None:
        """Return the source layer index for KV sharing, or None if not shared."""
        if not self.is_kv_shared_layer(layer_idx):
            return None
        # Find the last non-shared layer of the same type
        non_shared_types = list(self.layer_types[: self.first_kv_shared_layer])
        target_type = self.layer_types[layer_idx]
        for i in range(len(non_shared_types) - 1, -1, -1):
            if non_shared_types[i] == target_type:
                return i
        raise ValueError(
            f"No non-shared layer of type '{target_type}' found for layer {layer_idx}"
        )

    def stores_kv_for_sharing(self, layer_idx: int) -> bool:
        """True if this non-shared layer is the source for shared layers."""
        if self.is_kv_shared_layer(layer_idx) or self.num_kv_shared_layers == 0:
            return False
        non_shared_types = list(self.layer_types[: self.first_kv_shared_layer])
        target_type = self.layer_types[layer_idx]
        # Is this the LAST non-shared layer of its type?
        last_idx = len(non_shared_types) - 1 - non_shared_types[::-1].index(target_type)
        return layer_idx == last_idx

    def __post_init__(self) -> None:
        # Generate layer_types if not provided
        if not self.layer_types:
            types = []
            for i in range(self.num_layers):
                if (i + 1) % 6 == 0:
                    types.append("global")
                else:
                    types.append("sliding")
            object.__setattr__(self, "layer_types", tuple(types))

        self._validate()

    def _validate(self) -> None:
        if len(self.layer_types) != self.num_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) "
                f"must match num_layers ({self.num_layers})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        for i, lt in enumerate(self.layer_types):
            if lt not in ("sliding", "global"):
                raise ValueError(f"layer_types[{i}] must be 'sliding' or 'global', got '{lt}'")

    @property
    def gqa_ratio(self) -> int:
        """Number of query heads per KV head."""
        return self.num_heads // self.num_kv_heads

    def layer_params(self, layer_idx: int) -> LayerParams:
        """All attention parameters for a given layer — one lookup, one object."""
        for override_idx, override_window in self.per_layer_overrides:
            if override_idx == layer_idx:
                base = self.global_ if self.layer_types[layer_idx] == "global" else self.sliding
                return LayerParams(
                    head_dim=base.head_dim,
                    rope=base.rope,
                    window_size=override_window,
                )
        if self.layer_types[layer_idx] == "global":
            return self.global_
        return self.sliding

    def with_per_layer_budgets(self, budgets: dict[int, int | None]) -> Gemma4Config:
        """Return a copy of this config with per-layer window_size overrides.

        Used by iso-memory budgeting to give each layer its own token cap while
        keeping the rest of the config (head_dim, rope, layer_types) intact.
        """
        return replace(self, per_layer_overrides=tuple(sorted(budgets.items())))

    @classmethod
    def gemma2_2b(cls) -> Gemma4Config:
        """Config for Gemma 2 2B — same hybrid pattern, no PLE, no KV sharing.

        Verify exact values against the HuggingFace config.json before loading real weights.
        """
        return cls(
            num_layers=26,
            hidden_size=2304,
            intermediate_size=9216,
            num_heads=8,
            num_kv_heads=4,
            vocab_size=256000,
            max_position_embeddings=8192,
            layer_types=tuple("sliding" if i % 2 == 0 else "global" for i in range(26)),
            sliding=LayerParams(
                head_dim=256,
                rope=RoPEConfig(theta=10000.0),
                window_size=4096,
            ),
            global_=LayerParams(
                head_dim=256,
                rope=RoPEConfig(theta=10000.0),
                window_size=None,
            ),
            rms_norm_eps=1e-6,
            ple_dim=0,
            qk_norm=False,
            attn_logit_softcapping=50.0,
            final_logit_softcapping=30.0,
            num_kv_shared_layers=0,
        )

    @classmethod
    def tiny(cls) -> Gemma4Config:
        """Tiny config for CPU unit tests (4 layers: 3 sliding + 1 global)."""
        return cls(
            num_layers=4,
            hidden_size=128,
            intermediate_size=512,
            num_heads=4,
            num_kv_heads=2,
            vocab_size=256,
            max_position_embeddings=512,
            layer_types=("sliding", "sliding", "sliding", "global"),
            sliding=LayerParams(head_dim=32, rope=RoPEConfig(), window_size=64),
            global_=LayerParams(
                head_dim=64,
                rope=RoPEConfig(theta=1_000_000.0, partial_factor=0.25),
                window_size=None,
            ),
            ple_dim=16,
            num_kv_shared_layers=0,
        )


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for text generation."""

    max_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None
    cache_strategy: str = "standard"

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0.0, 1.0], got {self.top_p}")
        valid_strategies = {
            "naive",
            "standard",
            "quantized",
            "streaming",
            "h2o",
            "snapkv",
            "pyramidkv",
            "all_layers_h2o",
            "all_layers_snapkv",
            "all_layers_streaming",
            "all_layers_pyramidkv",
        }
        if self.cache_strategy not in valid_strategies:
            raise ValueError(
                f"cache_strategy must be one of {valid_strategies}, got '{self.cache_strategy}'"
            )

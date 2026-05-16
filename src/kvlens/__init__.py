"""KVLens — Educational Gemma 4 E4B inference engine with instrumented KV cache."""

from kvlens.attention import Attention
from kvlens.config import Gemma4Config, GenerationConfig, LayerParams
from kvlens.model import GemmaBlock, GemmaModel
from kvlens.norm import RMSNorm
from kvlens.ple import PerLayerEmbedder, PerLayerInjection, PerLayerProjection
from kvlens.rope import RoPE
from kvlens.tokenizer import GemmaTokenizer, SentencePieceTokenizer

__all__ = [
    "Attention",
    "Gemma4Config",
    "GemmaBlock",
    "GemmaModel",
    "GemmaTokenizer",
    "GenerationConfig",
    "LayerParams",
    "PerLayerEmbedder",
    "PerLayerInjection",
    "PerLayerProjection",
    "RMSNorm",
    "RoPE",
    "SentencePieceTokenizer",
]

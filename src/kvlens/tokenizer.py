"""Tokenizer wrapper supporting both SentencePiece (.model) and HF (.json) formats."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


class GemmaTokenizer:
    """Unified tokenizer that auto-detects SPM or HF fast tokenizer format.

    Gemma 4 ships ``tokenizer.json`` (HF fast tokenizer format) instead of
    the older ``tokenizer.model`` (SentencePiece).  This class supports both.
    """

    _BOS_ID = 2
    _EOS_ID = 1
    _PAD_ID = 0
    _SOT_ID = 105  # <|turn>  (start of turn)
    _EOT_ID = 106  # <turn|>  (end of turn)

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if path.is_dir():
            path = _find_tokenizer_file(path)

        self._path = path
        suffix = path.suffix.lower()

        if suffix == ".model":
            import sentencepiece as spm

            self._backend = "spm"
            self._spm = spm.SentencePieceProcessor(model_file=str(path))
        elif suffix == ".json":
            from tokenizers import Tokenizer

            self._backend = "hf"
            self._hf = Tokenizer.from_file(str(path))
        else:
            raise ValueError(
                f"Unsupported tokenizer file: {path.name}. "
                "Expected .model (SentencePiece) or .json (HF fast tokenizer)."
            )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        cache_dir: str | Path | None = None,
    ) -> GemmaTokenizer:
        for filename in ("tokenizer.model", "tokenizer.json"):
            try:
                model_path = hf_hub_download(
                    repo_id=repo_id, filename=filename, cache_dir=cache_dir
                )
                return cls(model_path)
            except (FileNotFoundError, OSError, EntryNotFoundError):
                continue
        raise FileNotFoundError(f"No tokenizer.model or tokenizer.json found in {repo_id}")

    @classmethod
    def from_dir(cls, directory: str | Path) -> GemmaTokenizer:
        """Load tokenizer from a local directory containing tokenizer files."""
        return cls(Path(directory))

    @property
    def bos_id(self) -> int:
        if self._backend == "spm":
            return self._spm.bos_id()
        return self._BOS_ID

    @property
    def eos_id(self) -> int:
        if self._backend == "spm":
            return self._spm.eos_id()
        return self._EOS_ID

    @property
    def eot_id(self) -> int:
        """End-of-turn token ``<turn|>`` used by Gemma 4 chat format."""
        return self._EOT_ID

    @property
    def sot_id(self) -> int:
        """Start-of-turn token ``<|turn>`` used by Gemma 4 chat format."""
        return self._SOT_ID

    @property
    def pad_id(self) -> int:
        if self._backend == "spm":
            return self._spm.pad_id()
        return self._PAD_ID

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> list[int]:
        if self._backend == "spm":
            token_ids = list(self._spm.encode(text, out_type=int))
            if add_bos:
                token_ids.insert(0, self.bos_id)
        else:
            # HF fast tokenizer auto-adds BOS via its config
            token_ids = self._hf.encode(text).ids
            if not add_bos and token_ids and token_ids[0] == self.bos_id:
                token_ids = token_ids[1:]

        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        if self._backend == "spm":
            return self._spm.decode(token_ids)
        return self._hf.decode(token_ids)

    def encode_chat_turn(
        self,
        text: str,
        *,
        role: str = "user",
        is_first_turn: bool = True,
    ) -> list[int]:
        """Encode a chat turn with Gemma 4 formatting.

        Produces: [<bos>] <|turn> {role}\\n{text} <turn|> \\n <|turn> model\\n
        The trailing ``<|turn>model\\n`` is the generation prefix that cues
        the model to produce an assistant response.
        """
        ids: list[int] = []
        if is_first_turn:
            ids.append(self.bos_id)
        ids.append(self.sot_id)
        ids.extend(self.encode(f"{role}\n{text}", add_bos=False))
        ids.append(self.eot_id)
        ids.extend(self.encode("\n", add_bos=False))
        ids.append(self.sot_id)
        ids.extend(self.encode("model\n", add_bos=False))
        return ids


def _find_tokenizer_file(directory: Path) -> Path:
    """Search a directory for a supported tokenizer file."""
    for name in ("tokenizer.model", "tokenizer.json"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No tokenizer.model or tokenizer.json in {directory}")


# Backwards compatibility alias
SentencePieceTokenizer = GemmaTokenizer

"""Tests for the SentencePiece tokenizer wrapper."""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

from kvlens.tokenizer import SentencePieceTokenizer


def train_sentencepiece_model(tmp_path: Path) -> Path:
    input_path = tmp_path / "train.txt"
    model_prefix = tmp_path / "toy"
    input_path.write_text("hello world\nhello cache\ncache world\n", encoding="utf-8")

    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=16,
        bos_id=2,
        eos_id=1,
        pad_id=0,
        unk_id=3,
    )
    return model_prefix.with_suffix(".model")


class TestSentencePieceTokenizer:
    def test_round_trip_encode_decode(self, tmp_path: Path) -> None:
        model_path = train_sentencepiece_model(tmp_path)
        tokenizer = SentencePieceTokenizer(model_path)

        encoded = tokenizer.encode("hello world", add_bos=False)
        decoded = tokenizer.decode(encoded)

        assert decoded == "hello world"

    def test_encode_prepends_bos(self, tmp_path: Path) -> None:
        model_path = train_sentencepiece_model(tmp_path)
        tokenizer = SentencePieceTokenizer(model_path)

        encoded = tokenizer.encode("hello cache")

        assert encoded[0] == tokenizer.bos_id

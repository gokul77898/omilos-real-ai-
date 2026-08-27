"""Indian Legal Tokenizer based on Byte-Level BPE subword modeling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, List, Optional, Union

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.config import TokenizerConfig, TokenizerSpecialTokens, load_config


@dataclass
class TokenizerOutput:
    """Structured result of tokenizing an input text."""
    ids: List[int]
    tokens: List[str]
    attention_mask: List[int]
    text: str


class LegalTokenizer:
    """Trainable Byte-Level Byte-Pair Encoding (BPE) tokenizer for Indian legal text.

    Key Features:
    - Byte-level base alphabet ensures 0 out-of-vocabulary (OOV) tokens across all Unicode scripts.
    - Lossless round-trip preservation of legal formatting, section symbols (§), quotes, and Indic scripts.
    - Explicit special tokens for autoregressive language modeling (<pad>, <unk>, <s>, </s>).
    - Supports serialization to and deserialization from standardized JSON artifacts.
    """

    def __init__(
        self,
        config: Optional[TokenizerConfig] = None,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self.config = config or TokenizerConfig()
        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = self._build_empty_tokenizer()

    def _build_empty_tokenizer(self) -> Tokenizer:
        """Construct the underlying Tokenizer pipeline with ByteLevel pre-tokenizer and decoder."""
        unk_tok = self.config.special_tokens.unk
        tok = Tokenizer(models.BPE(unk_token=unk_tok))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        return tok

    @property
    def special_tokens(self) -> List[str]:
        """List of configured special tokens."""
        return self.config.special_tokens.to_list()

    @property
    def pad_token(self) -> str:
        return self.config.special_tokens.pad

    @property
    def unk_token(self) -> str:
        return self.config.special_tokens.unk

    @property
    def bos_token(self) -> str:
        return self.config.special_tokens.bos

    @property
    def eos_token(self) -> str:
        return self.config.special_tokens.eos

    @property
    def pad_token_id(self) -> int:
        tid = self.token_to_id(self.pad_token)
        return tid if tid is not None else 0

    @property
    def unk_token_id(self) -> int:
        tid = self.token_to_id(self.unk_token)
        return tid if tid is not None else 1

    @property
    def bos_token_id(self) -> int:
        tid = self.token_to_id(self.bos_token)
        return tid if tid is not None else 2

    @property
    def eos_token_id(self) -> int:
        tid = self.token_to_id(self.eos_token)
        return tid if tid is not None else 3

    @property
    def vocab_size(self) -> int:
        """Return the current vocabulary size."""
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def token_to_id(self, token: str) -> Optional[int]:
        """Get integer token ID from token string."""
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Get string representation from integer token ID."""
        return self._tokenizer.id_to_token(token_id)

    def get_vocab(self) -> dict[str, int]:
        """Return the full mapping from token string to token ID."""
        return self._tokenizer.get_vocab(with_added_tokens=True)

    def train_from_iterator(
        self,
        texts: Iterable[str],
        vocab_size: Optional[int] = None,
        min_frequency: Optional[int] = None,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """Train subword BPE merges from an in-memory string iterator."""
        target_vocab_size = vocab_size or self.config.vocab_size
        target_min_freq = min_frequency if min_frequency is not None else self.config.min_frequency
        target_specials = special_tokens or self.special_tokens

        trainer = trainers.BpeTrainer(
            vocab_size=target_vocab_size,
            min_frequency=target_min_freq,
            special_tokens=target_specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )

        self._tokenizer.train_from_iterator(texts, trainer=trainer)

    def train_from_files(
        self,
        files: List[Union[str, Path]],
        vocab_size: Optional[int] = None,
        min_frequency: Optional[int] = None,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """Train subword BPE merges from text files on disk."""
        target_vocab_size = vocab_size or self.config.vocab_size
        target_min_freq = min_frequency if min_frequency is not None else self.config.min_frequency
        target_specials = special_tokens or self.special_tokens

        file_paths = [str(Path(f).resolve()) for f in files]

        trainer = trainers.BpeTrainer(
            vocab_size=target_vocab_size,
            min_frequency=target_min_freq,
            special_tokens=target_specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )

        self._tokenizer.train(file_paths, trainer=trainer)

    def encode(self, text: str, add_special_tokens: bool = False) -> TokenizerOutput:
        """Encode string text into subword token IDs and metadata."""
        encoded = self._tokenizer.encode(text)
        ids = list(encoded.ids)
        tokens = list(encoded.tokens)

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
            tokens = [self.bos_token] + tokens + [self.eos_token]

        attention_mask = [1] * len(ids)

        return TokenizerOutput(
            ids=ids,
            tokens=tokens,
            attention_mask=attention_mask,
            text=text,
        )

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a sequence of integer token IDs back into text."""
        if skip_special_tokens:
            special_ids = {
                self.pad_token_id,
                self.unk_token_id,
                self.bos_token_id,
                self.eos_token_id,
            }
            cleaned_ids = [i for i in ids if i not in special_ids]
        else:
            cleaned_ids = ids

        return self._tokenizer.decode(cleaned_ids)

    def save(self, save_dir: Union[str, Path]) -> dict[str, str]:
        """Save tokenizer configuration and model artifacts to directory."""
        dest = Path(save_dir)
        dest.mkdir(parents=True, exist_ok=True)

        tokenizer_file = dest / "tokenizer.json"
        config_file = dest / "tokenizer_config.json"

        self._tokenizer.save(str(tokenizer_file))

        config_dict = {
            "vocab_size": self.config.vocab_size,
            "model_type": self.config.model_type,
            "min_frequency": self.config.min_frequency,
            "special_tokens": {
                "pad": self.config.special_tokens.pad,
                "unk": self.config.special_tokens.unk,
                "bos": self.config.special_tokens.bos,
                "eos": self.config.special_tokens.eos,
            },
        }

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        return {
            "tokenizer_file": str(tokenizer_file),
            "config_file": str(config_file),
        }

    @classmethod
    def load(cls, load_dir: Union[str, Path]) -> LegalTokenizer:
        """Load tokenizer model and configuration from directory."""
        src = Path(load_dir)
        tokenizer_file = src / "tokenizer.json"
        config_file = src / "tokenizer_config.json"

        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")

        tokenizer = Tokenizer.from_file(str(tokenizer_file))

        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            st_data = cfg_data.get("special_tokens", {})
            special_tokens = TokenizerSpecialTokens(
                pad=st_data.get("pad", "<pad>"),
                unk=st_data.get("unk", "<unk>"),
                bos=st_data.get("bos", "<s>"),
                eos=st_data.get("eos", "</s>"),
            )
            config = TokenizerConfig(
                vocab_size=cfg_data.get("vocab_size", 32000),
                model_type=cfg_data.get("model_type", "bpe"),
                min_frequency=cfg_data.get("min_frequency", 2),
                special_tokens=special_tokens,
            )
        else:
            config = TokenizerConfig()

        return cls(config=config, tokenizer=tokenizer)


def main() -> None:
    """CLI testing interface for LegalTokenizer."""
    parser = argparse.ArgumentParser(description="Indian Legal Tokenizer CLI")
    parser.add_argument("--text", type=str, default="Section 187 BNSS के अंतर्गत (2024) 5 SCC 123 § 125", help="Text to tokenize")
    parser.add_argument("--corpus", type=str, default="data/synthetic_legal_corpus.txt", help="Path to training corpus")
    parser.add_argument("--save-dir", type=str, default="artifacts/tokenizer", help="Directory to save/load tokenizer")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Development vocab size")
    parser.add_argument("--add-special-tokens", action="store_true", help="Include BOS and EOS tokens in output")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    if (save_dir / "tokenizer.json").exists():
        print(f"Loading existing tokenizer from {save_dir}...")
        tokenizer = LegalTokenizer.load(save_dir)
    else:
        print(f"Training new tokenizer from corpus {args.corpus} (vocab_size={args.vocab_size})...")
        tokenizer = LegalTokenizer(config=TokenizerConfig(vocab_size=args.vocab_size, min_frequency=1))
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            print(f"Corpus file '{corpus_path}' not found! Creating default sample...")
            sample_data = ["Section 302 IPC", "Article 21 Constitution", "Section 187 BNSS"]
            tokenizer.train_from_iterator(sample_data, vocab_size=args.vocab_size, min_frequency=1)
        else:
            tokenizer.train_from_files([corpus_path], vocab_size=args.vocab_size, min_frequency=1)
        tokenizer.save(save_dir)
        print(f"Saved tokenizer to {save_dir}")

    output = tokenizer.encode(args.text, add_special_tokens=args.add_special_tokens)
    decoded = tokenizer.decode(output.ids, skip_special_tokens=not args.add_special_tokens)

    print("\n" + "=" * 50)
    print("TOKENIZER TEST RESULTS")
    print("=" * 50)
    print(f"Input:\n{args.text}\n")
    print(f"Tokens ({len(output.tokens)} tokens):\n{output.tokens}\n")
    print(f"Token IDs ({len(output.ids)} ids):\n{output.ids}\n")
    print(f"Decoded:\n{decoded}\n")
    print(f"Roundtrip Exact Match: {decoded == args.text}")
    print("=" * 50)


if __name__ == "__main__":
    main()

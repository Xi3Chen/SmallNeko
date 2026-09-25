"""A small Unicode character tokenizer for stage 0 pipeline debugging."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


FORMAT_VERSION = 1
DEFAULT_SPECIAL_TOKENS = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<|im_start|>",
    "<|im_end|>",
)


class CharTokenizer:
    """Encode Unicode characters while preserving selected special tokens."""

    def __init__(
        self,
        characters: Iterable[str],
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> None:
        self.special_tokens = tuple(special_tokens)
        self._validate_special_tokens()

        unique_characters = sorted(set(characters))
        overlapping = set(unique_characters).intersection(self.special_tokens)
        if overlapping:
            raise ValueError(f"Characters cannot be complete special tokens: {overlapping}")

        self.tokens = self.special_tokens + tuple(unique_characters)
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        self.itos = dict(enumerate(self.tokens))
        self._special_pattern = re.compile(
            "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> "CharTokenizer":
        """Build a vocabulary from one training text only."""

        special_tokens = tuple(special_tokens)
        normal_text = cls._remove_special_tokens(text, special_tokens)
        return cls(normal_text, special_tokens=special_tokens)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> "CharTokenizer":
        return cls.from_text(Path(path).read_text(encoding="utf-8"), special_tokens)

    @staticmethod
    def _remove_special_tokens(text: str, special_tokens: Iterable[str]) -> str:
        pattern = re.compile(
            "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        )
        return pattern.sub("", text)

    def _validate_special_tokens(self) -> None:
        if not self.special_tokens:
            raise ValueError("At least one special token is required")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("Special tokens must be unique")
        if any(not token for token in self.special_tokens):
            raise ValueError("Special tokens cannot be empty")

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def special_token_ids(self) -> dict[str, int]:
        return {token: self.stoi[token] for token in self.special_tokens}

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text, matching special tokens before individual characters."""

        token_ids: list[int] = []
        cursor = 0
        for match in self._special_pattern.finditer(text):
            token_ids.extend(self._encode_characters(text[cursor : match.start()]))
            token_ids.append(self.stoi[match.group(0)])
            cursor = match.end()
        token_ids.extend(self._encode_characters(text[cursor:]))

        if add_bos:
            token_ids.insert(0, self.stoi["<bos>"])
        if add_eos:
            token_ids.append(self.stoi["<eos>"])
        return token_ids

    def _encode_characters(self, text: str) -> list[int]:
        unknown_id = self.stoi.get("<unk>")
        if unknown_id is None:
            raise ValueError("The tokenizer must contain an <unk> token")
        return [self.stoi.get(character, unknown_id) for character in text]

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs and optionally omit special tokens."""

        pieces: list[str] = []
        for token_id in token_ids:
            try:
                token = self.itos[token_id]
            except KeyError as exc:
                raise ValueError(f"Token ID out of range: {token_id}") from exc
            if skip_special_tokens and token in self.special_tokens:
                continue
            pieces.append(token)
        return "".join(pieces)

    def unknown_characters(self, text: str) -> Counter[str]:
        """Count characters absent from this vocabulary, excluding special tokens."""

        normal_text = self._remove_special_tokens(text, self.special_tokens)
        return Counter(character for character in normal_text if character not in self.stoi)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": FORMAT_VERSION,
            "type": "unicode_character",
            "special_tokens": list(self.special_tokens),
            "characters": [token for token in self.tokens if token not in self.special_tokens],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported tokenizer format: {payload.get('format_version')}")
        if payload.get("type") != "unicode_character":
            raise ValueError(f"Unsupported tokenizer type: {payload.get('type')}")
        return cls(payload["characters"], special_tokens=payload["special_tokens"])

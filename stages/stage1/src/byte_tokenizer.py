"""A fixed-vocabulary UTF-8 byte tokenizer for the stage 1 baseline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


FORMAT_VERSION = 1
NUM_BYTES = 256
DEFAULT_SPECIAL_TOKENS = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<|im_start|>",
    "<|im_end|>",
)


class ByteTokenizer:
    """Encode every UTF-8 byte and keep conversation markers atomic.

    A byte vocabulary has the same IDs for every dataset. It can represent
    any valid Unicode text without rebuilding a character vocabulary.
    """

    def __init__(
        self,
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> None:
        self.special_tokens = tuple(special_tokens)
        self._validate_special_tokens()
        self.byte_offset = len(self.special_tokens)
        self.byte_tokens = tuple(f"<0x{value:02X}>" for value in range(NUM_BYTES))
        self.tokens = self.special_tokens + self.byte_tokens
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        self.itos = dict(enumerate(self.tokens))
        self._special_pattern = re.compile(
            "|".join(
                re.escape(token)
                for token in sorted(self.special_tokens, key=len, reverse=True)
            )
        )

    def _validate_special_tokens(self) -> None:
        if not self.special_tokens:
            raise ValueError("At least one special token is required")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("Special tokens must be unique")
        if any(not token for token in self.special_tokens):
            raise ValueError("Special tokens cannot be empty")

    @classmethod
    def from_text(
        cls,
        text: str,
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> "ByteTokenizer":
        """Build the fixed tokenizer; text is accepted for API symmetry."""

        del text
        return cls(special_tokens=special_tokens)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> "ByteTokenizer":
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        return cls(special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def special_token_ids(self) -> dict[str, int]:
        return {token: self.stoi[token] for token in self.special_tokens}

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text, matching special tokens before UTF-8 bytes."""

        # Windows-to-WSL terminal bridges can pass UTF-16 surrogate code
        # points instead of a normal Unicode scalar value. Repair pairs and
        # replace any unpaired surrogate before UTF-8 encoding.
        text = repair_surrogates(text)
        token_ids: list[int] = []
        cursor = 0
        for match in self._special_pattern.finditer(text):
            token_ids.extend(self._encode_bytes(text[cursor : match.start()].encode("utf-8")))
            token_ids.append(self.stoi[match.group(0)])
            cursor = match.end()
        token_ids.extend(self._encode_bytes(text[cursor:].encode("utf-8")))

        if add_bos:
            token_ids.insert(0, self.stoi["<bos>"])
        if add_eos:
            token_ids.append(self.stoi["<eos>"])
        return token_ids

    def _encode_bytes(self, values: bytes) -> list[int]:
        return [self.byte_offset + value for value in values]

    def decode(
        self,
        token_ids: Iterable[int],
        skip_special_tokens: bool = False,
        drop_incomplete_tail: bool = False,
    ) -> str:
        """Decode IDs as UTF-8, optionally dropping a partial final character."""

        pieces: list[str] = []
        pending = bytearray()

        def flush_bytes(final: bool = False) -> None:
            if pending:
                if final and drop_incomplete_tail:
                    decoded = bytes(pending).decode("utf-8", errors="ignore")
                else:
                    decoded = bytes(pending).decode("utf-8", errors="replace")
                pieces.append(decoded)
                pending.clear()

        for raw_token_id in token_ids:
            token_id = int(raw_token_id)
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token ID out of range: {token_id}")
            if token_id < self.byte_offset:
                flush_bytes()
                token = self.itos[token_id]
                if not skip_special_tokens:
                    pieces.append(token)
            else:
                pending.append(token_id - self.byte_offset)
        flush_bytes(final=True)
        return "".join(pieces)

    def unknown_characters(self, text: str) -> dict[str, int]:
        """Return an empty mapping because all valid Unicode has byte coverage."""

        del text
        return {}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": FORMAT_VERSION,
            "type": "utf8_byte",
            "num_bytes": NUM_BYTES,
            "special_tokens": list(self.special_tokens),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported tokenizer format: {payload.get('format_version')}")
        if payload.get("type") != "utf8_byte":
            raise ValueError(f"Unsupported tokenizer type: {payload.get('type')}")
        if payload.get("num_bytes") != NUM_BYTES:
            raise ValueError(f"Unsupported byte count: {payload.get('num_bytes')}")
        return cls(special_tokens=payload["special_tokens"])


def repair_surrogates(text: str) -> str:
    """Convert UTF-16 surrogate input into valid Unicode text."""

    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-16", errors="surrogatepass").decode("utf-16", errors="replace")
    return text

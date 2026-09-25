"""Encode stage 1 text splits into uint16 token streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from array import array
from pathlib import Path

from byte_tokenizer import ByteTokenizer


STAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = STAGE_ROOT / "data" / "processed" / "byte_tokenizer.json"
DEFAULT_INPUT_DIR = STAGE_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = STAGE_ROOT / "data" / "processed"


def split_safe_prefix(text: str, special_tokens: tuple[str, ...]) -> tuple[str, str]:
    """Keep only a suffix that could be the start of a special token."""

    max_prefix_length = max((len(token) for token in special_tokens), default=1) - 1
    for length in range(min(len(text), max_prefix_length), 0, -1):
        suffix = text[-length:]
        if any(
            len(token) > length and token.startswith(suffix)
            for token in special_tokens
        ):
            return text[:-length], suffix
    return text, ""


def encode_file(tokenizer: ByteTokenizer, input_path: Path, output_path: Path) -> int:
    if tokenizer.vocab_size > 65536:
        raise ValueError("The uint16 data format supports at most 65536 vocabulary entries")
    if array("H").itemsize != 2:
        raise RuntimeError("This platform does not provide a 16-bit unsigned array type")
    if sys.byteorder != "little":
        raise RuntimeError("The current binary format requires a little-endian platform")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # TextIOWrapper itself preserves UTF-8 characters. Keep only a suffix
    # that is actually a possible prefix of a special marker. Keeping an
    # arbitrary number of trailing characters would split complete markers
    # that happen to end near a chunk boundary.
    token_count = 0
    pending = ""
    with input_path.open("r", encoding="utf-8") as source, output_path.open("wb") as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                text = pending
                pending = ""
            else:
                pending += chunk
                text, pending = split_safe_prefix(pending, tokenizer.special_tokens)

            token_ids = tokenizer.encode(text)
            target.write(array("H", token_ids).tobytes())
            token_count += len(token_ids)
            if not chunk:
                break
    return token_count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    tokenizer_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, int]:
    tokenizer = ByteTokenizer.load(tokenizer_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    lengths: dict[str, int] = {}
    source_hashes: dict[str, str] = {}

    for split in ("train", "val"):
        input_path = input_dir / f"{split}.txt"
        if not input_path.exists():
            raise FileNotFoundError(
                f"Missing {input_path}. Create train.txt and val.txt first, "
                "or use format_chat.py for JSON data."
            )
        output_path = output_dir / f"{split}.bin"
        lengths[split] = encode_file(tokenizer, input_path, output_path)
        source_hashes[split] = sha256(input_path)

    metadata = {
        "format_version": 1,
        "dtype": "uint16",
        "byteorder": "little",
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": tokenizer_path.name,
        "lengths": lengths,
        "source_sha256": source_hashes,
    }
    (output_dir / "data_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    lengths = prepare(args.tokenizer, args.input_dir, args.output_dir)
    print(f"vocab size: {ByteTokenizer.load(args.tokenizer).vocab_size}")
    for split, length in lengths.items():
        print(f"{split}: {length} tokens")
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()

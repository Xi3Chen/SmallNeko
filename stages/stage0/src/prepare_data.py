"""Encode stage 0 text splits into uint16 binary token streams."""

from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path

from char_tokenizer import CharTokenizer


STAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = STAGE_ROOT / "data" / "processed" / "char_tokenizer.json"
DEFAULT_INPUT_DIR = STAGE_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = STAGE_ROOT / "data" / "processed"


def encode_file(tokenizer: CharTokenizer, input_path: Path, output_path: Path) -> int:
    text = input_path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)
    if tokenizer.vocab_size > 65536:
        raise ValueError("The uint16 data format supports at most 65536 vocabulary entries")
    if array("H").itemsize != 2:
        raise RuntimeError("This platform does not provide a 16-bit unsigned array type")

    encoded = array("H", token_ids)
    output_path.write_bytes(encoded.tobytes())
    return len(token_ids)


def prepare(
    tokenizer_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, int]:
    tokenizer = CharTokenizer.load(tokenizer_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    lengths: dict[str, int] = {}

    for split in ("train", "val"):
        input_path = input_dir / f"{split}.txt"
        output_path = output_dir / f"{split}.bin"
        lengths[split] = encode_file(tokenizer, input_path, output_path)

    metadata = {
        "format_version": 1,
        "dtype": "uint16",
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": tokenizer_path.name,
        "lengths": lengths,
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
    print(f"vocab size: {CharTokenizer.load(args.tokenizer).vocab_size}")
    for split, length in lengths.items():
        print(f"{split}: {length} tokens")
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()

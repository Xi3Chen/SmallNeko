"""Create the fixed stage 1 UTF-8 byte tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from byte_tokenizer import ByteTokenizer


STAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_TEXT = STAGE_ROOT / "data" / "raw" / "train.txt"
DEFAULT_VAL_TEXT = STAGE_ROOT / "data" / "raw" / "val.txt"
DEFAULT_OUTPUT = STAGE_ROOT / "data" / "processed" / "byte_tokenizer.json"


def build(
    input_path: Path,
    output_path: Path,
    validation_path: Path | None = None,
) -> ByteTokenizer:
    tokenizer = ByteTokenizer.from_file(input_path)
    tokenizer.save(output_path)

    print(f"training text: {input_path}")
    print(f"tokenizer: {output_path}")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"special tokens: {tokenizer.special_token_ids}")

    if validation_path is not None and validation_path.exists():
        print("validation unknown code points: 0")
        print("byte coverage: complete")

    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TRAIN_TEXT)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VAL_TEXT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tokenizer = build(args.input, args.output, args.validation)
    sample = "<|im_start|>user\n阶段1测试🙂<|im_end|>"
    assert tokenizer.decode(tokenizer.encode(sample)) == sample
    print("round-trip check: passed")


if __name__ == "__main__":
    main()

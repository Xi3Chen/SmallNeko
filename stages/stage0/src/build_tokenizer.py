"""Build the stage 0 character tokenizer from the training split."""

from __future__ import annotations

import argparse
from pathlib import Path

from char_tokenizer import CharTokenizer


STAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_TEXT = STAGE_ROOT / "data" / "raw" / "train.txt"
DEFAULT_VAL_TEXT = STAGE_ROOT / "data" / "raw" / "val.txt"
DEFAULT_OUTPUT = STAGE_ROOT / "data" / "processed" / "char_tokenizer.json"


def build(input_path: Path, output_path: Path, validation_path: Path | None = None) -> CharTokenizer:
    tokenizer = CharTokenizer.from_file(input_path)
    tokenizer.save(output_path)

    print(f"training text: {input_path}")
    print(f"tokenizer: {output_path}")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"special tokens: {tokenizer.special_token_ids}")

    if validation_path is not None:
        validation_text = validation_path.read_text(encoding="utf-8")
        unknown = tokenizer.unknown_characters(validation_text)
        print(f"validation unknown code points: {sum(unknown.values())}")
        print(f"validation unknown characters: {len(unknown)}")
        if unknown:
            preview = ", ".join(repr(character) for character, _ in unknown.most_common(12))
            print(f"unknown preview: {preview}")

    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TRAIN_TEXT)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VAL_TEXT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tokenizer = build(args.input, args.output, args.validation)
    sample = "<|im_start|>user\n阶段0测试<|im_end|>"
    assert tokenizer.decode(tokenizer.encode(sample)) == sample
    print("round-trip check: passed")


if __name__ == "__main__":
    main()

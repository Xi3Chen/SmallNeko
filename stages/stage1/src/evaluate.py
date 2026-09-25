"""Run a fixed prompt set and save reproducible baseline generations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from byte_tokenizer import ByteTokenizer
from generate import choose_device, answer
from model import GPT, GPTConfig


STAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = STAGE_ROOT / "data" / "processed"
DEFAULT_CHECKPOINT = STAGE_ROOT / "experiments" / "baseline-0.1b" / "best.pt"
DEFAULT_PROMPTS = STAGE_ROOT / "configs" / "prompts.json"
DEFAULT_OUTPUT = STAGE_ROOT / "experiments" / "baseline" / "fixed_prompts.jsonl"


def load_prompts(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Prompt file must contain a JSON array")
    prompts: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        if isinstance(item, str):
            prompts.append({"id": str(index), "prompt": item})
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            prompts.append({"id": str(item.get("id", index)), "prompt": item["prompt"]})
        else:
            raise ValueError(f"Invalid prompt entry at index {index}")
    if not prompts:
        raise ValueError("Prompt file cannot be empty")
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config") or checkpoint.get("config")
    if model_config is None:
        raise ValueError("Checkpoint does not contain a model config")
    config = GPTConfig(**model_config)
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = ByteTokenizer.load(args.data_dir / "byte_tokenizer.json")
    eos_id = tokenizer.stoi["<|im_end|>"]
    prompts = load_prompts(args.prompts)
    generation_args = SimpleNamespace(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, item in enumerate(prompts):
            seed = args.seed + index
            random.seed(seed)
            torch.manual_seed(seed)
            output = answer(model, tokenizer, item["prompt"], generation_args, eos_id)
            stream.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "prompt": item["prompt"],
                        "output": output,
                        "seed": seed,
                        "checkpoint": str(args.checkpoint),
                        "step": checkpoint.get("step"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(prompts)} generations to: {args.output}")


if __name__ == "__main__":
    main()

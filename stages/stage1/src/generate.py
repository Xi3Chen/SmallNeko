"""Generate replies from a stage 1 Transformer checkpoint."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from byte_tokenizer import ByteTokenizer
from model import GPT, GPTConfig


STAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = STAGE_ROOT / "data" / "processed"
DEFAULT_CHECKPOINT = STAGE_ROOT / "experiments" / "baseline-0.1b" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt", default=None, help="Question to answer; omit for interactive mode")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 selects greedy decoding")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


@torch.no_grad()
def generate(
    model: GPT,
    token_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_id: int,
) -> list[int]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if temperature < 0.0:
        raise ValueError("temperature cannot be negative")
    if top_k < 0:
        raise ValueError("top_k cannot be negative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    ids = torch.tensor([token_ids], dtype=torch.long, device=next(model.parameters()).device)
    for _ in range(max_new_tokens):
        context = ids[:, -model.config.block_size :]
        logits, _ = model(context)
        logits = logits[:, -1, :]
        if temperature == 0.0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = sorted_probs.cumsum(dim=-1)
                remove = cumulative_probs - sorted_probs > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_indices, sorted_logits
                )
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        ids = torch.cat((ids, next_id), dim=1)
        if int(next_id.item()) == eos_id:
            break
    return ids[0].tolist()


def answer(
    model: GPT,
    tokenizer: ByteTokenizer,
    question: str,
    args: argparse.Namespace,
    eos_id: int,
) -> str:
    prompt = f"<|im_start|>user\n{question.strip()}\n<|im_start|>assistant\n"
    token_ids = tokenizer.encode(prompt)
    output_ids = generate(
        model,
        token_ids,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        eos_id,
    )
    generated_ids = output_ids[len(token_ids) :]
    if eos_id in generated_ids:
        generated_ids = generated_ids[: generated_ids.index(eos_id)]
    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        drop_incomplete_tail=True,
    ).strip()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
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

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}; temperature: {args.temperature}; top-k: {args.top_k}; top-p: {args.top_p}")
    if args.prompt is not None:
        print(answer(model, tokenizer, args.prompt, args, eos_id))
        return
    print("输入问题，输入空行退出。")
    while True:
        question = input("\n你: ").strip()
        if not question:
            break
        print(f"模型: {answer(model, tokenizer, question, args, eos_id)}")


if __name__ == "__main__":
    main()

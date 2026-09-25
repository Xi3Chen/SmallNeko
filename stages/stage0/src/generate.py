"""Generate a reply from a stage 0 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from char_tokenizer import CharTokenizer
from model import GPT, GPTConfig


STAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = STAGE_ROOT / "data" / "processed"
DEFAULT_CHECKPOINT = STAGE_ROOT / "experiments" / "baseline" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt", default=None, help="Question to answer; omit for interactive mode")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def generate(model: GPT, token_ids: list[int], max_new_tokens: int, temperature: float, top_k: int) -> list[int]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ids = torch.tensor([token_ids], dtype=torch.long, device=next(model.parameters()).device)
    for _ in range(max_new_tokens):
        context = ids[:, -model.config.block_size :]
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        ids = torch.cat((ids, next_id), dim=1)
        if next_id.item() == model._eos_id:
            break
    return ids[0].tolist()


def answer(model: GPT, tokenizer: CharTokenizer, question: str, args: argparse.Namespace) -> str:
    prompt = f"<|im_start|>user\n{question}\n<|im_start|>assistant\n"
    token_ids = tokenizer.encode(prompt)
    output_ids = generate(model, token_ids, args.max_new_tokens, args.temperature, args.top_k)
    generated = tokenizer.decode(output_ids[len(token_ids) :], skip_special_tokens=True)
    return generated.strip()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model._eos_id = CharTokenizer.load(args.data_dir / "char_tokenizer.json").stoi["<|im_end|>"]
    model.eval()
    tokenizer = CharTokenizer.load(args.data_dir / "char_tokenizer.json")
    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}; temperature: {args.temperature}; top-k: {args.top_k}")
    if args.prompt is not None:
        print(answer(model, tokenizer, args.prompt, args))
        return
    print("输入问题，输入空行退出。")
    while True:
        question = input("\n你: ").strip()
        if not question:
            break
        print(f"模型: {answer(model, tokenizer, question, args)}")


if __name__ == "__main__":
    main()

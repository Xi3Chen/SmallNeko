"""Train the stage 0 Transformer baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from char_tokenizer import CharTokenizer
from dataset import load_stage0_data
from model import GPT, GPTConfig


STAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = STAGE_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = STAGE_ROOT / "experiments" / "baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data,
    val_data,
    batch_size: int,
    block_size: int,
    device: torch.device,
    eval_iters: int,
) -> dict[str, float]:
    model.eval()
    results: dict[str, float] = {}
    for name, dataset in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(eval_iters):
            inputs, targets = dataset.get_batch(batch_size, block_size, device)
            _, loss = model(inputs, targets)
            losses.append(loss.item())
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    metrics: dict[str, float],
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "metrics": metrics,
        "config": model.config_dict(),
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    tokenizer = CharTokenizer.load(args.data_dir / "char_tokenizer.json")
    train_data, val_data = load_stage0_data(args.data_dir)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_iteration = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["config"] != model.config_dict():
            raise ValueError("Checkpoint model config does not match the current training arguments")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = int(checkpoint["iteration"]) + 1
        best_val_loss = float(checkpoint.get("metrics", {}).get("val", best_val_loss))
        print(f"resumed from: {args.resume} at step {start_iteration}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(
        json.dumps(vars(args) | {"device": str(device)}, default=str, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"device: {device}")
    print(f"parameters: {model.num_parameters():,}")
    print(f"train tokens: {len(train_data):,}; val tokens: {len(val_data):,}")

    for iteration in range(start_iteration, args.max_iters + 1):
        if iteration % args.eval_interval == 0:
            metrics = estimate_loss(
                model,
                train_data,
                val_data,
                args.batch_size,
                args.block_size,
                device,
                args.eval_iters,
            )
            print(
                f"step {iteration:>6} | "
                f"train loss {metrics['train']:.4f} | val loss {metrics['val']:.4f}"
            )
            if metrics["val"] < best_val_loss:
                best_val_loss = metrics["val"]
                save_checkpoint(args.out_dir / "best.pt", model, optimizer, iteration, metrics)

        if iteration == args.max_iters:
            break
        inputs, targets = train_data.get_batch(args.batch_size, args.block_size, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"best checkpoint: {args.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

"""Train a reproducible standard decoder-only Transformer baseline."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from byte_tokenizer import ByteTokenizer
from dataset import TokenDataset, load_data
from model import GPT, GPTConfig


STAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = STAGE_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = STAGE_ROOT / "experiments" / "baseline-0.1b"
DEFAULT_TRAINING_CONFIG = STAGE_ROOT / "configs" / "baseline.json"


def load_training_defaults(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Training config must be a JSON object: {path}")
    return {key: value for key, value in payload.items() if key != "description"}


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    config_args, _ = config_parser.parse_known_args()
    config_defaults = load_training_defaults(config_args.config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=config_args.config,
        help="JSON defaults file; command-line options take precedence",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=5000, help="Number of optimizer updates")
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=250, help="Save last.pt every N optimizer updates")
    parser.add_argument("--resume", type=Path, default=None)
    parser.set_defaults(**config_defaults)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = ("batch_size", "block_size", "n_layer", "n_head", "n_embd", "max_iters", "eval_interval", "eval_iters", "log_interval", "save_interval")
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay cannot be negative")
    if args.grad_clip < 0.0:
        raise ValueError("--grad-clip cannot be negative")


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # These settings make the baseline easier to reproduce across runs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_sampler(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: TokenDataset,
    val_data: TokenDataset,
    batch_size: int,
    block_size: int,
    device: torch.device,
    eval_iters: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    results: dict[str, float] = {}
    for name, dataset, offset in (
        ("train", train_data, 0),
        ("val", val_data, 1),
    ):
        sampler = make_sampler(seed + 100_000 + offset)
        losses: list[float] = []
        for _ in range(eval_iters):
            inputs, targets = dataset.get_batch(
                batch_size,
                block_size,
                device,
                generator=sampler,
            )
            _, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("The model did not return an evaluation loss")
            losses.append(float(loss.item()))
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


def current_rng_state(sampler: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "sampler": sampler.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], sampler: torch.Generator) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"]
    if isinstance(torch_state, torch.Tensor):
        torch_state = torch_state.cpu()
    torch.set_rng_state(torch_state)
    sampler_state = state["sampler"]
    if isinstance(sampler_state, torch.Tensor):
        sampler_state = sampler_state.cpu()
    sampler.set_state(sampler_state)
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = [
            value.cpu() if isinstance(value, torch.Tensor) else value
            for value in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, float],
    sampler: torch.Generator,
    metadata: dict[str, Any],
    best_val_loss: float,
) -> None:
    checkpoint = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "metrics": metrics,
        "best_val_loss": best_val_loss,
        "model_config": model.config_dict(),
        "config": model.config_dict(),  # compatibility with the stage 0 loader shape
        "metadata": metadata,
        "rng_state": current_rng_state(sampler),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    tokenizer = ByteTokenizer.load(args.data_dir / "byte_tokenizer.json")
    train_data, val_data = load_data(args.data_dir)
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
    sampler = make_sampler(args.seed + 1)
    start_step = 0
    best_val_loss = float("inf")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved_config = checkpoint.get("model_config", checkpoint.get("config"))
        if saved_config != model.config_dict():
            raise ValueError(
                "Checkpoint model config does not match the current training arguments: "
                f"saved={saved_config}, current={model.config_dict()}"
            )
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", checkpoint.get("iteration", -1) + 1))
        best_val_loss = float(
            checkpoint.get(
                "best_val_loss",
                checkpoint.get("metrics", {}).get("val", best_val_loss),
            )
        )
        restore_rng_state(checkpoint.get("rng_state", {}), sampler)
        print(f"resumed from: {args.resume} at step {start_step}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "stage": 1,
        "seed": args.seed,
        "device": str(device),
        "tokenizer": "byte_tokenizer.json",
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "parameters": model.num_parameters(),
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "effective_batch_tokens": args.batch_size * args.block_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    write_json(
        args.out_dir / "config.json",
        {**vars(args), "device": str(device), "model_config": model.config_dict(), "metadata": metadata},
    )

    print(f"device: {device}")
    print(f"parameters: {model.num_parameters():,}")
    print(f"train tokens: {len(train_data):,}; val tokens: {len(val_data):,}")
    if start_step >= args.max_iters:
        print(f"checkpoint is already at step {start_step}; max-iters is {args.max_iters}")
        return

    log_path = args.out_dir / "metrics.jsonl"
    wall_start = time.perf_counter()
    last_metrics: dict[str, float] = {}

    for step in range(start_step, args.max_iters):
        if step % args.eval_interval == 0:
            metrics = estimate_loss(
                model,
                train_data,
                val_data,
                args.batch_size,
                args.block_size,
                device,
                args.eval_iters,
                args.seed + step,
            )
            metrics["step"] = float(step)
            metrics["elapsed_seconds"] = time.perf_counter() - wall_start
            last_metrics = metrics
            print(
                f"step {step:>6} | train loss {metrics['train']:.4f} | "
                f"val loss {metrics['val']:.4f}"
            )
            append_jsonl(log_path, metrics)
            if metrics["val"] < best_val_loss:
                best_val_loss = metrics["val"]
                save_checkpoint(
                    args.out_dir / "best.pt",
                    model,
                    optimizer,
                    step,
                    metrics,
                    sampler,
                    metadata,
                    best_val_loss,
                )

        inputs, targets = train_data.get_batch(
            args.batch_size,
            args.block_size,
            device,
            generator=sampler,
        )
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("The model did not return a training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        completed_step = step + 1
        if completed_step % args.save_interval == 0:
            checkpoint_metrics = {**last_metrics, "step": float(completed_step)}
            save_checkpoint(
                args.out_dir / "last.pt",
                model,
                optimizer,
                completed_step,
                checkpoint_metrics,
                sampler,
                metadata,
                best_val_loss,
            )
        if completed_step % args.log_interval == 0:
            print(f"step {completed_step:>6} | batch loss {loss.item():.4f}")

    final_metrics = estimate_loss(
        model,
        train_data,
        val_data,
        args.batch_size,
        args.block_size,
        device,
        args.eval_iters,
        args.seed + args.max_iters,
    )
    final_metrics["step"] = float(args.max_iters)
    final_metrics["elapsed_seconds"] = time.perf_counter() - wall_start
    append_jsonl(log_path, final_metrics)
    save_checkpoint(
        args.out_dir / "last.pt",
        model,
        optimizer,
        args.max_iters,
        final_metrics,
        sampler,
        metadata,
        best_val_loss,
    )
    if final_metrics["val"] < best_val_loss:
        save_checkpoint(
            args.out_dir / "best.pt",
            model,
            optimizer,
            args.max_iters,
            final_metrics,
            sampler,
            metadata,
            final_metrics["val"],
        )
    print(f"last checkpoint: {args.out_dir / 'last.pt'}")
    print(f"best checkpoint: {args.out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

"""Convert common JSON/JSONL chat records into stage 1 training text."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("data", payload.get("records", [payload]))
        else:
            raise ValueError("JSON input must be an object or an array")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every JSON record must be an object")
    return records


def content_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts).strip()
    return ""


def conversation_text(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        rendered: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user")).strip() or "user"
            content = content_text(message.get("content"))
            if content:
                rendered.append(f"<|im_start|>{role}\n{content}\n")
        if rendered:
            return "".join(rendered) + "<|im_end|>\n"

    conversations = record.get("conversations")
    if isinstance(conversations, list):
        rendered = []
        role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}
        for message in conversations:
            if not isinstance(message, dict):
                continue
            role = role_map.get(str(message.get("from", message.get("role", "user"))).lower(), "user")
            content = content_text(message.get("value", message.get("content")))
            if content:
                rendered.append(f"<|im_start|>{role}\n{content}\n")
        if rendered:
            return "".join(rendered) + "<|im_end|>\n"

    pairs = (
        ("instruction", "output"),
        ("question", "answer"),
        ("prompt", "response"),
        ("input", "output"),
    )
    instruction = content_text(record.get("instruction"))
    extra_input = content_text(record.get("input"))
    assistant = content_text(record.get("output"))
    if instruction and extra_input and assistant:
        return (
            f"<|im_start|>user\n{instruction}\n{extra_input}\n"
            f"<|im_start|>assistant\n{assistant}\n<|im_end|>\n"
        )
    for user_key, assistant_key in pairs:
        user = content_text(record.get(user_key))
        assistant = content_text(record.get(assistant_key))
        if user and assistant:
            return (
                f"<|im_start|>user\n{user}\n"
                f"<|im_start|>assistant\n{assistant}\n"
                "<|im_end|>\n"
            )

    text = content_text(record.get("text"))
    if text:
        return text + ("\n" if not text.endswith("\n") else "")
    raise ValueError(f"Could not find supported text fields in record: {record.keys()}")


def format_records(records: list[dict[str, Any]]) -> tuple[list[str], int]:
    rendered: list[str] = []
    seen: set[str] = set()
    skipped = 0
    for record in records:
        try:
            text = conversation_text(record)
        except ValueError:
            skipped += 1
            continue
        if not text.strip():
            skipped += 1
            continue
        user_match = re.search(
            r"<\|im_start\|>user\n(.*?)\n<\|im_start\|>assistant\n",
            text,
            flags=re.DOTALL,
        )
        identity = "user:" + " ".join(user_match.group(1).split()) if user_match else "text:" + " ".join(text.split())
        if identity in seen:
            skipped += 1
            continue
        seen.add(identity)
        rendered.append(text)
    return rendered, skipped


def write_split(path: Path, records: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(records), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    records = load_records(args.input)
    formatted, skipped = format_records(records)
    if len(formatted) < 2:
        raise ValueError("At least two non-empty, non-duplicate records are required")
    random.Random(args.seed).shuffle(formatted)
    val_count = min(len(formatted) - 1, max(1, round(len(formatted) * args.val_ratio)))
    write_split(args.output_dir / "train.txt", formatted[val_count:])
    write_split(args.output_dir / "val.txt", formatted[:val_count])
    report = {
        "input": str(args.input),
        "records_in": len(records),
        "records_written": len(formatted),
        "records_skipped": skipped,
        "train_records": len(formatted) - val_count,
        "val_records": val_count,
        "seed": args.seed,
    }
    (args.output_dir / "format_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

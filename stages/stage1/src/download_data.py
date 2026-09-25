"""Download a small, reproducible stage 1 data mixture from public sources.

The Hugging Face Dataset Viewer rows API is used instead of downloading full
parquet shards. This intentionally downloads only the configured budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


STAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = STAGE_ROOT / "configs" / "data-mixture.json"
DEFAULT_OUTPUT_DIR = STAGE_ROOT / "data" / "raw"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--total-bytes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("sources"), list):
        raise ValueError("Mixture config must contain a sources array")
    sources = config["sources"]
    if not sources:
        raise ValueError("Mixture config must contain at least one source")
    weights = [float(source.get("weight", 0.0)) for source in sources]
    if any(weight <= 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Source weights must be positive and sum to 1")
    if not 0.0 < float(config.get("validation_ratio", 0.1)) < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    return config


def fetch_rows(
    source: dict[str, Any],
    offset: int,
    length: int,
    timeout: int,
    request_delay: float,
    retries: int = 5,
) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "dataset": source["dataset"],
            "config": source["config"],
            "split": source.get("split", "train"),
            "offset": offset,
            "length": length,
        }
    )
    request = Request(
        f"{ROWS_ENDPOINT}?{query}",
        headers={"User-Agent": "SmallNeko-stage1/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            rows = payload.get("rows", [])
            return [item.get("row", {}) for item in rows if isinstance(item, dict)]
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < retries:
                retry_after = 0.0
                if isinstance(error, HTTPError):
                    try:
                        retry_after = float(error.headers.get("Retry-After", "0"))
                    except (TypeError, ValueError):
                        retry_after = 0.0
                time.sleep(max(request_delay, retry_after, min(2**attempt, 16)))
    raise RuntimeError(f"Could not fetch rows for {source['name']}: {last_error}") from last_error


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) < 40:
        return ""
    return text


def document_text(name: str, text: str) -> str:
    return f"<|im_start|>document:{name}\n{text}\n<|im_end|>\n"


def collect_hf_source(
    source: dict[str, Any],
    target_bytes: int,
    offset: int,
    batch_size: int,
    timeout: int,
    request_delay: float,
) -> tuple[list[str], dict[str, Any]]:
    records: list[str] = []
    seen: set[str] = set()
    total_bytes = 0
    skipped = 0
    current_offset = offset
    while total_bytes < target_bytes:
        rows = fetch_rows(
            source,
            current_offset,
            min(batch_size, 100),
            timeout,
            request_delay,
        )
        if not rows:
            break
        current_offset += len(rows)
        for row in rows:
            text = clean_text(row.get(source.get("text_field", "text")))
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest() if text else ""
            if not text or digest in seen:
                skipped += 1
                continue
            rendered = document_text(source["name"], text)
            records.append(rendered)
            seen.add(digest)
            total_bytes += len(rendered.encode("utf-8"))
            if total_bytes >= target_bytes:
                break
        if len(rows) < min(batch_size, 100):
            break
    return records, {
        "name": source["name"],
        "kind": source["kind"],
        "target_bytes": target_bytes,
        "actual_bytes": total_bytes,
        "records": len(records),
        "skipped": skipped,
        "start_offset": offset,
        "end_offset": current_offset,
    }


def collect_local_chat(
    source: dict[str, Any],
    target_bytes: int,
    validation: bool,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    source_path = (STAGE_ROOT / source["path"]).resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    from format_chat import format_records, load_records

    records, skipped = format_records(load_records(source_path))
    random.Random(seed + (1 if validation else 0)).shuffle(records)
    selected: list[str] = []
    actual_bytes = 0
    for record in records:
        if actual_bytes >= target_bytes:
            break
        selected.append(record)
        actual_bytes += len(record.encode("utf-8"))
    return selected, {
        "name": source["name"],
        "kind": source["kind"],
        "target_bytes": target_bytes,
        "actual_bytes": actual_bytes,
        "records": len(selected),
        "skipped": skipped,
    }


def write_records(path: Path, records: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(records), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    config = load_config(args.config)
    total_target = int(args.total_bytes or config["total_target_bytes"])
    seed = int(args.seed if args.seed is not None else config.get("seed", 1337))
    validation_ratio = float(config.get("validation_ratio", 0.1))
    train_records: list[str] = []
    val_records: list[str] = []
    report: dict[str, Any] = {
        "config": str(args.config),
        "total_target_bytes": total_target,
        "validation_ratio": validation_ratio,
        "seed": seed,
        "sources": [],
    }

    for index, source in enumerate(config["sources"]):
        total_source_target = max(1, round(total_target * float(source["weight"])))
        train_target = max(1, round(total_source_target * (1.0 - validation_ratio)))
        val_target = max(1, total_source_target - train_target)
        if source["kind"] == "hf_rows":
            train, train_info = collect_hf_source(
                source,
                train_target,
                0,
                args.batch_size,
                args.timeout,
                args.request_delay,
            )
            val, val_info = collect_hf_source(
                source,
                val_target,
                int(source.get("validation_offset", 100000)),
                args.batch_size,
                args.timeout,
                args.request_delay,
            )
        elif source["kind"] == "local_json":
            train, train_info = collect_local_chat(source, train_target, False, seed + index)
            val, val_info = collect_local_chat(source, val_target, True, seed + index)
        else:
            raise ValueError(f"Unsupported source kind: {source['kind']}")
        train_records.extend(train)
        val_records.extend(val)
        report["sources"].append({"train": train_info, "val": val_info, "weight": source["weight"]})
        print(
            f"{source['name']}: train {train_info['actual_bytes']:,} bytes / "
            f"val {val_info['actual_bytes']:,} bytes; "
            f"records {train_info['records']} / {val_info['records']}"
        )

    random.Random(seed).shuffle(train_records)
    random.Random(seed + 1).shuffle(val_records)
    write_records(args.output_dir / "train.txt", train_records)
    write_records(args.output_dir / "val.txt", val_records)
    report["actual_train_bytes"] = sum(len(record.encode("utf-8")) for record in train_records)
    report["actual_val_bytes"] = sum(len(record.encode("utf-8")) for record in val_records)
    report["train_records"] = len(train_records)
    report["val_records"] = len(val_records)
    (args.output_dir / "download_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"train: {report['actual_train_bytes']:,} bytes, {len(train_records)} records")
    print(f"val: {report['actual_val_bytes']:,} bytes, {len(val_records)} records")
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()

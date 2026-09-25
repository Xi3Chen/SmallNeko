"""Download selected public parquet shards and materialize stage 1 text.

This path is used when Dataset Viewer pagination is rate-limited. It reads
parquet rows incrementally, so only selected text and a small row batch stay
in memory. Downloaded parquet files remain in data/downloads and are ignored
by git; the resulting train/val text is written to data/raw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pyarrow.parquet as pq


STAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = STAGE_ROOT / "configs" / "data-mixture.json"
DEFAULT_OUTPUT_DIR = STAGE_ROOT / "data" / "raw"
DEFAULT_DOWNLOAD_DIR = STAGE_ROOT / "data" / "downloads"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--total-bytes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-rows", type=int, default=128)
    parser.add_argument(
        "--mirror-url",
        default="https://hf-mirror.com",
        help="Mirror for Hugging Face downloads; use an empty value for the official URL",
    )
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"using existing download: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "SmallNeko-stage1/1.0"})
    with urlopen(request, timeout=120) as response, temporary.open("wb") as target:
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            total += len(chunk)
            if total % (50 * 1024 * 1024) < len(chunk):
                print(f"downloaded {total / 1024 / 1024:.0f} MiB: {destination.name}")
    temporary.replace(destination)


def download_url(url: str, mirror_url: str) -> str:
    if mirror_url and url.startswith("https://huggingface.co/"):
        return mirror_url.rstrip("/") + url[len("https://huggingface.co") :]
    return url


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return text if len(text) >= 40 else ""


def render(name: str, text: str) -> str:
    return f"<|im_start|>document:{name}\n{text}\n<|im_end|>\n"


def collect_parquet(
    path: Path,
    source: dict[str, Any],
    target_bytes: int,
    validation: bool,
    seed: int,
    batch_rows: int,
) -> tuple[list[str], dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    text_field = source.get("text_field", "text")
    records: list[str] = []
    seen: set[str] = set()
    actual_bytes = 0
    skipped = 0
    row_number = 0
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=[text_field]):
        values = batch.column(text_field).to_pylist()
        indexed = list(enumerate(values, start=row_number))
        random.Random(seed + row_number + (1 if validation else 0)).shuffle(indexed)
        for index, value in indexed:
            text = clean_text(value)
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest() if text else ""
            if not text or digest in seen:
                skipped += 1
                continue
            # Assign duplicate documents to the same split, even when the
            # source parquet contains repeated rows at different indices.
            is_validation_row = int(digest[:8], 16) % 10 == 0
            if is_validation_row != validation:
                # Deterministically reserve about one tenth of documents for val.
                skipped += 1
                continue
            record = render(source["name"], text)
            records.append(record)
            seen.add(digest)
            actual_bytes += len(record.encode("utf-8"))
            if actual_bytes >= target_bytes:
                break
        row_number += len(values)
        if actual_bytes >= target_bytes:
            break
    return records, {
        "name": source["name"],
        "kind": source["kind"],
        "download": str(path),
        "target_bytes": target_bytes,
        "actual_bytes": actual_bytes,
        "records": len(records),
        "skipped": skipped,
        "parquet_rows_read": row_number,
    }


def collect_local_chat(source: dict[str, Any], target_bytes: int, validation: bool, seed: int) -> tuple[list[str], dict[str, Any]]:
    source_path = (STAGE_ROOT / source["path"]).resolve()
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


def select_local_records(records: list[str], target_bytes: int) -> list[str]:
    selected: list[str] = []
    actual_bytes = 0
    for record in records:
        if actual_bytes >= target_bytes:
            break
        selected.append(record)
        actual_bytes += len(record.encode("utf-8"))
    return selected


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    total_target = int(args.total_bytes or config["total_target_bytes"])
    seed = int(args.seed if args.seed is not None else config.get("seed", 1337))
    validation_ratio = float(config.get("validation_ratio", 0.1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.download_dir.mkdir(parents=True, exist_ok=True)
    train_records: list[str] = []
    val_records: list[str] = []
    local_cache: dict[str, tuple[list[str], list[str], int]] = {}
    report: dict[str, Any] = {
        "config": str(args.config),
        "total_target_bytes": total_target,
        "validation_ratio": validation_ratio,
        "seed": seed,
        "sources": [],
    }
    for source_index, source in enumerate(config["sources"]):
        source_target = max(1, round(total_target * float(source["weight"])))
        train_target = max(1, round(source_target * (1.0 - validation_ratio)))
        val_target = max(1, source_target - train_target)
        if source["kind"] == "parquet":
            filename = Path(source["url"].split("/")[-1].split("?")[0])
            local_path = args.download_dir / filename
            download(download_url(source["url"], args.mirror_url), local_path)
            train, train_info = collect_parquet(local_path, source, train_target, False, seed + source_index, args.batch_rows)
            val, val_info = collect_parquet(local_path, source, val_target, True, seed + source_index, args.batch_rows)
        elif source["kind"] == "local_json":
            source_path = str((STAGE_ROOT / source["path"]).resolve())
            if source_path not in local_cache:
                from format_chat import format_records, load_records

                formatted, skipped = format_records(load_records(Path(source_path)))
                random.Random(seed + source_index).shuffle(formatted)
                val_count = max(1, round(len(formatted) * validation_ratio))
                local_cache[source_path] = (formatted[val_count:], formatted[:val_count], skipped)
            train_pool, val_pool, skipped = local_cache[source_path]
            train = select_local_records(train_pool, train_target)
            val = select_local_records(val_pool, val_target)
            train_info = {
                "name": source["name"], "kind": source["kind"],
                "target_bytes": train_target,
                "actual_bytes": sum(len(record.encode("utf-8")) for record in train),
                "records": len(train), "skipped": skipped,
            }
            val_info = {
                "name": source["name"], "kind": source["kind"],
                "target_bytes": val_target,
                "actual_bytes": sum(len(record.encode("utf-8")) for record in val),
                "records": len(val), "skipped": skipped,
            }
        else:
            raise ValueError(f"Unsupported source kind: {source['kind']}")
        train_records.extend(train)
        val_records.extend(val)
        report["sources"].append({"weight": source["weight"], "train": train_info, "val": val_info})
        print(f"{source['name']}: train {train_info['actual_bytes']:,} / val {val_info['actual_bytes']:,} bytes")

    random.Random(seed).shuffle(train_records)
    random.Random(seed + 1).shuffle(val_records)
    (args.output_dir / "train.txt").write_text("".join(train_records), encoding="utf-8")
    (args.output_dir / "val.txt").write_text("".join(val_records), encoding="utf-8")
    report["actual_train_bytes"] = sum(len(record.encode("utf-8")) for record in train_records)
    report["actual_val_bytes"] = sum(len(record.encode("utf-8")) for record in val_records)
    report["train_records"] = len(train_records)
    report["val_records"] = len(val_records)
    (args.output_dir / "download_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"train: {report['actual_train_bytes']:,} bytes, {len(train_records)} records")
    print(f"val: {report['actual_val_bytes']:,} bytes, {len(val_records)} records")


if __name__ == "__main__":
    main()

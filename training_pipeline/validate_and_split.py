"""Validate a source dataset and create leakage-resistant train/val/golden sets."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from training_pipeline.pipeline_common import (
        iter_jsonl,
        load_schema,
        sha256_jsonl,
        utc_now,
        validate_schema,
        write_json_atomic,
        write_jsonl,
    )
except ImportError:
    from pipeline_common import (
        iter_jsonl,
        load_schema,
        sha256_jsonl,
        utc_now,
        validate_schema,
        write_json_atomic,
        write_jsonl,
    )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--schema-dir", type=Path, default=None)
    parser.add_argument("--min-golden", type=int, default=10)
    parser.add_argument("--allow-small", action="store_true")
    return parser.parse_args()
def group_key(record: dict[str, Any], fallback_index: int) -> str:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("document_id", "source_document_id", "group_id"):
            if metadata.get(key) is not None:
                return str(metadata[key])
    # A stable content group is safer than a random row split when metadata is absent.
    payload = json.dumps(
        {"input": record.get("input"), "output": record.get("output")},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest() or str(fallback_index)
def _validate_record(
    record: dict[str, Any],
    line_no: int,
    schema_dir: Path | None,
) -> tuple[bool, str | None]:
    if not isinstance(record.get("input"), str) or not record["input"].strip():
        return False, "empty_input"
    output = record.get("output")
    if not isinstance(output, dict) or not output:
        return False, "output_not_non_empty_object"
    metadata = record.get("metadata")
    document_type = metadata.get("document_type") if isinstance(metadata, dict) else None
    schema = load_schema(schema_dir, str(document_type) if document_type else None)
    try:
        errors = validate_schema(output, schema)
    except RuntimeError as exc:
        return False, str(exc)
    if errors:
        return False, "schema: " + "; ".join(errors[:3])
    if not record.get("instruction"):
        return False, "missing_instruction"
    return True, None
def _allocate_groups(
    groups: dict[str, list[dict[str, Any]]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    total = sum(len(groups[key]) for key in keys)
    targets = [total * train_ratio, total * val_ratio, total * (1 - train_ratio - val_ratio)]
    buckets: list[list[dict[str, Any]]] = [[], [], []]
    counts = [0, 0, 0]
    # Seed every split with one independent group.  Without this guard, a
    # three-group dataset can be assigned entirely to train by the deficit
    # heuristic, leaving validation or golden empty.
    for bucket_index, key in enumerate(keys[:3]):
        buckets[bucket_index].extend(groups[key])
        counts[bucket_index] += len(groups[key])
    for key in keys[3:]:
        group = groups[key]
        # Put the next group into the split furthest below its target.
        deficits = [targets[i] - counts[i] for i in range(3)]
        bucket_index = max(range(3), key=lambda index: deficits[index])
        buckets[bucket_index].extend(group)
        counts[bucket_index] += len(group)
    return buckets[0], buckets[1], buckets[2]
def validate_and_split(args: argparse.Namespace) -> Path:
    source_file = args.dataset / "train_source.jsonl"
    if not source_file.exists():
        raise FileNotFoundError(f"Missing source dataset: {source_file}")
    if not args.dataset.is_dir():
        raise NotADirectoryError(args.dataset)
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_group_inputs: set[tuple[str, str]] = set()
    for line_no, record in iter_jsonl(source_file):
        ok, reason = _validate_record(record, line_no, args.schema_dir)
        if not ok:
            rejected.append({"line": line_no, "reason": reason})
            continue
        key = group_key(record, line_no)
        input_key = (key, str(record["input"]).strip())
        if input_key in seen_group_inputs:
            rejected.append({"line": line_no, "reason": "duplicate_input_in_group"})
            continue
        seen_group_inputs.add(input_key)
        valid.append(record)
        groups[key].append(record)
    if len(groups) < 3:
        raise RuntimeError(
            "At least 3 independent document groups are required; "
            "the old fallback that duplicated records across splits is disabled."
        )
    train, validation, golden = _allocate_groups(
        groups, args.train_ratio, args.val_ratio, args.seed
    )
    if len(golden) < args.min_golden and not args.allow_small:
        raise RuntimeError(
            f"Golden set has {len(golden)} records; minimum is {args.min_golden}. "
            "Use --allow-small only for a local smoke test."
        )
    if not train or not validation or not golden:
        raise RuntimeError("Train, validation and golden sets must all be non-empty")
    write_jsonl(args.dataset / "train.jsonl", train)
    write_jsonl(args.dataset / "val.jsonl", validation)
    write_jsonl(args.dataset / "golden.jsonl", golden)
    report = {
        "created_at": utc_now(),
        "source_sha256": sha256_jsonl(source_file),
        "total_source": len(valid) + len(rejected),
        "total_valid": len(valid),
        "rejected_count": len(rejected),
        "rejected_details": rejected,
        "train_count": len(train),
        "val_count": len(validation),
        "golden_count": len(golden),
        "seed": args.seed,
        "ratios_requested": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "golden": 1 - args.train_ratio - args.val_ratio,
        },
        "group_count": len(groups),
        "pipeline_stage": "SPLIT",
    }
    write_json_atomic(args.dataset / "validation_report.json", report)
    manifest = {
        "dataset_version": args.dataset.name,
        "source_sha256": report["source_sha256"],
        "train_sha256": sha256_jsonl(args.dataset / "train.jsonl"),
        "val_sha256": sha256_jsonl(args.dataset / "val.jsonl"),
        "golden_sha256": sha256_jsonl(args.dataset / "golden.jsonl"),
        "golden_frozen": True,
        "seed": args.seed,
        "created_at": report["created_at"],
        "pipeline_stage": "SPLIT",
    }
    write_json_atomic(args.dataset / "split_manifest.json", manifest)
    print(
        f"Split {args.dataset}: train={len(train)}, val={len(validation)}, "
        f"golden={len(golden)}, rejected={len(rejected)}"
    )
    return args.dataset
def main() -> int:
    try:
        validate_and_split(parse_args())
        return 0
    except Exception as exc:
        print(f"VALIDATION_FAILED: {exc}", file=sys.stderr)
        return 1
if __name__ == "__main__":
    raise SystemExit(main())
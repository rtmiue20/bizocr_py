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
    parser.add_argument(
        "--golden-ids",
        type=str,
        default=None,
        help="Path to golden set dir/file or comma-separated IDs",
    )
    parser.add_argument(
        "--allow-random-golden",
        action="store_true",
        help="Explicitly allow randomly partitioning a golden set from source data if no frozen golden exists",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--schema-dir", type=Path, default=None)
    parser.add_argument("--min-golden", type=int, default=50)
    parser.add_argument("--allow-small", action="store_true")
    return parser.parse_args()


def content_or_image_hash(record: dict[str, Any]) -> str:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for img_key in ("image_hash", "image_sha256", "content_hash"):
            val = metadata.get(img_key)
            if val:
                return f"img_{val}"
    raw_input = str(record.get("input") or "").strip()
    normalized_input = " ".join(raw_input.split())
    return hashlib.sha256(normalized_input.encode("utf-8")).hexdigest()


def group_key(record: dict[str, Any], fallback_index: int) -> str:
    # Group by content/image hash to strictly prevent identical or duplicate cards leaking across splits
    chash = content_or_image_hash(record)
    if chash:
        return chash
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("group_id", "document_id", "source_document_id"):
            if metadata.get(key) is not None:
                return str(metadata[key])
    return str(fallback_index)


def load_golden_references(
    golden_ids_arg: str | None,
) -> tuple[set[str], list[dict[str, Any]], set[str], bool]:
    """Returns (golden_id_set, golden_records, golden_content_hashes, source_specified)."""
    id_set: set[str] = set()
    golden_records: list[dict[str, Any]] = []
    golden_content_hashes: set[str] = set()
    source_specified = False

    golden_path: Path | None = None
    if golden_ids_arg:
        source_specified = True
        p = Path(golden_ids_arg)
        if p.exists():
            golden_path = p
        else:
            for val in golden_ids_arg.split(","):
                val = val.strip()
                if val:
                    id_set.add(val)
    else:
        for candidate in (Path("golden"), Path("datasets/golden")):
            if candidate.is_dir():
                golden_path = candidate
                source_specified = True
                break

    if golden_path is not None:
        if golden_path.is_file():
            if golden_path.suffix == ".jsonl":
                for _, rec in iter_jsonl(golden_path):
                    golden_records.append(rec)
                    doc_id = (
                        rec.get("metadata", {}).get("document_id")
                        if isinstance(rec.get("metadata"), dict)
                        else rec.get("document_id")
                    )
                    if doc_id is not None:
                        id_set.add(str(doc_id))
                    chash = content_or_image_hash(rec)
                    if chash:
                        golden_content_hashes.add(chash)
            elif golden_path.suffix == ".json":
                try:
                    data = json.loads(golden_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        doc_ids = data.get("document_ids", [])
                        if isinstance(doc_ids, list):
                            id_set |= {str(i) for i in doc_ids if i is not None}
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, (str, int)):
                                id_set.add(str(item))
                except Exception:
                    pass
            else:
                for line in golden_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        id_set.add(line)

        elif golden_path.is_dir():
            # 1. Load document_ids from golden_manifest.json if present
            mf = golden_path / "golden_manifest.json"
            if mf.is_file():
                try:
                    manifest_data = json.loads(mf.read_text(encoding="utf-8"))
                    if isinstance(manifest_data, dict):
                        doc_ids = manifest_data.get("document_ids", [])
                        if isinstance(doc_ids, list):
                            id_set |= {str(i) for i in doc_ids if i is not None}
                except Exception:
                    pass

            # 2. Load golden records ONLY from golden*.jsonl files
            for jsonl_file in sorted(golden_path.glob("golden*.jsonl")):
                for _, rec in iter_jsonl(jsonl_file):
                    golden_records.append(rec)
                    doc_id = (
                        rec.get("metadata", {}).get("document_id")
                        if isinstance(rec.get("metadata"), dict)
                        else rec.get("document_id")
                    )
                    if doc_id is not None:
                        id_set.add(str(doc_id))
                    chash = content_or_image_hash(rec)
                    if chash:
                        golden_content_hashes.add(chash)

    return id_set, golden_records, golden_content_hashes, source_specified


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


def _allocate_groups_two_way(
    groups: dict[str, list[dict[str, Any]]],
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    total = sum(len(groups[key]) for key in keys)
    target_train = total * train_ratio

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    train_count = 0

    if len(keys) >= 2:
        train.extend(groups[keys[0]])
        train_count += len(groups[keys[0]])
        val.extend(groups[keys[1]])
        remaining_keys = keys[2:]
    else:
        remaining_keys = keys

    for key in remaining_keys:
        group = groups[key]
        if train_count < target_train:
            train.extend(group)
            train_count += len(group)
        else:
            val.extend(group)

    return train, val


def _allocate_groups_three_way(
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
    for bucket_index, key in enumerate(keys[:3]):
        buckets[bucket_index].extend(groups[key])
        counts[bucket_index] += len(groups[key])
    for key in keys[3:]:
        group = groups[key]
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

    golden_ids, frozen_golden_records, golden_hashes, golden_source_specified = load_golden_references(
        args.golden_ids
    )

    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    golden_isolated_records: list[dict[str, Any]] = list(frozen_golden_records)
    seen_group_inputs: set[tuple[str, str]] = set()
    leaked_golden_exclusions = 0

    for line_no, record in iter_jsonl(source_file):
        ok, reason = _validate_record(record, line_no, args.schema_dir)
        if not ok:
            rejected.append({"line": line_no, "reason": reason})
            continue

        doc_id = str(record.get("metadata", {}).get("document_id") or "")
        chash = content_or_image_hash(record)

        # Check if record belongs to frozen golden or leaks golden content
        is_golden_doc = bool(doc_id and doc_id in golden_ids)
        is_golden_hash = bool(chash in golden_hashes)

        if is_golden_doc or is_golden_hash:
            leaked_golden_exclusions += 1
            if not any(
                r.get("metadata", {}).get("document_id") == doc_id
                for r in golden_isolated_records
                if doc_id
            ):
                golden_isolated_records.append(record)
            continue

        key = group_key(record, line_no)
        input_key = (key, str(record["input"]).strip())
        if input_key in seen_group_inputs:
            rejected.append({"line": line_no, "reason": "duplicate_input_in_group"})
            continue
        seen_group_inputs.add(input_key)
        valid.append(record)
        groups[key].append(record)

    has_frozen_golden = len(golden_isolated_records) > 0

    if has_frozen_golden:
        if len(groups) < 2 and not args.allow_small:
            raise RuntimeError(
                f"At least 2 independent document groups are required for train/val; found {len(groups)}."
            )
        normalized_train_ratio = (
            args.train_ratio / (args.train_ratio + args.val_ratio)
            if (args.train_ratio + args.val_ratio) > 0
            else 0.8
        )
        train, validation = _allocate_groups_two_way(groups, normalized_train_ratio, args.seed)
        golden = golden_isolated_records
    else:
        if golden_source_specified:
            raise RuntimeError(
                "Golden set directory or --golden-ids was specified/found, but no golden records or matching document IDs were loaded. "
                "Please check your golden files or golden_manifest.json."
            )
        if not args.allow_random_golden:
            raise RuntimeError(
                "No frozen golden dataset found. Passing source records randomly into a golden split is disabled by default. "
                "To explicitly allow random golden partitioning, re-run with --allow-random-golden."
            )
        if len(groups) < 3:
            raise RuntimeError(
                "At least 3 independent document groups are required for random train/val/golden split; "
                "the fallback that duplicated records across splits is disabled."
            )
        train, validation, golden = _allocate_groups_three_way(
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
        "total_source": len(valid) + len(rejected) + leaked_golden_exclusions,
        "total_valid": len(valid),
        "rejected_count": len(rejected),
        "golden_isolated_exclusions": leaked_golden_exclusions,
        "rejected_details": rejected,
        "train_count": len(train),
        "val_count": len(validation),
        "golden_count": len(golden),
        "golden_frozen_outside": has_frozen_golden,
        "seed": args.seed,
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
        "golden_frozen": has_frozen_golden,
        "golden_source": (
            str(args.golden_ids)
            if args.golden_ids
            else ("external_golden" if has_frozen_golden else "split")
        ),
        "seed": args.seed,
        "created_at": report["created_at"],
        "pipeline_stage": "SPLIT",
    }
    write_json_atomic(args.dataset / "split_manifest.json", manifest)

    print(
        f"Split {args.dataset}: train={len(train)}, val={len(validation)}, "
        f"golden={len(golden)}, rejected={len(rejected)}, golden_leak_prevented={leaked_golden_exclusions}"
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
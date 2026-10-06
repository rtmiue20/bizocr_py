"""Build field-level correction dictionary from dataset export (train_source.jsonl)."""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from training_pipeline.pipeline_common import (
        iter_jsonl,
        utc_now,
        write_json_atomic,
    )
except ImportError:
    from pipeline_common import (
        iter_jsonl,
        utc_now,
        write_json_atomic,
    )

EXCLUDED_FIELDS = {
    "phone",
    "email",
    "tax_code",
    "website",
    "phone_number",
    "tax_id",
    "vat_code",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build correction dictionary from export dataset"
    )
    parser.add_argument(
        "--train-source",
        type=Path,
        default=None,
        help="Path to train_source.jsonl file",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to dataset directory containing train_source.jsonl",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=3,
        help="Minimum distinct documents required to retain rule (default 3)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.6,
        help="Minimum SequenceMatcher ratio between original and corrected values (default 0.6)",
    )
    parser.add_argument(
        "--min-tenants-global",
        type=int,
        default=2,
        help="Minimum distinct tenants required to make a rule global (default 2)",
    )
    parser.add_argument(
        "--keep-versions",
        type=int,
        default=5,
        help="Number of versioned history files to retain on disk (default 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("saved_models/corrections"),
    )
    return parser.parse_args()


def parse_predicted_fields(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            res = json.loads(raw)
            if isinstance(res, dict):
                return res
        except Exception:
            pass
    return {}


def cleanup_old_versions(output_dir: Path, keep_versions: int) -> None:
    if keep_versions <= 0:
        return
    version_files = sorted(
        output_dir.glob("correction_corr-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old_file in version_files[keep_versions:]:
        try:
            old_file.unlink()
        except OSError:
            pass


def build_dictionary_from_source(
    train_source_path: Path,
    min_frequency: int,
    similarity_threshold: float,
    min_tenants_global: int,
    output_dir: Path,
    keep_versions: int,
) -> Path:
    if not train_source_path.exists():
        raise FileNotFoundError(f"Source file not found: {train_source_path}")

    # occurrences[field][orig][corr] -> {"docs": set(), "tenants": set()}
    occurrences: dict[str, dict[str, dict[str, dict[str, set[str]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"docs": set(), "tenants": set()}))
    )

    for line_no, record in iter_jsonl(train_source_path):
        metadata = record.get("metadata") or {}
        doc_id = str(metadata.get("document_id") or metadata.get("id") or f"line_{line_no}")
        tenant_id = str(
            metadata.get("owner_company_id")
            or metadata.get("tenant_id")
            or "default_tenant"
        )

        pred_fields = parse_predicted_fields(metadata.get("predicted_fields_json"))
        target_output = record.get("output") or {}

        if not isinstance(target_output, dict):
            continue

        for field, target_val in target_output.items():
            field_clean = str(field).strip()
            field_lower = field_clean.lower()

            # Skip identifier fields
            if field_lower in EXCLUDED_FIELDS:
                continue

            corr = str(target_val or "").strip()
            orig = str(pred_fields.get(field_clean) or pred_fields.get(field) or "").strip()

            if not orig or not corr or orig == corr:
                continue

            # Similarity check using SequenceMatcher
            ratio = difflib.SequenceMatcher(None, orig, corr).ratio()
            if ratio < similarity_threshold:
                continue

            entry = occurrences[field_clean][orig][corr]
            entry["docs"].add(doc_id)
            entry["tenants"].add(tenant_id)

    global_rules: dict[str, dict[str, str]] = defaultdict(dict)
    tenant_rules: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))

    for field, orig_map in occurrences.items():
        for orig, corr_map in orig_map.items():
            candidates = []
            for corr, stats in corr_map.items():
                candidates.append(
                    {
                        "corr": corr,
                        "doc_count": len(stats["docs"]),
                        "tenants": stats["tenants"],
                    }
                )

            # Sort by frequency (distinct doc count) descending
            candidates.sort(key=lambda x: x["doc_count"], reverse=True)

            if not candidates:
                continue

            top = candidates[0]

            # Rule 1: Must appear in at least min_frequency distinct documents
            if top["doc_count"] < min_frequency:
                continue

            # Rule 2: Top candidate frequency must be >= 2x second candidate frequency
            if len(candidates) >= 2:
                second = candidates[1]
                if top["doc_count"] < 2 * second["doc_count"]:
                    # Ambiguous rule, discard
                    continue

            top_corr = top["corr"]
            top_tenants = top["tenants"]

            # Rule 3: Add to global rules if seen across >= min_tenants_global distinct tenants
            if len(top_tenants) >= min_tenants_global:
                global_rules[field][orig] = top_corr

            # Add to tenant-specific rules
            for t_id in top_tenants:
                tenant_rules[t_id][field][orig] = top_corr

    output_dir.mkdir(parents=True, exist_ok=True)
    version_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    total_global_rules = sum(len(m) for m in global_rules.values())
    total_tenant_rules = sum(
        sum(len(m) for m in t_dict.values()) for t_dict in tenant_rules.values()
    )

    dict_payload = {
        "version": f"corr-{version_str}",
        "created_at": utc_now(),
        "min_frequency": min_frequency,
        "similarity_threshold": similarity_threshold,
        "min_tenants_global": min_tenants_global,
        "total_global_rules": total_global_rules,
        "total_tenant_rules": total_tenant_rules,
        "corrections": dict(global_rules),
        "tenant_corrections": {t: dict(m) for t, m in tenant_rules.items()},
    }

    active_path = output_dir / "correction_dictionary.json"
    versioned_path = output_dir / f"correction_{dict_payload['version']}.json"

    write_json_atomic(versioned_path, dict_payload)
    write_json_atomic(active_path, dict_payload)

    # Cleanup old versioned files
    cleanup_old_versions(output_dir, keep_versions)

    print(
        f"Built correction dictionary: {total_global_rules} global rules, "
        f"{total_tenant_rules} tenant rules across {len(tenant_rules)} tenants "
        f"(saved to {active_path})"
    )
    return active_path


def main() -> int:
    args = parse_args()
    try:
        train_source = args.train_source
        if not train_source and args.dataset:
            train_source = args.dataset / "train_source.jsonl"
        if not train_source:
            train_source = Path("datasets/train_source.jsonl")

        build_dictionary_from_source(
            train_source_path=train_source,
            min_frequency=args.min_frequency,
            similarity_threshold=args.similarity_threshold,
            min_tenants_global=args.min_tenants_global,
            output_dir=args.output_dir,
            keep_versions=args.keep_versions,
        )
        return 0
    except Exception as exc:
        print(f"BUILD_CORRECTION_FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
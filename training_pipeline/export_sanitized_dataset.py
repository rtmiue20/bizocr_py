"""Export trusted human-reviewed corrections into a versioned JSONL dataset.
Default SQL assumes the project's snake_case tables:
  documents
  field_correction_audits
If the deployed schema differs, pass --sql-file with a SELECT returning these
aliases:
  document_id, raw_ocr_text, document_type, extracted_fields,
  field_name, corrected_value, reviewed_by_role, audit_created_at, audit_id
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import mysql.connector
except ImportError:  # pragma: no cover - dependency is environment-specific
    mysql = None

try:
    from training_pipeline.pipeline_common import (
        parse_json_value,
        set_field_path,
        sha256_jsonl,
        utc_now,
        write_json_atomic,
        write_jsonl,
    )
except ImportError:
    from pipeline_common import (
        parse_json_value,
        set_field_path,
        sha256_jsonl,
        utc_now,
        write_json_atomic,
        write_jsonl,
    )
DEFAULT_SQL = """
SELECT
    d.id AS document_id,
    d.raw_ocr_text AS raw_ocr_text,
    d.document_type AS document_type,
    d.extracted_fields AS extracted_fields,
    fa.field_name AS field_name,
    fa.corrected_value AS corrected_value,
    fa.reviewed_by_role AS reviewed_by_role,
    fa.created_at AS audit_created_at,
    fa.id AS audit_id
FROM documents d
JOIN field_correction_audits fa ON fa.document_id = d.id
WHERE d.status = 'APPROVED'
  AND fa.reviewed_by_role IN ('ADMIN', 'REVIEWER')
ORDER BY d.id, fa.field_name, fa.created_at, fa.id
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=os.getenv("DATASET_ROOT", "datasets"))
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--sql-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()
def read_sql(path: Path | None) -> str:
    if path is None:
        return DEFAULT_SQL
    return path.read_text(encoding="utf-8")
def connect_database():
    if mysql is None:
        raise RuntimeError(
            "mysql-connector-python is required. Install it with "
            "pip install mysql-connector-python."
        )
    required = ["DB_HOST", "DB_USER", "DB_NAME"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing database environment variables: {', '.join(missing)}")
    return mysql.connector.connect(
        host=os.environ["DB_HOST"],
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.environ["DB_USER"],
        password=os.getenv("DB_PASSWORD", ""),
        database=os.environ["DB_NAME"],
        autocommit=True,
    )
def _sortable_audit_key(row: dict[str, Any]) -> tuple[str, str]:
    timestamp = str(row.get("audit_created_at") or "")
    audit_id = str(row.get("audit_id") or "")
    return timestamp, audit_id
def _base_output(row: dict[str, Any]) -> dict[str, Any]:
    base = parse_json_value(row.get("extracted_fields"))
    if base is None:
        return {}
    if not isinstance(base, dict):
        raise ValueError(
            f"Document {row.get('document_id')} extracted_fields is not a JSON object"
        )
    return deepcopy(base)
def build_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        document_id = row.get("document_id")
        field_name = row.get("field_name")
        if document_id is None or not field_name:
            continue
        by_document[str(document_id)].append(row)
    records: list[dict[str, Any]] = []
    counters = {
        "documents_seen": len(by_document),
        "documents_exported": 0,
        "documents_rejected": 0,
        "fields_applied": 0,
        "documents_with_history": 0,
    }
    for document_id, document_rows in by_document.items():
        try:
            first = document_rows[0]
            raw_ocr_text = str(first.get("raw_ocr_text") or "").strip()
            document_type = str(first.get("document_type") or "").strip()
            if not raw_ocr_text:
                raise ValueError("raw OCR text is empty")
            if not document_type:
                raise ValueError("document type is empty")
            final_json = _base_output(first)
            latest_by_field: dict[str, dict[str, Any]] = {}
            for row in document_rows:
                field_name = str(row["field_name"])
                previous = latest_by_field.get(field_name)
                if previous is None or _sortable_audit_key(row) >= _sortable_audit_key(previous):
                    latest_by_field[field_name] = row
            history_conflicts = 0
            for field_name, row in latest_by_field.items():
                value = parse_json_value(row.get("corrected_value"))
                if value is None:
                    raise ValueError(f"field {field_name!r} has NULL corrected_value")
                set_field_path(final_json, field_name, value)
                counters["fields_applied"] += 1
                distinct_values = {
                    json.dumps(parse_json_value(item.get("corrected_value")),
                               ensure_ascii=False, sort_keys=True)
                    for item in document_rows
                    if str(item.get("field_name")) == field_name
                }
                if len(distinct_values) > 1:
                    history_conflicts += 1
            if history_conflicts:
                counters["documents_with_history"] += 1
            records.append(
                {
                    "instruction": (
                        f"Trích xuất chứng từ loại {document_type} thành JSON theo "
                        f"schema của hệ thống. Chỉ sử dụng thông tin có trong OCR."
                    ),
                    "input": raw_ocr_text,
                    "output": final_json,
                    "metadata": {
                        "document_id": document_id,
                        "document_type": document_type,
                        "source": "TRUSTED_HUMAN_REVIEW",
                        "reviewed_roles": sorted(
                            {str(row.get("reviewed_by_role")) for row in document_rows}
                        ),
                        "field_count": len(latest_by_field),
                        "correction_history_conflicts": history_conflicts,
                    },
                }
            )
            counters["documents_exported"] += 1
        except (TypeError, ValueError, json.JSONDecodeError):
            counters["documents_rejected"] += 1
    return records, counters
def export_dataset(args: argparse.Namespace) -> Path:
    dataset_version = args.dataset_version or (
        "ds-" + utc_now().replace(":", "").replace("-", "").replace("+00:00", "Z")
    )
    output_dir = Path(args.output_root) / dataset_version
    output_dir.mkdir(parents=True, exist_ok=False)
    connection = connect_database()
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(read_sql(args.sql_file))
        rows = cursor.fetchall()
        cursor.close()
    finally:
        connection.close()
    records, counters = build_records(rows)
    if not records:
        raise RuntimeError("No trusted records were exported; refusing to create a dataset")
    source_path = output_dir / "train_source.jsonl"
    write_jsonl(source_path, records)
    source_hash = sha256_jsonl(source_path)
    report = {
        "created_at": utc_now(),
        "dataset_version": dataset_version,
        "source_sha256": source_hash,
        "query_source": str(args.sql_file) if args.sql_file else "built-in",
        **counters,
    }
    write_json_atomic(output_dir / "export_report.json", report)
    manifest = {
        "dataset_version": dataset_version,
        "source_file": source_path.name,
        "source_sha256": source_hash,
        "document_count": len(records),
        "schema_versions": [],
        "created_at": report["created_at"],
        "pipeline_stage": "EXPORTED",
    }
    write_json_atomic(output_dir / "dataset_manifest.json", manifest)
    print(f"Exported {len(records)} trusted documents to {output_dir}")
    return output_dir
def main() -> int:
    args = parse_args()
    if args.dry_run:
        print("Dry run only validates CLI configuration; database was not queried.")
        return 0
    try:
        export_dataset(args)
        return 0
    except Exception as exc:
        print(f"EXPORT_FAILED: {exc}", file=sys.stderr)
        return 1
if __name__ == "__main__":
    raise SystemExit(main())
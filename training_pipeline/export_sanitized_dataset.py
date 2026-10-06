"""Export trusted human-reviewed corrections into a versioned JSONL dataset.
Default SQL assumes the project's snake_case tables:
  documents
  field_correction_audits
If the deployed schema differs, pass --sql-file with a SELECT returning these
aliases:
  document_id, owner_company_id, raw_ocr_text, document_type, extracted_fields,
  predicted_fields_json, model_version, correction_count
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
        sha256_jsonl,
        utc_now,
        write_json_atomic,
        write_jsonl,
    )
except ImportError:
    from pipeline_common import (
        parse_json_value,
        sha256_jsonl,
        utc_now,
        write_json_atomic,
        write_jsonl,
    )

UNWANTED_KEYS = {"field_confidences", "anomaly_analysis", "review_status"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=os.getenv("DATASET_ROOT", "datasets"))
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--batch-id", type=int, default=None, help="Filter by training_batch_id")
    parser.add_argument(
        "--doc-ids",
        type=str,
        default=None,
        help="Comma-separated list of document IDs for Golden Dataset export (bypasses training_eligible filter)",
    )
    parser.add_argument("--sql-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_doc_ids(doc_ids_str: str | None) -> list[int] | None:
    if not doc_ids_str:
        return None
    ids = []
    for item in doc_ids_str.split(","):
        item = item.strip()
        if item:
            try:
                ids.append(int(item))
            except ValueError:
                raise ValueError(f"Invalid document ID in --doc-ids: {item!r}")
    return ids if ids else None


def read_sql(
    path: Path | None,
    batch_id: int | None = None,
    doc_ids: list[int] | None = None,
) -> tuple[str, list[Any]]:
    if path is not None:
        query = path.read_text(encoding="utf-8")
        params = []
        if batch_id is not None and "%s" in query:
            params.append(batch_id)
        return query, params

    query = """
SELECT
    d.id AS document_id,
    d.owner_company_id AS owner_company_id,
    d.raw_ocr_text AS raw_ocr_text,
    d.document_type AS document_type,
    d.extracted_fields AS extracted_fields,
    d.predicted_fields_json AS predicted_fields_json,
    d.model_version AS model_version,
    (SELECT COUNT(*) FROM field_correction_audits fa WHERE fa.document_id = d.id) AS correction_count
FROM documents d
"""
    params: list[Any] = []
    where_clauses: list[str] = []

    if doc_ids:
        placeholders = ", ".join(["%s"] * len(doc_ids))
        where_clauses.append(f"d.id IN ({placeholders})")
        params.extend(doc_ids)
    else:
        where_clauses.append("d.training_eligible = 1")
        if batch_id is not None:
            where_clauses.append("d.training_batch_id = %s")
            params.append(batch_id)

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += "\nORDER BY d.id\n"
    return query, params


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


def sanitize_dict(data: Any) -> Any:
    """Recursively clean runtime operational/evaluation keys."""
    if isinstance(data, dict):
        return {
            k: sanitize_dict(v)
            for k, v in data.items()
            if k not in UNWANTED_KEYS
        }
    if isinstance(data, list):
        return [sanitize_dict(item) for item in data]
    return data


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
    records: list[dict[str, Any]] = []
    counters = {
        "documents_seen": len(rows),
        "documents_exported": 0,
        "documents_rejected": 0,
        "total_corrections_counted": 0,
    }

    for row in rows:
        document_id = row.get("document_id")
        if document_id is None:
            continue

        try:
            # 1. Handle raw_ocr_text (proper decoding for byte/bytearray)
            raw_ocr_raw = row.get("raw_ocr_text")
            if isinstance(raw_ocr_raw, (bytes, bytearray)):
                raw_ocr_text = raw_ocr_raw.decode("utf-8", errors="replace").strip()
            else:
                raw_ocr_text = str(raw_ocr_raw or "").strip()

            document_type = str(row.get("document_type") or "").strip()
            if not raw_ocr_text:
                raise ValueError("raw OCR text is empty")
            if not document_type:
                raise ValueError("document type is empty")

            # 2. Output is extracted_fields (final target) minus runtime keys
            final_json = _base_output(row)
            final_json = sanitize_dict(final_json)

            # 3. Sanitize predicted_fields_json with identical key filter
            predicted_fields = parse_json_value(row.get("predicted_fields_json"))
            if predicted_fields is not None:
                predicted_fields = sanitize_dict(predicted_fields)

            correction_count = int(row.get("correction_count") or 0)
            counters["total_corrections_counted"] += correction_count
            owner_company_id = row.get("owner_company_id")

            # Debug sample print for the first exported record
            if counters["documents_exported"] == 0:
                print(f"[DEBUG] Sample Document ID: {document_id}")
                print(f"[DEBUG] Raw OCR Text Type from DB: {type(raw_ocr_raw)}")
                print(f"[DEBUG] Raw OCR Text Preview (first 100 chars): {raw_ocr_text[:100]!r}")

            records.append(
                {
                    "instruction": (
                        f"Trích xuất chứng từ loại {document_type} thành JSON theo "
                        f"schema của hệ thống. Chỉ sử dụng thông tin có trong OCR."
                    ),
                    "input": raw_ocr_text,
                    "output": final_json,
                    "metadata": {
                        "document_id": str(document_id),
                        "owner_company_id": owner_company_id,
                        "document_type": document_type,
                        "model_version": str(row.get("model_version") or ""),
                        "predicted_fields_json": predicted_fields,
                        "source": "TRUSTED_HUMAN_REVIEW",
                        "correction_count": correction_count,
                    },
                }
            )
            counters["documents_exported"] += 1

        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"[WARN] Skipping Document {document_id}: {exc}", file=sys.stderr)
            counters["documents_rejected"] += 1

    return records, counters


def export_dataset(args: argparse.Namespace) -> Path:
    doc_ids = parse_doc_ids(args.doc_ids)
    dataset_version = args.dataset_version or (
        "ds-" + utc_now().replace(":", "").replace("-", "").replace("+00:00", "Z")
    )
    output_dir = Path(args.output_root) / dataset_version
    output_dir.mkdir(parents=True, exist_ok=False)
    connection = connect_database()
    try:
        cursor = connection.cursor(dictionary=True)
        query, params = read_sql(args.sql_file, args.batch_id, doc_ids)
        cursor.execute(query, params)
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
"""Shared utilities for the Model B training pipeline.
The pipeline is intentionally file/manifest based:
export -> validate/split -> train -> evaluate -> register -> inference
Every stage receives an explicit path or version.  No stage silently picks a
random "latest" directory, because that is unsafe in CI and production.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value

def read_json_any(path: Path) -> Any:
    """Read a JSON file and return any valid JSON type (dict, list, etc.)."""
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)

def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON atomically in the same directory as the final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield line_number, value

def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def sha256_jsonl(path: Path) -> str:
    """Hash normalized JSONL, ignoring blank lines and key ordering."""
    digest = hashlib.sha256()
    for _, record in iter_jsonl(path):
        encoded = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()

def parse_json_value(value: Any) -> Any:
    """Decode JSON columns without accidentally converting ordinary text."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] in '{["' or stripped in {"true", "false", "null"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    if re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
        if len(stripped.lstrip("-")) > 1 and stripped.lstrip("-").startswith("0"):
            return value
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value

def set_field_path(target: dict[str, Any], field_name: str, value: Any) -> None:
    """Set a simple field or a dotted nested field."""
    if field_name in target:
        target[field_name] = value
        return
    parts = [part for part in field_name.split(".") if part]
    if not parts:
        return
    cursor: dict[str, Any] = target
    for part in parts[:-1]:
        current = cursor.get(part)
        if not isinstance(current, dict):
            current = {}
            cursor[part] = current
        cursor = current
    cursor[parts[-1]] = value

def load_schema(schema_dir: Path | None, document_type: str | None) -> dict[str, Any] | None:
    if schema_dir is None or not document_type:
        return None
    candidates = [
        schema_dir / f"{document_type}.json",
        schema_dir / f"{document_type.lower()}.json",
        schema_dir / "default.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return read_json(candidate)
    return None

def validate_schema(value: Any, schema: dict[str, Any] | None) -> list[str]:
    if schema is None:
        return []
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError(
            "jsonschema is required when a schema directory is configured"
        ) from exc
    validator = Draft202012Validator(schema)
    return [error.message for error in validator.iter_errors(value)]

def normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        value = unicodedata.normalize("NFC", value)
        return re.sub(r"\s+", " ", value.strip()).casefold()
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return value

def values_equal(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False

    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        return all(
            key in left and key in right and values_equal(left[key], right[key])
            for key in keys
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            values_equal(a, b) for a, b in zip(left, right)
        )
    return normalize_scalar(left) == normalize_scalar(right)

def extract_json_object(raw: str) -> Any:
    """Parse a JSON object from a model response without trusting markdown."""
    if not isinstance(raw, str):
        raise ValueError("Model output is not text")
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        raise ValueError("No JSON object or array found in model output")
    value, _ = decoder.raw_decode(text[min(starts):])
    if not isinstance(value, dict):
        raise ValueError("Model output must be a JSON object")
    return value

# =========================================================================
# Shared Domain & Contract Handlers
# =========================================================================

def extract_document_id(record: dict[str, Any]) -> str | None:
    """Trích xuất ID tài liệu chuẩn hóa từ bản ghi theo thứ tự ưu tiên."""
    if "document_id" in record and record["document_id"]:
        return str(record["document_id"])

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        if "document_id" in metadata and metadata["document_id"]:
            return str(metadata["document_id"])
        if "id" in metadata and metadata["id"]:
            return str(metadata["id"])

    if "id" in record and record["id"]:
        return str(record["id"])
    return None

def load_golden_manifest_ids(dataset_dir: Path) -> set[str]:
    """Đọc danh sách document_ids từ golden_manifest.json theo đúng hợp đồng."""
    manifest_path = dataset_dir / "golden_manifest.json"
    if not manifest_path.exists():
        manifests = list(dataset_dir.glob("golden_manifest*.json"))
        if manifests:
            manifest_path = manifests[0]
        else:
            return set()

    data = read_json(manifest_path)

    # Ưu tiên khóa document_ids chuẩn
    if "document_ids" in data and isinstance(data["document_ids"], list):
        return {str(doc_id) for doc_id in data["document_ids"]}

    # Hỗ trợ tương thích ngược nếu có danh sách documents dạng object
    if "documents" in data and isinstance(data["documents"], list):
        extracted_ids = set()
        for item in data["documents"]:
            if isinstance(item, dict):
                doc_id = extract_document_id(item)
                if doc_id:
                    extracted_ids.add(doc_id)
            elif item:
                extracted_ids.add(str(item))
        return extracted_ids

    return set()

def load_golden_docs(dataset_dir: Path) -> list[dict[str, Any]]:
    """Tải toàn bộ tài liệu từ các tập tin golden*.jsonl và đồng bộ hóa document_id."""
    docs = []
    golden_files = sorted(dataset_dir.glob("golden*.jsonl"))
    for golden_file in golden_files:
        for _, record in iter_jsonl(golden_file):
            doc_id = extract_document_id(record)
            if doc_id:
                record["document_id"] = doc_id
                if "metadata" in record and isinstance(record["metadata"], dict):
                    record["metadata"]["document_id"] = doc_id
            docs.append(record)
    return docs
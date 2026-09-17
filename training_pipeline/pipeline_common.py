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
    """Decode JSON columns without accidentally converting ordinary text.
    Database drivers may return JSON columns as dict/list, while VARCHAR
    correction values are normally strings.  Only JSON-looking strings are
    decoded.
    """
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
        # Preserve identifiers such as tax codes with leading zeros.
        if len(stripped.lstrip("-")) > 1 and stripped.lstrip("-").startswith("0"):
            return value
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value
def set_field_path(target: dict[str, Any], field_name: str, value: Any) -> None:
    """Set a simple field or a dotted nested field.
    Existing literal field names win.  This keeps names such as "tax.code"
    safe while still supporting nested paths when the root does not exist.
    """
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
        return re.sub(r"\s+", " ", value.strip()).casefold()
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return value
def values_equal(left: Any, right: Any) -> bool:
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
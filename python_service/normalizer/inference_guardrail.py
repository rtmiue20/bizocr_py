"""Schema and business guardrail for model output.
This module does not silently turn invalid output into a valid document.  It
returns an explicit NEEDS_REVIEW envelope that the caller must persist and
route to human review.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from training_pipeline.pipeline_common import (
        extract_json_object,
        load_schema,
        read_json,
        utc_now,
        validate_schema,
    )
except ImportError:
    from training_pipeline.pipeline_common import (
        extract_json_object,
        load_schema,
        read_json,
        utc_now,
        validate_schema,
    )
@dataclass
class GuardrailResult:
    status: str
    data: dict[str, Any] | None
    reason: str | None
    errors: list[str]
    model_version: str | None
    schema_version: str | None
    raw_output_sha256: str
    checked_at: str
    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
class InferenceGuardrail:
    def __init__(
        self,
        active_root: Path = Path("saved_models/active_model"),
        schema_dir: Path | None = None,
    ):
        self.active_root = active_root
        self.schema_dir = schema_dir
    def active_manifest(self) -> dict[str, Any] | None:
        manifest_path = self.active_root / "production_manifest.json"
        if not manifest_path.exists():
            return None
        manifest = read_json(manifest_path)
        if manifest.get("status") != "ACTIVE":
            raise RuntimeError("Active model manifest is not ACTIVE")
        return manifest
    @staticmethod
    def _business_errors(value: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        # Conservative generic checks; document-specific rules belong in JSON
        # Schema or a separate validator, not in this generic helper.
        for key in ("total_amount", "subtotal", "tax_amount", "quantity"):
            if key in value and isinstance(value[key], (int, float)) and value[key] < 0:
                errors.append(f"{key} must not be negative")
        return errors
    def validate_and_clean_json(
        self,
        raw_llm_output: str,
        document_type: str,
        include_raw_on_error: bool = False,
    ) -> dict[str, Any]:
        raw_hash = hashlib.sha256(str(raw_llm_output).encode("utf-8")).hexdigest()
        manifest = self.active_manifest()
        model_version = manifest.get("active_version") if manifest else None
        schema = load_schema(self.schema_dir, document_type)
        schema_version = None
        if schema:
            schema_version = str(schema.get("$id") or schema.get("version") or "")
        base = {
            "model_version": model_version,
            "schema_version": schema_version,
            "raw_output_sha256": raw_hash,
            "checked_at": utc_now(),
        }
        try:
            value = extract_json_object(raw_llm_output)
            schema_errors = validate_schema(value, schema)
            business_errors = self._business_errors(value)
            errors = schema_errors + business_errors
            result = GuardrailResult(
                status="ACCEPTED" if not errors else "NEEDS_REVIEW",
                data=value if not errors else None,
                reason=None if not errors else "GUARDRAIL_VALIDATION_FAILED",
                errors=errors,
                **base,
            ).as_dict()
        except Exception as exc:
            result = GuardrailResult(
                status="NEEDS_REVIEW",
                data=None,
                reason="INVALID_JSON_OUTPUT",
                errors=[str(exc)],
                **base,
            ).as_dict()
        if include_raw_on_error and result["status"] != "ACCEPTED":
            result["raw_output"] = str(raw_llm_output)[:10000]
        return result
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-file", type=Path, required=True)
    parser.add_argument("--document-type", required=True)
    parser.add_argument("--schema-dir", type=Path, default=None)
    parser.add_argument("--active-root", type=Path, default=Path("saved_models/active_model"))
    args = parser.parse_args()
    try:
        raw = args.raw_file.read_text(encoding="utf-8")
        result = InferenceGuardrail(args.active_root, args.schema_dir).validate_and_clean_json(
            raw, args.document_type
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "ACCEPTED" else 2
    except Exception as exc:
        print(f"GUARDRAIL_FAILED: {exc}", file=sys.stderr)
        return 1
if __name__ == "__main__":
    raise SystemExit(main())
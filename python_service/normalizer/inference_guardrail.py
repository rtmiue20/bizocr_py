"""Schema and business guardrail for model output.
This module does not silently turn invalid output into a valid document. It
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
from typing import Any, Callable, Optional

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
        """Validate và làm sạch JSON output thu được từ LLM thuần."""
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

    def merge_and_guardrail_entities(
        self,
        regex_extracted: dict[str, Any],
        raw_ocr_text: str,
        document_type: str,
        ollama_client_func: Optional[Callable[[str, list[str]], dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Xử lý BƯỚC 5 (Hybrid Extraction & Guardrail):
        1. Tính Completeness score của Regex/Heuristics.
        2. Nếu đủ 100% -> Trả về ACCEPTED (Không gọi LLM).
        3. Nếu thiếu -> Gọi Ollama có điều kiện (chỉ truyền danh sách các trường rỗng).
        4. Kiểm tra Grounding: Ép chuỗi LLM trả về phải xuất hiện trong raw_ocr_text.
        5. Gắn nguồn 'LLM_OLLAMA' và trạng thái grounded: True/False.
        6. Nếu có can thiệp của LLM -> Khóa cứng trạng thái ở NEEDS_REVIEW.
        """
        manifest = self.active_manifest()
        model_version = manifest.get("active_version") if manifest else "HYBRID_REGEX_OLLAMA"
        schema = load_schema(self.schema_dir, document_type)
        schema_version = str(schema.get("$id") or schema.get("version") or "") if schema else None

        raw_hash = hashlib.sha256(str(raw_ocr_text).encode("utf-8")).hexdigest()
        base_meta = {
            "model_version": model_version,
            "schema_version": schema_version,
            "raw_output_sha256": raw_hash,
            "checked_at": utc_now(),
        }

        # 1. Tính Completeness Score
        total_fields = len(regex_extracted)
        filled_fields = sum(
            1 for v in regex_extracted.values()
            if v is not None and str(v).strip() != ""
        )
        completeness_score = round(filled_fields / total_fields, 4) if total_fields else 0.0

        # Khởi tạo danh sách kết quả với nguồn Regex
        final_entities: dict[str, Any] = {}
        for k, v in regex_extracted.items():
            has_val = v is not None and str(v).strip() != ""
            final_entities[k] = {
                "value": str(v).strip() if has_val else None,
                "source": "REGEX_HEURISTICS" if has_val else None,
                "grounded": True if has_val else False
            }

        # Nếu Regex đã tìm đủ 100% -> Chấp nhận luôn (ACCEPTED)
        if completeness_score >= 1.0:
            return GuardrailResult(
                status="ACCEPTED",
                data={
                    "entities": final_entities,
                    "completenessScore": 1.0,
                    "llmInvoked": False
                },
                reason=None,
                errors=[],
                **base_meta
            ).as_dict()

        # 2. Xác định các trường còn thiếu để gọi Ollama
        missing_keys = [
            k for k, v in regex_extracted.items()
            if v is None or str(v).strip() == ""
        ]

        if not ollama_client_func:
            return GuardrailResult(
                status="NEEDS_REVIEW",
                data={
                    "entities": final_entities,
                    "completenessScore": completeness_score,
                    "llmInvoked": False
                },
                reason="MISSING_FIELDS_NO_LLM_CLIENT",
                errors=[f"Thiếu các trường {missing_keys} nhưng không tìm thấy llm_client_func."],
                **base_meta
            ).as_dict()

        # 3. Gọi Ollama trích xuất riêng các trường thiếu
        try:
            llm_output = ollama_client_func(raw_ocr_text, missing_keys) or {}
        except Exception as exc:
            return GuardrailResult(
                status="NEEDS_REVIEW",
                data={
                    "entities": final_entities,
                    "completenessScore": completeness_score,
                    "llmInvoked": True
                },
                reason="LLM_EXECUTION_FAILED",
                errors=[f"Lỗi khi thực thi Ollama: {str(exc)}"],
                **base_meta
            ).as_dict()

        grounding_errors: list[str] = []
        raw_text_lower = raw_ocr_text.lower()

        # 4. Điền trường thiếu + Grounding check
        for key in missing_keys:
            val = llm_output.get(key)
            if val is not None and str(val).strip() != "":
                val_str = str(val).strip()
                # Grounding: Chuỗi LLM trả về bắt buộc phải có trong văn bản gốc OCR
                is_grounded = val_str.lower() in raw_text_lower

                final_entities[key] = {
                    "value": val_str,
                    "source": "LLM_OLLAMA",
                    "grounded": is_grounded
                }

                if not is_grounded:
                    grounding_errors.append(
                        f"Trường '{key}'='{val_str}' do LLM trả về không khớp với văn bản OCR gốc (Grounding Failed)."
                    )

        # 5. Có sự can thiệp của LLM -> Bắt buộc khóa ở trạng thái NEEDS_REVIEW
        return GuardrailResult(
            status="NEEDS_REVIEW",
            data={
                "entities": final_entities,
                "completenessScore": completeness_score,
                "llmInvoked": True
            },
            reason="LLM_INVOKED_FOR_MISSING_FIELDS" if not grounding_errors else "GROUNDING_VALIDATION_FAILED",
            errors=grounding_errors,
            **base_meta
        ).as_dict()


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
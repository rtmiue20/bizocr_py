"""Schema and business guardrail for model output.
This module does not silently turn invalid output into a valid document. It
returns an explicit NEEDS_REVIEW envelope that the caller must persist and
route to human review.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

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


# Danh sách các trường BẮT BUỘC theo loại tài liệu (Dùng làm dự phòng khi Schema không khai báo 'required')
REQUIRED_FIELDS_BY_DOC_TYPE: dict[str, list[str]] = {
    "NAMECARD": ["name"],
    "BUSINESS_CARD": ["name"],
    "CARD": ["name"],
    "INVOICE": ["invoice_number", "total_amount", "invoice_date"],
    "RECEIPT": ["total_amount", "date"],
    "ID_CARD": ["id_number", "full_name"],
}


def _normalize_for_grounding(text: str) -> str:
    """
    Chuẩn hóa chuỗi văn bản phục vụ so sánh grounding:
    Loại bỏ khoảng trắng, dấu câu, gạch nối, gạch chéo, ngắt dòng và chuyển về chữ thường.
    """
    if not text:
        return ""
    return re.sub(r"[\s\.\-\(\)/,:\n\r\t]+", "", str(text).lower())


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
        """
        FIX 1: Đọc linh hoạt từ production_manifest.json hoặc classifier_manifest.json
        để tương thích với luồng xuất manifest của classifier_ml.py.
        """
        for name in ["production_manifest.json", "classifier_manifest.json"]:
            p = self.active_root / name
            if p.exists():
                manifest = read_json(p)
                if manifest.get("status") == "ACTIVE":
                    return manifest
                return manifest
        return None

    @staticmethod
    def _business_errors(value: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in ("total_amount", "subtotal", "tax_amount", "quantity"):
            if key in value and isinstance(value[key], (int, float)) and value[key] < 0:
                errors.append(f"{key} must not be negative")
        return errors

    def _get_required_fields(
        self, schema: Optional[dict[str, Any]], document_type: str, fallback_keys: list[str]
    ) -> list[str]:
        """Xác định danh sách các trường BẮT BUỘC của loại tài liệu."""
        if schema and isinstance(schema.get("required"), list) and schema.get("required"):
            return schema["required"]

        doc_type_upper = document_type.upper() if document_type else ""
        if doc_type_upper in REQUIRED_FIELDS_BY_DOC_TYPE:
            return REQUIRED_FIELDS_BY_DOC_TYPE[doc_type_upper]

        return fallback_keys

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
        1. Tính Completeness score dựa trên các trường BẮT BUỘC.
        2. Nếu đủ 100% trường bắt buộc -> Trả về ACCEPTED (Không gọi LLM).
        3. Nếu thiếu trường bắt buộc -> Gọi Ollama trích xuất CHỈ CÁC TRƯỜNG BẮT BUỘC RỖNG.
        4. Grounding Check: So sánh chuỗi đã qua chuẩn hóa dấu câu/khoảng trắng.
        5. Nếu tất cả dữ liệu trích xuất hợp lệ và grounded -> Trả về kết quả phù hợp.
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

        # 1. Xác định tập các trường BẮT BUỘC và tính Completeness Score
        required_keys = self._get_required_fields(schema, document_type, list(regex_extracted.keys()))

        filled_required = sum(
            1 for k in required_keys
            if k in regex_extracted and regex_extracted[k] is not None and str(regex_extracted[k]).strip() != ""
        )
        completeness_score = round(filled_required / len(required_keys), 4) if required_keys else 1.0

        # Khởi tạo dict lưu kết quả entities
        final_entities: dict[str, Any] = {}
        for k, v in regex_extracted.items():
            has_val = v is not None and str(v).strip() != ""
            final_entities[k] = {
                "value": str(v).strip() if has_val else None,
                "source": "REGEX_HEURISTICS" if has_val else None,
                "grounded": True if has_val else False,
            }

        # Nếu Regex đã tìm đủ 100% các trường bắt buộc -> Chấp nhận luôn (ACCEPTED - Bỏ qua Ollama)
        if completeness_score >= 1.0:
            return GuardrailResult(
                status="ACCEPTED",
                data={
                    "entities": final_entities,
                    "completenessScore": 1.0,
                    "llmInvoked": False,
                },
                reason=None,
                errors=[],
                **base_meta,
            ).as_dict()

        # 2. FIX 2: CHỈ lấy các trường BẮT BUỘC bị thiếu/rỗng để hỏi Ollama (Bỏ qua trường Optional)
        missing_keys = [
            k for k in required_keys
            if k not in regex_extracted or regex_extracted[k] is None or str(regex_extracted[k]).strip() == ""
        ]

        if not ollama_client_func:
            return GuardrailResult(
                status="NEEDS_REVIEW",
                data={
                    "entities": final_entities,
                    "completenessScore": completeness_score,
                    "llmInvoked": False,
                },
                reason="MISSING_FIELDS_NO_LLM_CLIENT",
                errors=[f"Thiếu các trường bắt buộc {missing_keys} nhưng không tìm thấy llm_client_func."],
                **base_meta,
            ).as_dict()

        # 3. Gọi Ollama trích xuất chỉ những trường bắt buộc còn thiếu
        try:
            llm_output = ollama_client_func(raw_ocr_text, missing_keys) or {}
        except Exception as exc:
            return GuardrailResult(
                status="NEEDS_REVIEW",
                data={
                    "entities": final_entities,
                    "completenessScore": completeness_score,
                    "llmInvoked": True,
                },
                reason="LLM_EXECUTION_FAILED",
                errors=[f"Lỗi khi thực thi Ollama: {str(exc)}"],
                **base_meta,
            ).as_dict()

        grounding_errors: list[str] = []
        norm_raw_text = _normalize_for_grounding(raw_ocr_text)

        # 4. Điền trường thiếu & Kiểm tra Grounding
        for key in missing_keys:
            val = llm_output.get(key)
            if val is not None and str(val).strip() != "":
                val_str = str(val).strip()
                norm_val = _normalize_for_grounding(val_str)

                # Grounding: Chuỗi LLM sau khi loại bỏ khoảng trắng/dấu câu phải xuất hiện trong OCR gốc
                is_grounded = bool(norm_val) and (norm_val in norm_raw_text)

                final_entities[key] = {
                    "value": val_str,
                    "source": "LLM_OLLAMA",
                    "grounded": is_grounded,
                }

                if not is_grounded:
                    grounding_errors.append(
                        f"Trường '{key}'='{val_str}' do LLM trả về không khớp với văn bản OCR gốc (Grounding Failed)."
                    )

        # 5. Có sự can thiệp của LLM -> Chuyển về trạng thái NEEDS_REVIEW để con người xác nhận
        return GuardrailResult(
            status="NEEDS_REVIEW",
            data={
                "entities": final_entities,
                "completenessScore": completeness_score,
                "llmInvoked": True,
            },
            reason="LLM_INVOKED_FOR_MISSING_FIELDS" if not grounding_errors else "GROUNDING_VALIDATION_FAILED",
            errors=grounding_errors,
            **base_meta,
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
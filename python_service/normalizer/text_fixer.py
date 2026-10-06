import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Danh sách các trường định danh không áp dụng tự động sửa lỗi
EXCLUDED_FIELDS = {
    "phone",
    "email",
    "tax_code",
    "website",
    "phone_number",
    "tax_id",
    "vat_code",
}


class TextFixer:
    """
    Áp dụng từ điển sửa lỗi theo trường (field-level correction dictionary).
    Hỗ trợ phân chia quy tắc theo Tenant (owner_company_id) và quy tắc chung (global).
    Sử dụng BIZOCR_HOME để xác định chính xác đường dẫn file từ điển.
    """

    def __init__(self, dictionary_path: Optional[str] = None):
        if dictionary_path:
            self.dictionary_path = Path(dictionary_path)
        else:
            bizocr_home = Path(os.getenv("BIZOCR_HOME", "."))
            self.dictionary_path = (
                bizocr_home / "saved_models" / "corrections" / "correction_dictionary.json"
            )

        self._lock = threading.Lock()
        self.version = "v0"
        self.corrections: Dict[str, Dict[str, str]] = {}
        self.tenant_corrections: Dict[str, Dict[str, Dict[str, str]]] = {}
        self.reload()

    def reload(self, path: Optional[Path] = None) -> bool:
        """Nạp lại từ điển sửa lỗi từ đĩa."""
        target_path = path or self.dictionary_path
        if not target_path.is_file():
            logger.warning(
                f"Chưa có file từ điển sửa lỗi tại {target_path} "
                f"(BIZOCR_HOME={os.getenv('BIZOCR_HOME', 'Chưa thiết lập')}). "
                f"Sử dụng từ điển rỗng."
            )
            with self._lock:
                self.corrections = {}
                self.tenant_corrections = {}
                self.version = "v0"
            return False

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            new_corrections = data.get("corrections", {})
            new_tenant_corrections = data.get("tenant_corrections", {})
            new_version = data.get("version", "v1")

            with self._lock:
                self.corrections = new_corrections
                self.tenant_corrections = new_tenant_corrections
                self.version = new_version

            global_rule_count = sum(len(m) for m in new_corrections.values())
            tenant_rule_count = sum(
                sum(len(m) for m in t_dict.values())
                for t_dict in new_tenant_corrections.values()
            )
            logger.info(
                f"Đã nạp từ điển sửa lỗi {new_version} từ {target_path}: "
                f"{global_rule_count} quy tắc chung, {tenant_rule_count} quy tắc tenant "
                f"trên {len(new_tenant_corrections)} tenant."
            )
            return True
        except Exception as e:
            logger.error(f"Lỗi khi nạp từ điển sửa lỗi {target_path}: {e}")
            return False

    def fix_field(
        self, field_name: str, value: Any, owner_company_id: Optional[str] = None
    ) -> Any:
        """
        Sửa lỗi cho 1 giá trị của trường cụ thể.
        Ưu tiên quy tắc của Tenant trước, sau đó tới quy tắc chung.
        Bỏ qua các trường định danh.
        """
        if not isinstance(value, str) or not value.strip():
            return value

        field_key = field_name.strip()
        field_lower = field_key.lower()

        # Bỏ qua các trường định danh
        if field_lower in EXCLUDED_FIELDS:
            return value

        val_str = str(value)
        val_stripped = val_str.strip()

        with self._lock:
            # 1. Tra cứu ưu tiên theo (owner_company_id, field)
            if owner_company_id and str(owner_company_id) in self.tenant_corrections:
                tenant_rules = self.tenant_corrections[str(owner_company_id)].get(field_key)
                if tenant_rules:
                    if val_str in tenant_rules:
                        return tenant_rules[val_str]
                    if val_stripped in tenant_rules:
                        return tenant_rules[val_stripped]

            # 2. Tra cứu từ điển chung (global)
            field_rules = self.corrections.get(field_key)
            if field_rules:
                if val_str in field_rules:
                    return field_rules[val_str]
                if val_stripped in field_rules:
                    return field_rules[val_stripped]

        return value

    def fix_extracted_fields(
        self, fields: Dict[str, Any], owner_company_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Duyệt và sửa lỗi cho toàn bộ dict extracted_fields.
        """
        if not isinstance(fields, dict):
            return fields

        fixed: Dict[str, Any] = {}
        for k, v in fields.items():
            if isinstance(v, dict):
                fixed[k] = self.fix_extracted_fields(v, owner_company_id=owner_company_id)
            elif isinstance(v, list):
                fixed[k] = [
                    self.fix_extracted_fields(elem, owner_company_id=owner_company_id)
                    if isinstance(elem, dict)
                    else self.fix_field(k, elem, owner_company_id=owner_company_id)
                    for elem in v
                ]
            else:
                fixed[k] = self.fix_field(k, v, owner_company_id=owner_company_id)
        return fixed


text_fixer = TextFixer()


# Tích hợp FastAPI Endpoint /fix-fields
try:
    from fastapi import APIRouter, Body
    from pydantic import BaseModel, Field

    router = APIRouter(tags=["Text Fixer"])

    class FixFieldsRequest(BaseModel):
        extracted_fields: Dict[str, Any] = Field(
            ..., description="Dictionary chứa các trường thông tin đã trích xuất"
        )
        owner_company_id: Optional[str] = Field(
            None, description="ID công ty/tenant để ưu tiên quy tắc riêng"
        )

    class FixFieldsResponse(BaseModel):
        version: str
        fixed_fields: Dict[str, Any]

    @router.post("/fix-fields", response_model=FixFieldsResponse)
    def fix_fields_endpoint(request: FixFieldsRequest):
        """
        Endpoint hậu xử lý sửa lỗi dữ liệu trích xuất từ từ điển sửa lỗi.
        """
        fixed = text_fixer.fix_extracted_fields(
            request.extracted_fields, owner_company_id=request.owner_company_id
        )
        return FixFieldsResponse(version=text_fixer.version, fixed_fields=fixed)

except ImportError:
    router = None
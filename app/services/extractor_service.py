import re


class DocumentExtractor:
    """
    Trích xuất field theo từng loại chứng từ (doc_type), đồng bộ logic
    và tên field với FieldExtractorService.java bên phía Java Spring Boot:
      - VAT_INVOICE / HOA_DON_VAT      -> tax_code, total_amount
      - ELECTRICITY_BILL / HOA_DON_DIEN -> customer_id, total_amount
      - WATER_BILL / HOA_DON_NUOC       -> customer_id, total_amount
    """

    # Regex tổng tiền dùng chung, đồng bộ với Java (đủ 4 biến thể từ khóa)
    AMOUNT_PATTERN = re.compile(
        r'(?:Tổng cộng tiền thanh toán|Cộng tiền hàng|Tổng tiền thanh toán|'
        r'Tổng tiền|Thành tiền)[:\-]?\s*([0-9\.\,]+)',
        re.IGNORECASE
    )

    MST_PATTERN = re.compile(
        r'(?:Mã số thuế|MST|Tax code)[:\-]?\s*([0-9\-\s]{10,14})',
        re.IGNORECASE
    )

    CUSTOMER_ID_PATTERN = re.compile(
        r'(?:Thẻ khách hàng|Mã KH|Mã khách hàng|Mã danh bộ|Số danh bộ)[:\-]?\s*([A-Za-z0-9]+)',
        re.IGNORECASE
    )

    def extract_fields(self, raw_text: str, doc_type: str = "UNKNOWN") -> dict:
        if not raw_text:
            return {}

        normalized_type = (doc_type or "UNKNOWN").upper()

        if normalized_type in ("VAT_INVOICE", "HOA_DON_VAT"):
            return self._extract_vat_invoice(raw_text)
        elif normalized_type in ("ELECTRICITY_BILL", "HOA_DON_DIEN"):
            return self._extract_utility_bill(raw_text)
        elif normalized_type in ("WATER_BILL", "HOA_DON_NUOC"):
            return self._extract_utility_bill(raw_text)
        else:
            # Loại chứng từ chưa có quy tắc Regex riêng -> thử fallback VAT
            # (giữ hành vi an toàn thay vì trả rỗng hoàn toàn)
            return self._extract_vat_invoice(raw_text)

    def _extract_vat_invoice(self, raw_text: str) -> dict:
        fields = {}

        mst_match = self.MST_PATTERN.search(raw_text)
        if mst_match:
            fields['tax_code'] = mst_match.group(1).replace(' ', '').replace('-', '')

        amount = self._parse_amount(raw_text)
        if amount is not None:
            fields['total_amount'] = amount

        return fields

    def _extract_utility_bill(self, raw_text: str) -> dict:
        fields = {}

        cust_match = self.CUSTOMER_ID_PATTERN.search(raw_text)
        if cust_match:
            fields['customer_id'] = cust_match.group(1)

        amount = self._parse_amount(raw_text)
        if amount is not None:
            fields['total_amount'] = amount

        return fields

    def _parse_amount(self, raw_text: str):
        money_match = self.AMOUNT_PATTERN.search(raw_text)
        if not money_match:
            return None
        clean_money = re.sub(r'[^\d]', '', money_match.group(1))
        return int(clean_money) if clean_money else None


# Khởi tạo instance để ml_router import
extractor_service = DocumentExtractor()
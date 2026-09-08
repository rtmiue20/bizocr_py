import re

class DocumentExtractor:
    # Định nghĩa field bắt buộc theo từng loại chứng từ, dùng để tính confidence thực tế
    REQUIRED_FIELDS = {
        "VAT_INVOICE": ["tax_code", "total_amount"],
        "HOA_DON_VAT": ["tax_code", "total_amount"],
        "ELECTRICITY_BILL": ["customer_id", "total_amount"],
        "HOA_DON_DIEN": ["customer_id", "total_amount"],
        "WATER_BILL": ["customer_id", "total_amount"],
        "HOA_DON_NUOC": ["customer_id", "total_amount"],
    }

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
            fields = self._extract_vat_invoice(raw_text)
        elif normalized_type in ("ELECTRICITY_BILL", "HOA_DON_DIEN", "WATER_BILL", "HOA_DON_NUOC"):
            fields = self._extract_utility_bill(raw_text)
        else:
            fields = self._extract_vat_invoice(raw_text)

        # Gắn cờ field nào bị thiếu so với danh sách bắt buộc, dùng để tính confidence phía router
        required = self.REQUIRED_FIELDS.get(normalized_type, [])
        fields["_missing_required_fields"] = [f for f in required if f not in fields or fields.get(f) is None]

        return fields

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
        # Loại bỏ tất cả ký tự không phải số (bao gồm cả dấu chấm, phẩy)
        clean_money = re.sub(r'[^\d]', '', money_match.group(1))
        return int(clean_money) if clean_money else None


# Khởi tạo instance để ml_router import
extractor_service = DocumentExtractor()


# Phần test độc lập: Chỉ chạy khi thực thi trực tiếp file này (python document_extractor.py)
if __name__ == "__main__":
    test_cases = [
        {
            "doc_type": "HOA_DON_VAT",
            "text": "Công ty TNHH ABC\nMã số thuế: 0101234567-001\nCộng tiền hàng: 1.500.000 đ\nThuế VAT: 10%"
        },
        {
            "doc_type": "ELECTRICITY_BILL",
            "text": "Điện lực Cầu Giấy\nMã khách hàng: PE0123456789\nChỉ số cũ: 100\nTổng cộng tiền thanh toán: 850,500 VNĐ"
        },
        {
            "doc_type": "WATER_BILL",
            "text": "Công ty nước sạch Hà Nội\nSố danh bộ: 09A12345\nThành tiền: 125.000"
        },
        {
            "doc_type": "UNKNOWN_DOC",
            "text": "Biên lai thu tiền\nMST: 0312456789\nTổng tiền thanh toán: 50.000"
        }
    ]

    print("--- CHẠY THỬ NGHIỆM TRÍCH XUẤT ---")
    for idx, case in enumerate(test_cases, 1):
        result = extractor_service.extract_fields(case["text"], case["doc_type"])
        print(f"\nTest {idx} - Loại: {case['doc_type']}")
        print(f"Dữ liệu đầu vào: {case['text'].replace(chr(10), ' | ')}")
        print(f"Kết quả: {result}")
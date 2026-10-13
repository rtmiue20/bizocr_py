import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from python_service.config import settings
from python_service.normalizer.text_fixer import text_fixer
from python_service.normalizer.qwen_normalizer import QwenTextNormalizer

logger = logging.getLogger(__name__)


# Danh mục chức danh thông dụng tiếng Việt và tiếng Anh
TITLE_KEYWORDS = [
    "chủ tịch", "tổng giám đốc", "phó tổng giám đốc", "giám đốc điều hành", "giám đốc",
    "phó giám đốc", "trưởng phòng", "phó phòng", "quản lý", "trưởng nhóm", "chuyên viên",
    "kỹ sư", "kiến trúc sư", "kế toán trưởng", "kế toán", "luật sư", "bác sĩ", "dược sĩ",
    "ceo", "cto", "cfo", "coo", "cmo", "cio", "founder", "co-founder", "co founder",
    "managing director", "general manager", "director", "manager", "team leader", "leader",
    "executive", "officer", "supervisor", "representative", "specialist", "expert",
    "consultant", "engineer", "architect", "developer", "designer", "sales", "accountant",
]

# Các tiền tố nhận diện công ty / tổ chức
COMPANY_PREFIXES = [
    r"công\s+ty\b",
    r"cty\b",
    r"tập\s+đoàn\b",
    r"doanh\s+nghiệp\b",
    r"chi\s+nhánh\b",
    r"văn\s+phòng\s+đại\s+diện\b",
    r"viện\b",
    r"trung\s+tâm\b",
    r"hợp\s+tác\s+xã\b",
    r"ngân\s+hàng\b",
    r"tổng\s+công\s+ty\b",
    r"\bcorp\b",
    r"\bcorporation\b",
    r"\bgroup\b",
    r"\bjsc\b",
    r"\bltd\b",
    r"\bco\.,?\s*ltd\b",
    r"\binc\b",
    r"\benterprise\b",
    r"\bholding\b",
]

COMPANY_INDICATORS = [
    "tnhh", "cổ phần", "cp", "mtv", "liên doanh", "thương mại", "dịch vụ",
    "sản xuất", "xây dựng", "đầu tư", "quốc tế", "xuất nhập khẩu", "logistics"
]


class NamecardFieldTagger:
    """
    Trích xuất và chuẩn hóa các thực thể danh thiếp (Namecard Field Tagger):
    - Tầng 1: Luật heuristic + Regex tối ưu hóa cho danh thiếp tiếng Việt.
    - Tầng 2: Dynamic Few-Shot Prompting với Qwen2.5:3B nếu thiếu thông tin quan trọng.
    - Tầng 3: Sửa lỗi chính tả qua TextFixer và Grounding Check chống ảo giác.
    """

    def __init__(self):
        self.qwen = QwenTextNormalizer()

    @staticmethod
    def extract_phones(text: str) -> List[str]:
        """Trích xuất danh sách số điện thoại định dạng Việt Nam."""
        phones = []
        # Regex hỗ trợ đầu số VN 10 số (03x, 05x, 07x, 08x, 09x) và mã quốc gia +84
        pattern = r"(?:\+?84|0)(?:[3|5|7|8|9]\d{8}|2\d{9}|2\d{8})"
        # Loại bỏ các ký tự phân cách thông dụng trước khi dò
        clean_search = re.sub(r"[\.\-\s\(\)]", "", text)
        for match in re.finditer(pattern, clean_search):
            p = match.group()
            # Chuẩn hóa về đầu số 0
            if p.startswith("+84"):
                p = "0" + p[3:]
            elif p.startswith("84") and len(p) >= 11:
                p = "0" + p[2:]
            if p not in phones:
                phones.append(p)

        # Bổ sung regex quét trực tiếp từng dòng để giữ format nếu clean_search sót
        line_pattern = r"(?:\+84|84|0)[1-9](?:[\s\.\-]?\d){7,9}\b"
        for match in re.finditer(line_pattern, text):
            raw_p = re.sub(r"[^\d+]", "", match.group())
            if raw_p.startswith("+84"):
                norm = "0" + raw_p[3:]
            elif raw_p.startswith("84") and len(raw_p) >= 11:
                norm = "0" + raw_p[2:]
            else:
                norm = raw_p
            if 9 <= len(norm) <= 11 and norm not in phones:
                phones.append(norm)

        return phones

    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """Trích xuất và sửa lỗi email thông dụng."""
        # Bước 1: Sửa các lỗi OCR @ thành O, o, 0 hoặc khoảng trắng
        fixed_text = QwenTextNormalizer.fix_broken_emails(text)
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        emails = []
        for match in re.finditer(email_pattern, fixed_text):
            em = match.group().lower()
            if em not in emails:
                emails.append(em)
        return emails

    @staticmethod
    def extract_websites(text: str) -> List[str]:
        """Trích xuất địa chỉ website / domain."""
        web_pattern = r"(?:https?:\/\/|www\.)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\/[^\s]*)?|[a-zA-Z0-9.-]+\.(?:com\.vn|vn|com|net|org|edu\.vn|io)\b"
        websites = []
        for match in re.finditer(web_pattern, text, re.IGNORECASE):
            w = match.group().strip()
            # Bỏ qua nếu là đuôi của email
            if "@" in w:
                continue
            if not w.startswith("http://") and not w.startswith("https://") and not w.startswith("www."):
                w = "https://" + w
            if w not in websites:
                websites.append(w)
        return websites

    @staticmethod
    def extract_tax_code(text: str) -> Optional[str]:
        """Trích xuất mã số thuế (10 chữ số hoặc 13 chữ số có gạch nối)."""
        pattern = r"\b\d{10}(?:-\d{3})?\b"
        match = re.search(pattern, text)
        return match.group() if match else None

    @classmethod
    def extract_company_name(cls, lines: List[str]) -> Optional[str]:
        """Phát hiện tên công ty từ các dòng văn bản."""
        for line in lines:
            line_strip = line.strip()
            line_lower = line_strip.lower()
            if len(line_strip) < 6:
                continue

            # Kiểm tra tiền tố công ty
            for prefix in COMPANY_PREFIXES:
                if re.search(prefix, line_lower):
                    # Làm sạch các tiền tố nhãn thừa như: "Tên công ty:", "Company:"
                    cleaned = re.sub(r"^(?:tên\s+công\s+ty|company|đơn\s+vị)\s*[:\-]\s*", "", line_strip, flags=re.I)
                    return cleaned.strip()

            # Kiểm tra các từ khóa đặc trưng (TNHH, Cổ Phần...)
            if any(ind in line_lower for ind in COMPANY_INDICATORS) and len(line_strip.split()) >= 3:
                cleaned = re.sub(r"^(?:tên\s+công\s+ty|company|đơn\s+vị)\s*[:\-]\s*", "", line_strip, flags=re.I)
                return cleaned.strip()

        return None

    @classmethod
    def extract_title(cls, lines: List[str]) -> Optional[str]:
        """Phát hiện chức danh nghề nghiệp."""
        for line in lines:
            line_strip = line.strip()
            line_lower = line_strip.lower()
            if len(line_strip) < 2 or len(line_strip) > 60:
                continue

            # Bỏ qua dòng có chứa số điện thoại hoặc email
            if "@" in line_strip or re.search(r"\d{4,}", line_strip):
                continue

            for kw in TITLE_KEYWORDS:
                if kw in line_lower:
                    # Làm sạch nhãn "Chức vụ: ", "Title: "
                    cleaned = re.sub(r"^(?:chức\s+vụ|title|position|role)\s*[:\-]\s*", "", line_strip, flags=re.I)
                    return cleaned.strip()

        return None

    @classmethod
    def extract_person_name(cls, lines: List[str], title_line_idx: Optional[int], company_name: Optional[str]) -> Optional[str]:
        """
        Nhận diện họ tên người đại diện trên danh thiếp.
        Ưu tiên các dòng nằm kề chức danh, chữ viết hoa, 2-4 từ, không phải công ty/địa chỉ.
        """
        exclude_words = [
            "công ty", "tnhh", "cổ phần", "địa chỉ", "address", "phone", "email",
            "website", "hotline", "tax", "mst", "tel", "fax", "chi nhánh", "group"
        ]

        candidates = []

        for idx, line in enumerate(lines):
            line_strip = line.strip()
            line_lower = line_strip.lower()

            if len(line_strip) < 3 or len(line_strip) > 35:
                continue

            if any(ex in line_lower for ex in exclude_words):
                continue
            if company_name and (line_strip in company_name or company_name in line_strip):
                continue
            if "@" in line_strip or re.search(r"\d{3,}", line_strip):
                continue

            words = line_strip.split()
            if not (2 <= len(words) <= 5):
                continue

            score = 0
            # Điểm cộng nếu là chữ IN HOA TOÀN BỘ (phổ biến cho tên trên namecard)
            if line_strip.isupper():
                score += 3
            elif all(w[0].isupper() for w in words if w):
                score += 2

            # Điểm cộng nếu đứng sát dòng chức vụ
            if title_line_idx is not None:
                dist = abs(idx - title_line_idx)
                if dist == 1:
                    score += 4
                elif dist == 2:
                    score += 2

            # Loại bỏ các chuỗi là chức vụ đơn thuần
            if any(kw == line_lower for kw in TITLE_KEYWORDS):
                continue

            candidates.append((score, line_strip))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None

    @classmethod
    def extract_address(cls, lines: List[str]) -> Optional[str]:
        """Phát hiện địa chỉ trụ sở / văn phòng."""
        addr_labels = ["địa chỉ", "đ/c", "đc", "address", "add", "trụ sở", "vpgd", "văn phòng", "tầng", "tòa nhà"]
        addr_lines = []

        for line in lines:
            line_strip = line.strip()
            line_lower = line_strip.lower()

            if any(lbl in line_lower for lbl in addr_labels) or re.search(r"\b(quận|huyện|phường|xã|tp\.|thành phố|tỉnh|đường|phố)\b", line_lower):
                # Bỏ qua nếu chỉ là email hoặc số điện thoại
                if "@" in line_strip or (re.search(r"\d{9,}", line_strip) and not any(lbl in line_lower for lbl in addr_labels)):
                    continue
                clean = re.sub(r"^(?:địa\s+chỉ|đ/c|đc|address|add|trụ\s+sở|vpgd)\s*[:\-]\s*", "", line_strip, flags=re.I)
                addr_lines.append(clean.strip())

        return ", ".join(addr_lines) if addr_lines else None

    async def _extract_via_qwen_fewshot(self, raw_text: str, current_fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gọi mô hình Qwen2.5:3B kèm Few-Shot Examples khi các trường chính (name, company) bị thiếu.
        """
        # Các ví dụ mẫu chuẩn hóa danh thiếp chất lượng cao
        few_shot_examples = [
            {
                "input": "CÔNG TY CỔ PHẦN CÔNG NGHỆ BIZTECH\nNGUYỄN VĂN HẢI\nGiám đốc Điều hành (CEO)\nHotline: 0988.123.456\nEmail: hai.nguyen@biztech.vn\nWebsite: https://biztech.vn\nAdd: Tầng 6, Keangnam Landmark, Nam Từ Liêm, Hà Nội",
                "output": {
                    "name": "Nguyễn Văn Hải",
                    "title": "Giám đốc Điều hành (CEO)",
                    "company": "Công ty Cổ phần Công nghệ Biztech",
                    "phones": ["0988123456"],
                    "emails": ["hai.nguyen@biztech.vn"],
                    "websites": ["https://biztech.vn"],
                    "address": "Tầng 6, Keangnam Landmark, Nam Từ Liêm, Hà Nội"
                }
            },
            {
                "input": "LOGISTICS TOÀN CẦU JSC\nTrần Thị Mai\nTrưởng phòng Kinh doanh Xuất nhập khẩu\nTel: (028) 3822 5566 - Mob: 0909 789 654\nEmail: maitran 0global-logistics.com.vn\nMST: 0312456789\nĐ/c: 123 Nguyễn Văn Linh, Quận 7, TP. Hồ Chí Minh",
                "output": {
                    "name": "Trần Thị Mai",
                    "title": "Trưởng phòng Kinh doanh Xuất nhập khẩu",
                    "company": "Logistics Toàn Cầu JSC",
                    "phones": ["02838225566", "0909789654"],
                    "emails": ["maitran@global-logistics.com.vn"],
                    "websites": [],
                    "address": "123 Nguyễn Văn Linh, Quận 7, TP. Hồ Chí Minh",
                    "taxCode": "0312456789"
                }
            }
        ]

        system_prompt = (
            "Bạn là trợ lý AI chuyên bóc tách thông tin danh thiếp (namecard/business card) tiếng Việt.\n"
            "Nhiệm vụ: Phân tích văn bản OCR và trích xuất cấu trúc thực thể JSON.\n"
            "QUY TẮC:\n"
            "1. Chỉ trích xuất thông tin có trong văn bản, TUYỆT ĐỐI KHÔNG tự bịa ra thông tin mới (Grounding).\n"
            "2. Trả về đúng định dạng JSON không kèm lời dẫn hay markdown.\n"
            "3. Các trường cần trả về: name (Họ tên), title (Chức vụ), company (Tên công ty), phones (mảng sđt), emails (mảng email), websites (mảng web), address (Địa chỉ)."
        )

        few_shot_prompt = ""
        for ex in few_shot_examples:
            few_shot_prompt += f"<|im_start|>user\n{ex['input']}<|im_end|>\n<|im_start|>assistant\n{json.dumps(ex['output'], ensure_ascii=False)}<|im_end|>\n"

        user_content = f"Văn bản danh thiếp cần trích xuất:\n\"\"\"\n{raw_text}\n\"\"\""
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n{few_shot_prompt}<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

        import httpx
        payload = {
            "model": self.qwen.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 1024}
        }

        try:
            async with httpx.AsyncClient(timeout=self.qwen.timeout) as client:
                res = await client.post(self.qwen.ollama_url, json=payload)
                if res.status_code == 200:
                    raw_out = res.json().get("response", "").strip()
                    # Làm sạch markdown json
                    clean_json = re.sub(r"^```[a-zA-Z]*\n?", "", raw_out)
                    clean_json = re.sub(r"\n?```$", "", clean_json).strip()
                    data = json.loads(clean_json)
                    return data
        except Exception as e:
            logger.warning(f"Lỗi khi gọi Qwen Few-shot: {e}")

        return {}

    async def tag_fields(
        self,
        raw_text: str,
        qr_vcard: Optional[Dict[str, Any]] = None,
        owner_company_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Quy trình trích xuất tổng hợp đa phương thức:
        1. Khởi tạo từ QR vCard (nếu có).
        2. Bổ sung / trích xuất bằng luật regex & heuristics.
        3. Nếu thiếu Name hoặc Company -> Gọi Qwen Few-shot.
        4. Chuẩn hóa qua TextFixer và tính Completeness score.
        """
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        # Khởi tạo từ vCard nếu có
        extracted: Dict[str, Any] = {
            "name": None,
            "title": None,
            "company": None,
            "phones": [],
            "emails": [],
            "websites": [],
            "address": None,
            "taxCode": None,
            "industry": None,
            "industryConfidence": 0.0,
            "extractionMethod": "REGEX_HEURISTICS",
        }

        if qr_vcard:
            extracted["name"] = qr_vcard.get("name")
            extracted["title"] = qr_vcard.get("title")
            extracted["company"] = qr_vcard.get("company")
            extracted["phones"] = qr_vcard.get("phones", [])
            extracted["emails"] = qr_vcard.get("emails", [])
            extracted["websites"] = qr_vcard.get("websites", [])
            extracted["address"] = qr_vcard.get("address")
            extracted["extractionMethod"] = "QR_VCARD"

        # Bổ sung các trường từ Regex trên raw text
        regex_phones = self.extract_phones(raw_text)
        for p in regex_phones:
            if p not in extracted["phones"]:
                extracted["phones"].append(p)

        regex_emails = self.extract_emails(raw_text)
        for em in regex_emails:
            if em not in extracted["emails"]:
                extracted["emails"].append(em)

        regex_webs = self.extract_websites(raw_text)
        for w in regex_webs:
            if w not in extracted["websites"]:
                extracted["websites"].append(w)

        if not extracted["taxCode"]:
            extracted["taxCode"] = self.extract_tax_code(raw_text)

        if not extracted["company"]:
            extracted["company"] = self.extract_company_name(lines)

        title_line_idx = None
        if not extracted["title"]:
            for idx, l in enumerate(lines):
                if any(kw in l.lower() for kw in TITLE_KEYWORDS):
                    title_line_idx = idx
                    break
            extracted["title"] = self.extract_title(lines)

        if not extracted["name"]:
            extracted["name"] = self.extract_person_name(lines, title_line_idx, extracted["company"])

        if not extracted["address"]:
            extracted["address"] = self.extract_address(lines)

        # Kiểm tra độ đầy đủ: Cần có Name và (Company hoặc Phone hoặc Email)
        has_primary = bool(extracted["name"] and (extracted["company"] or extracted["phones"] or extracted["emails"]))
        
        # Fast-Path: Nếu đã có đủ các trường quan trọng từ Regex/Heuristics và chất lượng Text cao, bỏ qua LLM
        is_complete_enough = bool(
            extracted["name"] and 
            extracted["company"] and 
            (extracted["phones"] or extracted["emails"]) and
            extracted["title"] and
            extracted["address"]
        )

        llm_invoked = False
        if not is_complete_enough and not has_primary and len(raw_text) >= 15:
            # Gọi LLM Qwen Few-shot để bù đắp
            llm_res = await self._extract_via_qwen_fewshot(raw_text, extracted)
            if llm_res:
                llm_invoked = True
                extracted["extractionMethod"] = "HYBRID_LLM"
                if not extracted["name"] and llm_res.get("name"):
                    extracted["name"] = llm_res.get("name")
                if not extracted["company"] and llm_res.get("company"):
                    extracted["company"] = llm_res.get("company")
                if not extracted["title"] and llm_res.get("title"):
                    extracted["title"] = llm_res.get("title")
                if not extracted["address"] and llm_res.get("address"):
                    extracted["address"] = llm_res.get("address")
                for p in llm_res.get("phones", []):
                    if p not in extracted["phones"]:
                        extracted["phones"].append(p)
                for em in llm_res.get("emails", []):
                    if em not in extracted["emails"]:
                        extracted["emails"].append(em)

        # Bước chuẩn hóa qua từ điển TextFixer
        if extracted.get("company"):
            extracted["company"] = text_fixer.fix_field("company", extracted["company"], owner_company_id)
        if extracted.get("title"):
            extracted["title"] = text_fixer.fix_field("title", extracted["title"], owner_company_id)
        if extracted.get("name"):
            extracted["name"] = text_fixer.fix_field("name", extracted["name"], owner_company_id)

        # Tính Completeness Score
        key_fields = ["name", "company", "phones", "emails", "title"]
        filled_count = sum(1 for k in key_fields if extracted.get(k))
        completeness = round(filled_count / len(key_fields), 2)

        return {
            "status": "ACCEPTED" if completeness >= 0.4 else "NEEDS_REVIEW",
            "extractedFields": extracted,
            "completenessScore": completeness,
            "llmInvoked": llm_invoked,
            "extractionMethod": extracted["extractionMethod"],
        }


namecard_tagger = NamecardFieldTagger()

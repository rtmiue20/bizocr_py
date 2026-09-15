import httpx
import logging
import re
import unicodedata
from python_service.config import settings

logger = logging.getLogger(__name__)


class QwenTextNormalizer:
    """
    Module Qwen2.5:3B: OCR thô -> Văn bản chuẩn (Chuẩn hóa lỗi chính tả OCR tiếng Việt, giữ nguyên số liệu)
    """

    def __init__(self):
        self.ollama_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        self.model_name = settings.OLLAMA_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT_SECONDS

    @staticmethod
    def fix_broken_emails(text: str) -> str:
        """
        Lớp phòng ngự 1: Tìm và sửa các email bị OCR đọc sai chữ @ thành O, o, 0 hoặc khoảng trắng.
        Chỉ áp dụng cho các TLD (Top-Level Domain) phổ biến để tránh nhận diện nhầm.
        """
        if not text:
            return ""

        # Regex tìm: [local_part] + [khoảng trắng/O/o/0] + [domain] + [.tld phổ biến]
        # TLD whitelist: vn, com, net, org, edu.vn, com.vn, gov.vn
        pattern = r"([a-zA-Z0-9._%+-]+)\s*[Oo0\s]\s*([a-zA-Z0-9.-]+\.(?:com|vn|net|org|edu\.vn|com\.vn|gov\.vn))"

        # \1 là nhóm 1 (local_part), \2 là nhóm 2 (domain.tld)
        return re.sub(pattern, r"\1@\2", text)

    @staticmethod
    def rule_based_fallback_clean(text: str) -> str:
        """
        Chuẩn hóa văn bản bằng quy tắc (Rule-based Fallback) khi LLM không khả dụng
        """
        if not text:
            return ""

        # 1. Chuẩn hóa dạng Unicode (NFC)
        normalized = unicodedata.normalize("NFC", text)

        # 2. Xóa các ký tự điều khiển lạ hoặc rác scan
        normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", normalized)

        # 3. Chuẩn hóa dấu 2 chấm và khoảng trắng thừa quanh nhãn
        normalized = re.sub(r"[ \t]+:[ \t]*", ": ", normalized)
        normalized = re.sub(r"[ \t]+", " ", normalized)

        # 4. Sửa các lỗi OCR thông dụng trong từ khóa kế toán
        replacements = [
            (r"(?i)\bMa\s*so\s*thue\b", "Mã số thuế"),
            (r"(?i)\bHoa\s*don\s*gia\s*tri\s*gia\s*tang\b", "Hóa đơn giá trị gia tăng"),
            (r"(?i)\bTong\s*tien\s*thanh\s*toan\b", "Tổng tiền thanh toán"),
            (r"(?i)\bCong\s*tien\s*hang\b", "Cộng tiền hàng"),
            (r"(?i)\bTien\s*thue\s*GTGT\b", "Tiền thuế GTGT"),
            (r"(?i)\bHop\s*dong\s*kinh\s*te\b", "Hợp đồng kinh tế"),
            (r"(?i)\bPhieu\s*thanh\s*toan\b", "Phiếu thanh toán"),
            (r"(?i)\bBien\s*ban\s*nghiem\s*thu\b", "Biên bản nghiệm thu"),
            (r"(?i)\bGiay\s*chung\s*nhan\b", "Giấy chứng nhận"),
        ]
        for pat, rep in replacements:
            normalized = re.sub(pat, rep, normalized)

        # 5. Dọn dẹp dòng trống thừa
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        return "\n".join(lines)

    async def normalize_text(self, raw_ocr_text: str) -> str:
        """
        Gửi đoạn OCR thô sang mô hình qwen2.5:3b để sửa lỗi OCR tiếng Việt và định dạng thành Văn bản chuẩn.
        """
        if not raw_ocr_text or len(raw_ocr_text.strip()) < 5:
            return raw_ocr_text or ""

        # Tiền xử lý: Dùng Regex quét và sửa email trước khi đưa vào LLM
        preprocessed_text = self.fix_broken_emails(raw_ocr_text)

        system_prompt = (
            "Bạn là trợ lý AI chuyên chuẩn hóa và chỉnh sửa lỗi OCR cho chứng từ, hóa đơn kế toán Việt Nam.\n"
            "Nhiệm vụ: Đọc văn bản OCR thô và sửa các lỗi chính tả tiếng Việt do scan/nhận diện sai, phục hồi dấu thanh chính xác.\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. TUYỆT ĐỐI GIỮ NGUYÊN tất cả các số liệu: Mã số thuế (MST), Số hóa đơn, Số tiền, Ngày tháng, Số tài khoản, Mã khách hàng.\n"
            "2. Khôi phục đúng các từ khóa kế toán (vd: 'Mã số thuế', 'Hóa đơn GTGT', 'Tổng tiền thanh toán', 'Ký hiệu', 'Mẫu số').\n"
            "3. Loại bỏ ký tự rác, nhiễu scan (như ~~~, ___, ký hiệu vô nghĩa).\n"
            "4. CHÚ Ý SỬA EMAIL: Nếu thấy chuỗi có dạng <tên> <ký_tự_lạ/khoảng_trắng> <domain>.<tld> (ví dụ: 'tientran Ohitien.vn', 'nguyen.a 0gmail.com'), hãy tự sửa thành định dạng email chuẩn bằng cách thay bằng dấu '@' (ví dụ: 'tientran@hitien.vn').\n"
            "5. Giữ nguyên cấu trúc từng dòng của văn bản, không tóm tắt, không thêm giải thích hay markdown formatting.\n"
            "6. CHỈ TRẢ VỀ DUY NHẤT nội dung văn bản sau khi đã chuẩn hóa."
        )

        user_content = f"Văn bản OCR thô cần chuẩn hóa:\n\"\"\"\n{preprocessed_text}\n\"\"\""

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.9,
                "num_predict": 2048
            }
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(
                    f"Đang gửi OCR text ({len(preprocessed_text)} chars) tới Ollama model '{self.model_name}'...")
                response = await client.post(self.ollama_url, json=payload)

                if response.status_code == 200:
                    data = response.json()
                    normalized = data.get("response", "").strip()
                    if normalized:
                        # Làm sạch markdown nếu LLM tự chèn
                        clean_text = re.sub(r"^```[a-zA-Z]*\n", "", normalized)
                        clean_text = re.sub(r"\n```$", "", clean_text).strip()
                        logger.info("Chuẩn hóa văn bản thành công bằng Qwen2.5:3B!")
                        return clean_text
                else:
                    logger.warning(f"Ollama trả về status {response.status_code}: {response.text}")
        except Exception as e:
            logger.warning(f"Không thể kết nối tới Ollama ({e}). Sử dụng Rule-based Normalization fallback.")

        return self.rule_based_fallback_clean(preprocessed_text)
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class StructuralHeuristics:

    @staticmethod
    def classify_candidate(
        aspect_ratio: float,
        raw_text: str,
        boxes: List[Any]
    ) -> Dict[str, Any]:
        """
        Phân loại nhanh loại chứng từ dựa trên:
        1. Aspect Ratio (Tỷ lệ khung hình sau crop)
        2. Mật độ văn bản (Độ dài raw_text)
        3. Số lượng Bounding Boxes (Cấu trúc dòng/bảng)
        """
        text_len = len(raw_text.strip()) if raw_text else 0
        box_count = len(boxes) if boxes else 0

        # Ngưỡng Aspect Ratio của Namecard (Ngang: ~1.45 - 2.10 | Đứng: ~0.48 - 0.72)
        is_horizontal_namecard_ratio = 1.45 <= aspect_ratio <= 2.10
        is_vertical_namecard_ratio = 0.48 <= aspect_ratio <= 0.72
        is_namecard_ratio = is_horizontal_namecard_ratio or is_vertical_namecard_ratio

        # Ngưỡng cấu trúc hình học đặc trưng của Namecard
        is_short_text = text_len < 350
        is_few_boxes = box_count < 25

        # 1. CANDIDATE: NAME_CARD
        if is_namecard_ratio and is_short_text and is_few_boxes:
            logger.info(f"[Heuristic] Phát hiện candidate NAME_CARD (Ratio: {aspect_ratio}, TextLen: {text_len}, Boxes: {box_count})")
            return {
                "candidateType": "NAME_CARD",
                "confidence": 0.85,
                "reason": "Aspect ratio namecard + văn bản ngắn + ít bounding boxes"
            }

        # 2. CANDIDATE: PAGE_DOCUMENT (A4: Hóa đơn, Hợp đồng, Phiếu thu...)
        is_a4_ratio = 0.65 <= aspect_ratio <= 1.45
        if is_a4_ratio or text_len >= 350 or box_count >= 25:
            logger.info(f"[Heuristic] Phát hiện candidate PAGE_DOCUMENT (Ratio: {aspect_ratio}, TextLen: {text_len}, Boxes: {box_count})")
            return {
                "candidateType": "PAGE_DOCUMENT",
                "confidence": 0.75,
                "reason": "Khung hình dạng trang/A4 + mật độ chữ và bảng biểu cao"
            }

        # 3. UNKNOWN
        return {
            "candidateType": "UNKNOWN",
            "confidence": 0.30,
            "reason": "Không đủ dấu hiệu định hình rõ ràng"
        }
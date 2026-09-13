import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from python_service.config import settings

logger = logging.getLogger(__name__)

class PaddleOcrEngine:
    """
    Module PaddleOCR: Ảnh sạch -> OCR thô (Bounding Boxes + Text lines + Spatial Sorting)
    """

    def __init__(self):
        self._ocr = None
        self._initialized = False
        self._engine_name = "paddleocr"

    def _get_ocr(self):
        if not self._initialized:
            try:
                from paddleocr import PaddleOCR
                logger.info(f"Đang khởi tạo PaddleOCR (lang={settings.OCR_LANG}, use_gpu={settings.USE_GPU})...")
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=settings.OCR_LANG,
                    use_gpu=settings.USE_GPU,
                    show_log=False
                )
                self._engine_name = "paddleocr"
                self._initialized = True
                logger.info("PaddleOCR đã khởi tạo thành công!")
            except Exception as e:
                logger.warning(f"Không thể khởi tạo PaddleOCR ({e}). Đang thử chuyển sang RapidOCR/Fallback...")
                try:
                    from rapidocr_onnxruntime import RapidOCR
                    self._ocr = RapidOCR()
                    self._engine_name = "rapidocr"
                    self._initialized = True
                    logger.info("RapidOCR đã khởi tạo thành công!")
                except Exception as ex:
                    logger.error(f"Cả PaddleOCR và RapidOCR đều không khả dụng: {ex}")
                    self._ocr = None
                    self._initialized = True
        return self._ocr

    @staticmethod
    def _sort_reading_order(ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sắp xếp các bounding box theo thứ tự đọc tự nhiên của con người (Top-to-Bottom, Left-to-Right)
        """
        if not ocr_results:
            return []

        # Tính toán Y_center, X_min, Height cho mỗi box
        boxes_with_metrics = []
        for item in ocr_results:
            pts = item["points"] # [x1,y1, x2,y2, x3,y3, x4,y4]
            xs = pts[0::2]
            ys = pts[1::2]
            y_min, y_max = min(ys), max(ys)
            x_min, x_max = min(xs), max(xs)
            y_center = (y_min + y_max) / 2.0
            height = max(1.0, y_max - y_min)
            boxes_with_metrics.append({
                "item": item,
                "y_center": y_center,
                "x_min": x_min,
                "height": height
            })

        # Sắp xếp thô theo Y
        boxes_with_metrics.sort(key=lambda b: b["y_center"])

        # Gom nhóm theo từng dòng văn bản
        lines = []
        current_line = []
        current_line_y = None
        current_line_h = None

        for b in boxes_with_metrics:
            if current_line_y is None:
                current_line = [b]
                current_line_y = b["y_center"]
                current_line_h = b["height"]
            else:
                # Nếu độ lệch Y_center < 0.6 * chiều cao dòng thì thuộc cùng 1 dòng
                threshold = max(current_line_h, b["height"]) * 0.6
                if abs(b["y_center"] - current_line_y) <= threshold:
                    current_line.append(b)
                    # Cập nhật Y_center trung bình của dòng
                    current_line_y = sum(x["y_center"] for x in current_line) / len(current_line)
                    current_line_h = sum(x["height"] for x in current_line) / len(current_line)
                else:
                    # Sắp xếp các box trong dòng hiện tại từ trái qua phải
                    current_line.sort(key=lambda x: x["x_min"])
                    lines.append(current_line)
                    current_line = [b]
                    current_line_y = b["y_center"]
                    current_line_h = b["height"]

        if current_line:
            current_line.sort(key=lambda x: x["x_min"])
            lines.append(current_line)

        # Gộp tất cả theo thứ tự dòng
        sorted_results = []
        for line in lines:
            for b in line:
                sorted_results.append(b["item"])

        return sorted_results

    def recognize(self, image: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]], str]:
        """
        Thực hiện OCR trên ảnh sạch:
        Returns:
            raw_text: Chuỗi văn bản thô ghép từ các dòng
            confidence: Điểm tin cậy trung bình (0.0 - 1.0)
            boxes: Danh sách bounding boxes [{"text": ..., "points": [x1,y1,...]}]
            engine_name: Tên engine ("paddleocr")
        """
        engine = self._get_ocr()
        if engine is None:
            return "", 0.0, [], "paddleocr"

        parsed_boxes = []
        confidences = []

        try:
            if self._engine_name == "paddleocr":
                results = engine.ocr(image, cls=True)
                if results and len(results) > 0 and results[0] is not None:
                    for line in results[0]:
                        box_points_2d, (text, conf) = line
                        # Flatten 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] -> [x1,y1, x2,y2, x3,y3, x4,y4]
                        flat_points = [int(coord) for pt in box_points_2d for coord in pt]
                        parsed_boxes.append({
                            "text": text.strip(),
                            "points": flat_points
                        })
                        confidences.append(float(conf))
            elif self._engine_name == "rapidocr":
                results, elapse = engine(image)
                if results:
                    for line in results:
                        box_points_2d, text, conf = line
                        flat_points = [int(coord) for pt in box_points_2d for coord in pt]
                        parsed_boxes.append({
                            "text": text.strip(),
                            "points": flat_points
                        })
                        confidences.append(float(conf))
        except Exception as e:
            logger.error(f"Lỗi khi chạy OCR: {e}", exc_info=True)
            return "", 0.0, [], self._engine_name

        # Sắp xếp spatial reading order
        sorted_boxes = self._sort_reading_order(parsed_boxes)
        
        # Ghép thành raw_text thô
        raw_text_lines = [b["text"] for b in sorted_boxes if b.get("text")]
        raw_text = "\n".join(raw_text_lines)

        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        return raw_text, avg_confidence, sorted_boxes, self._engine_name

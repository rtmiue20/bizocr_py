import logging
import torch
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
from PIL import Image
from python_service.config import settings

logger = logging.getLogger(__name__)


class PaddleOcrEngine:
    """
    Module Hybrid OCR:
    - PaddleOCR Text Detector: Nhận diện vùng chứa chữ (Detection) trực tiếp từ DBNet
    - VietOCR: Đọc chữ tiếng Việt (Recognition) để đảm bảo chính xác dấu tiếng Việt
    """

    def __init__(self):
        self._det_engine = None
        self._rec_engine = None
        self._initialized = False
        self._engine_name = "hybrid_paddle_vietocr"

    def _init_engines(self):
        if not self._initialized:
            try:
                # Lưu ý: Trên Windows, cần import torch / VietOCR trước PaddleOCR để tránh xung đột DLL (shm.dll)
                from vietocr.tool.predictor import Predictor
                from vietocr.tool.config import Cfg
                from paddleocr import PaddleOCR

                # 1. Khởi tạo PaddleOCR chỉ phục vụ Text Detection
                max_dim = settings.OCR_MAX_IMAGE_DIM
                det_limit = (max_dim // 32) * 32
                if det_limit < max_dim:
                    det_limit += 32

                logger.info(f"Khởi tạo PaddleOCR Det-only (det_limit_side_len={det_limit})...")
                self._det_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=settings.USE_GPU,
                    det_limit_side_len=det_limit,
                    det_db_unclip_ratio=2.0,
                    show_log=False,
                    use_onnx=False,
                    enable_mkldnn=not settings.USE_GPU
                )

                # 2. VietOCR chuyên biệt cho Recognition tiếng Việt
                logger.info("Khởi tạo VietOCR vgg_transformer...")
                config = Cfg.load_config_from_name('vgg_transformer')
                config['device'] = 'cuda:0' if settings.USE_GPU else 'cpu'
                self._rec_engine = Predictor(config)

                self._initialized = True
                logger.info("Hybrid OCR khởi tạo thành công!")
            except Exception as e:
                logger.error(f"Lỗi khởi tạo Hybrid OCR: {e}", exc_info=True)
                self._initialized = False
                raise RuntimeError(f"Lỗi khởi tạo Hybrid OCR: {e}") from e

    def _crop_image_by_points(self, img: np.ndarray, points: list) -> Image.Image:
        """Cắt và nắn thẳng bounding box từ ảnh gốc bằng Perspective Transform"""
        pts = np.array(points, dtype=np.float32).reshape(4, 2)

        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _sort_reading_order(ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sắp xếp các bounding box theo thứ tự đọc tự nhiên (Top-to-Bottom, Left-to-Right)"""
        if not ocr_results:
            return []

        boxes_with_metrics = []
        for item in ocr_results:
            pts = item["points"]
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

        boxes_with_metrics.sort(key=lambda b: b["y_center"])

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
                threshold = max(current_line_h, b["height"]) * settings.OCR_LINE_MERGE_THRESHOLD_RATIO
                if abs(b["y_center"] - current_line_y) <= threshold:
                    current_line.append(b)
                    current_line_y = sum(x["y_center"] for x in current_line) / len(current_line)
                    current_line_h = sum(x["height"] for x in current_line) / len(current_line)
                else:
                    current_line.sort(key=lambda x: x["x_min"])
                    lines.append(current_line)
                    current_line = [b]
                    current_line_y = b["y_center"]
                    current_line_h = b["height"]

        if current_line:
            current_line.sort(key=lambda x: x["x_min"])
            lines.append(current_line)

        return [b["item"] for line in lines for b in line]

    def recognize(self, image: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]], str]:
        self._init_engines()
        if not self._det_engine or not self._rec_engine:
            raise RuntimeError("OCR Engine chưa được khởi tạo thành công.")

        parsed_boxes = []
        confidences = []

        try:
            # Gọi trực tiếp module text_detector để tránh lỗi 'if not dt_boxes:' của wrapper .ocr()
            dt_boxes, _ = self._det_engine.text_detector(image)

            if dt_boxes is not None and len(dt_boxes) > 0:
                valid_boxes = []
                cropped_images = []

                for box_points in dt_boxes:
                    if box_points is None or len(box_points) == 0:
                        continue

                    # Lọc nhiễu: Bỏ qua các box quá nhỏ (chấm viền, hạt scan)
                    pts = np.array(box_points, dtype=np.float32)
                    w = np.linalg.norm(pts[1] - pts[0])
                    h = np.linalg.norm(pts[3] - pts[0])
                    if w < 5.0 or h < 5.0 or (w * h) < 25.0:
                        continue

                    # Cắt ảnh nắn thẳng theo box
                    cropped_pil = self._crop_image_by_points(image, box_points)
                    valid_boxes.append(box_points)
                    cropped_images.append(cropped_pil)

                # Tối ưu hóa: Nhận diện song song theo Batch thay vì for loop tuần tự
                if cropped_images:
                    texts, probs = self._rec_engine.predict_batch(cropped_images, return_prob=True)
                    for box_points, text, conf in zip(valid_boxes, texts, probs):
                        clean_text = str(text).strip()
                        if clean_text:
                            flat_points = [int(coord) for pt in box_points for coord in pt]
                            parsed_boxes.append({
                                "text": clean_text,
                                "points": flat_points
                            })
                            confidences.append(float(conf))

        except Exception as e:
            logger.error(f"Lỗi khi chạy Hybrid OCR: {e}", exc_info=True)
            return "", 0.0, [], self._engine_name

        sorted_boxes = self._sort_reading_order(parsed_boxes)
        raw_text = "\n".join([b["text"] for b in sorted_boxes if b.get("text")])
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return raw_text, avg_confidence, sorted_boxes, self._engine_name
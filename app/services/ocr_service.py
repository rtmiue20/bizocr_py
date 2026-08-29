import logging
from rapidocr_onnxruntime import RapidOCR
from app.services.image_service import crop_and_flatten_document

logger = logging.getLogger(__name__)

try:
    engine = RapidOCR()
except Exception as e:
    logger.error(f"Không thể khởi tạo RapidOCR: {e}")
    engine = None


def extract_text(image_input):
    if engine is None:
        return {"rawText": "", "confidence": 0.0, "boxes": []}

    try:
        processed_image = crop_and_flatten_document(image_input)
    except Exception as e:
        logger.warning(f"Lỗi khi tiền xử lý ảnh: {e}. Sử dụng ảnh gốc.")
        processed_image = image_input

    try:
        result, elapsed_time = engine(processed_image)
    except Exception as e:
        logger.error(f"Lỗi khi chạy engine OCR: {e}")
        return {"rawText": "", "confidence": 0.0, "boxes": []}

    full_text = []
    confidence_sum = 0.0
    count = 0
    boxes = []

    if result:
        for line in result:
            # line structure: [dt_boxes, txt, score]
            text = line[1]
            score = float(line[2]) if line[2] is not None else 0.0

            # Phẳng hóa tọa độ 4 góc polygon thành danh sách int [x1, y1, x2, y2, x3, y3, x4, y4]
            raw_box = line[0]
            flat_points = []
            if raw_box:
                for pt in raw_box:
                    flat_points.extend([int(pt[0]), int(pt[1])])

            full_text.append(text)
            confidence_sum += score
            count += 1

            boxes.append({
                "text": text,
                "points": flat_points
            })

    raw_text = "\n".join(full_text)
    avg_confidence = round(confidence_sum / count, 2) if count > 0 else 0.0

    # Trả về các key khớp 100% với DTO OcrResult phía Java
    return {
        "rawText": raw_text,
        "confidence": avg_confidence,
        "boxes": boxes
    }
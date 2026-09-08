import logging
import cv2
import numpy as np
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger(__name__)

# ==============================================================================
# CẤU HÌNH NGƯỠNG (THRESHOLDS)
# ==============================================================================
CONFIG = {
    "MAX_IMAGE_SIDE": 1100,  # Kích thước tối đa khi resize để quét hướng
    "HORIZ_ASPECT_RATIO": 1.3,  # Tỷ lệ w/h để xác định một dòng chữ là nằm ngang
    "VIET_SCORE_FLIP_THRES": 40.0,  # Ngưỡng điểm tiếng Việt tối thiểu để kiểm tra lộn ngược
    "MIN_PAGE_AREA_RATIO": 0.5  # Tỷ lệ diện tích giấy tối thiểu so với ảnh (50%) để nắn 3D
}

try:
    engine = RapidOCR()
except Exception as e:
    logger.error(f"Không thể khởi tạo RapidOCR: {e}")
    engine = None

VIET_CHARS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")


# ==============================================================================
# THUẬT TOÁN XOAY VÀ NẮN ẢNH
# ==============================================================================

def _limit(img, max_side=CONFIG["MAX_IMAGE_SIDE"]):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _estimate_skew_angle(quads):
    """Ước lượng góc nghiêng lẻ (30°, 120°, ...) bằng trung vị góc các cạnh chữ dài nhất."""
    angles = []
    for q in quads:
        max_sq_len = 0
        best_dx, best_dy = 0, 0
        for i in range(4):
            dx = q[(i + 1) % 4][0] - q[i][0]
            dy = q[(i + 1) % 4][1] - q[i][1]
            sq_len = dx ** 2 + dy ** 2
            if sq_len > max_sq_len:
                max_sq_len = sq_len
                best_dx, best_dy = dx, dy

        if max_sq_len > 64:  # Cạnh > 8px
            if best_dx < 0:
                best_dx, best_dy = -best_dx, -best_dy
            if best_dx != 0 or best_dy != 0:
                angles.append(np.arctan2(best_dy, best_dx))

    if len(angles) < 5:
        return 0.0
    angles.sort()
    mid = len(angles) // 2
    return float(angles[mid] if len(angles) % 2 else (angles[mid - 1] + angles[mid]) / 2)


def _rotate_by_angle(img, angle_rad):
    """Xoay bù góc nghiêng và tự động mở rộng canvas để không xén mất 4 góc nội dung."""
    h, w = img.shape[:2]
    angle_deg = np.degrees(angle_rad)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(img, M, (new_w, new_h), borderValue=(255, 255, 255))


def _order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _flatten_page(img):
    """Nắn phẳng phối cảnh 3D khi chụp nghiêng/tạo hình thang."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    frame_area = h * w
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    corners = None
    for min_ratio in (0.20, 0.10, 0.05):
        for c in contours:
            if cv2.contourArea(c) < frame_area * min_ratio:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(np.float32)
                break
        if corners is not None:
            break

    if corners is None and cv2.contourArea(contours[0]) >= frame_area * 0.15:
        corners = cv2.boxPoints(cv2.minAreaRect(contours[0])).astype(np.float32)

    if corners is None:
        return img

    rect = _order_corners(corners)

    # Kiểm tra chống cắt lẹm nội dung
    area_of_polygon = cv2.contourArea(rect)
    if area_of_polygon < frame_area * CONFIG["MIN_PAGE_AREA_RATIO"]:
        logger.info("Bỏ qua nắn 3D: Diện tích vùng cắt quá nhỏ so với ảnh gốc.")
        return img

    tl, tr, br, bl = rect
    tw = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    th = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if tw < 200 or th < 200:
        return img

    dst = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (tw, th), borderMode=cv2.BORDER_REPLICATE)


def _get_quads(ocr_engine, img):
    result, _ = ocr_engine(img, use_det=True, use_cls=False, use_rec=False)
    quads = []
    for item in result or []:
        arr = np.asarray(item, dtype=np.float32)
        if arr.shape != (4, 2):
            arr = np.asarray(item[0], dtype=np.float32)
        if arr.shape == (4, 2):
            quads.append(arr)
    return quads


def _orient_score(ocr_engine, img):
    """Tính điểm hướng chữ dựa trên số lượng khung chữ nằm ngang."""
    total = 0.0
    for q in _get_quads(ocr_engine, img):
        w = q[:, 0].max() - q[:, 0].min()
        h = q[:, 1].max() - q[:, 1].min()
        if w > CONFIG["HORIZ_ASPECT_RATIO"] * h:
            total += float(w)
    return total


def _viet_score(results):
    """Tính điểm tin cậy tiếng Việt để phát hiện ảnh bị ngược 180°."""
    score = 0.0
    for line in results or []:
        text, conf = str(line[1]), float(line[2] or 0)
        if conf >= 0.65 and len(text) >= 6 and any(c in VIET_CHARS for c in text.lower()):
            score += conf * len(text)
    return score


def load_and_orient_upright(ocr_engine, image_input):
    """Xử lý toàn bộ các góc xoay khó: 30°, 120°, 270°, lộn ngược 180°."""
    if isinstance(image_input, str):
        pil = ImageOps.exif_transpose(Image.open(image_input)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        img = image_input.copy()

    # 1. Đo góc nghiêng lẻ
    quads = _get_quads(ocr_engine, img)
    angle = _estimate_skew_angle(quads)
    if abs(angle) >= 0.008:
        img = _rotate_by_angle(img, angle)

    # 2. Nắn phẳng phối cảnh 3D
    img = _flatten_page(img)

    # 3. Quét góc 90° / 270° (Trái / Phải)
    small = _limit(img, CONFIG["MAX_IMAGE_SIDE"])
    base_score = _orient_score(ocr_engine, small)
    best_code, best_score = None, base_score

    for code in (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE):
        score = _orient_score(ocr_engine, cv2.rotate(small, code))
        if score > best_score:
            best_code, best_score = code, score

    if best_code is not None and best_score > base_score * 1.15:
        img = cv2.rotate(img, best_code)

    return img


# ==============================================================================
# HÀM EXPORT TRÍCH XUẤT CHÍNH
# ==============================================================================

def extract_text(image_input):
    if engine is None:
        return {"rawText": "", "confidence": 0.0, "boxes": []}

    try:
        # Tự động xoay thẳng đúng chiều trước khi OCR
        processed_image = load_and_orient_upright(engine, image_input)
    except Exception as e:
        logger.warning(f"Lỗi khi xoay/nắn ảnh: {e}. Dùng ảnh gốc.")
        processed_image = image_input

    try:
        result, _ = engine(processed_image)

        # 4. Kiểm tra lộn ngược 180° bằng điểm Tiếng Việt
        if _viet_score(result) < CONFIG["VIET_SCORE_FLIP_THRES"]:
            flipped_img = cv2.rotate(processed_image, cv2.ROTATE_180)
            flipped_result, _ = engine(flipped_img)
            if _viet_score(flipped_result) > _viet_score(result) * 1.3 + 10:
                result = flipped_result
                processed_image = flipped_img

    except Exception as e:
        logger.error(f"Lỗi khi chạy engine OCR: {e}")
        return {"rawText": "", "confidence": 0.0, "boxes": []}

    full_text = []
    confidence_sum = 0.0
    count = 0
    boxes = []

    if result:
        for line in result:
            text = line[1]
            score = float(line[2]) if line[2] is not None else 0.0
            raw_box = line[0]
            flat_points = []
            if raw_box:
                for pt in raw_box:
                    flat_points.extend([int(pt[0]), int(pt[1])])

            full_text.append(text)
            confidence_sum += score
            count += 1
            boxes.append({"text": text, "points": flat_points})

    return {
        "rawText": "\n".join(full_text),
        "confidence": round(confidence_sum / count, 2) if count > 0 else 0.0,
        "boxes": boxes
    }
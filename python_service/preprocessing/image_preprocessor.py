import cv2
import numpy as np
from typing import Tuple, Dict, Any

class ImagePreprocessor:

    @staticmethod
    def process(image_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Không thể decode dữ liệu ảnh từ request")

        orig_h, orig_w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return ImagePreprocessor._fallback_response(image_bytes, orig_w, orig_h)

        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)

        if contour_area < (orig_w * orig_h * 0.08):
            return ImagePreprocessor._fallback_response(image_bytes, orig_w, orig_h)

        rect = cv2.minAreaRect(largest_contour)
        (cx, cy), (w, h), angle = rect

        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        else:
            angle = angle

        processed_img = img.copy()
        if abs(angle) > 0.8:
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            processed_img = cv2.warpAffine(
                processed_img, M, (orig_w, orig_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            # Transform contour to match rotated image
            largest_contour = cv2.transform(largest_contour, M)

        x, y, crop_w, crop_h = cv2.boundingRect(largest_contour)
        
        # Clamp coordinates to image boundaries before padding
        x_min = max(0, x)
        y_min = max(0, y)
        x_max = min(orig_w, x + crop_w)
        y_max = min(orig_h, y + crop_h)
        
        # Apply padding safely
        x_start = max(0, x_min - 5)
        y_start = max(0, y_min - 5)
        x_end = min(orig_w, x_max + 5)
        y_end = min(orig_h, y_max + 5)

        cropped_img = processed_img[y_start:y_end, x_start:x_end]
        final_h, final_w = cropped_img.shape[:2]
        aspect_ratio = round(final_w / float(final_h), 4)

        _, buffer = cv2.imencode('.jpg', cropped_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        metadata = {
            "width": final_w,
            "height": final_h,
            "aspect_ratio": aspect_ratio,
            "deskew_angle": round(angle, 2),
            "is_cropped": True
        }

        return buffer.tobytes(), metadata

    @staticmethod
    def _fallback_response(image_bytes: bytes, width: int, height: int) -> Tuple[bytes, Dict[str, Any]]:
        return image_bytes, {
            "width": width,
            "height": height,
            "aspect_ratio": round(width / float(height), 4),
            "deskew_angle": 0.0,
            "is_cropped": False
        }
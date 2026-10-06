import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from python_service.preprocessing.qr_reader import QrReader


class ImagePreprocessor:
    MIN_CONTOUR_AREA_RATIO_FOR_CROP = 0.08

    @staticmethod
    def process(image_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Không thể decode dữ liệu ảnh từ request")

        orig_h, orig_w = img.shape[:2]
        orig_area = float(orig_w * orig_h)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_area = 0.0
        largest_contour = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(largest_contour))

        contour_area_ratio = round(contour_area / orig_area, 4) if orig_area else 0.0

        if largest_contour is None or contour_area_ratio < ImagePreprocessor.MIN_CONTOUR_AREA_RATIO_FOR_CROP:
            return ImagePreprocessor._fallback_response(
                image_bytes, img, orig_w, orig_h, contour_area_ratio
            )

        rect = cv2.minAreaRect(largest_contour)
        (cx, cy), (_w, _h), angle = rect

        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        processed_img = img.copy()
        if abs(angle) > 0.8:
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            processed_img = cv2.warpAffine(
                processed_img, M, (orig_w, orig_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            largest_contour = cv2.transform(largest_contour, M)

        x, y, crop_w, crop_h = cv2.boundingRect(largest_contour)

        x_min = max(0, x)
        y_min = max(0, y)
        x_max = min(orig_w, x + crop_w)
        y_max = min(orig_h, y + crop_h)

        x_start = max(0, x_min - 5)
        y_start = max(0, y_min - 5)
        x_end = min(orig_w, x_max + 5)
        y_end = min(orig_h, y_max + 5)

        cropped_img = processed_img[y_start:y_end, x_start:x_end]
        if cropped_img.size == 0:
            return ImagePreprocessor._fallback_response(
                image_bytes, img, orig_w, orig_h, contour_area_ratio
            )

        final_h, final_w = cropped_img.shape[:2]
        aspect_after = round(final_w / float(final_h), 4)
        _, buffer = cv2.imencode(".jpg", cropped_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        qr_res = QrReader.read_qr(cropped_img)
        if not qr_res.has_qr:
            qr_res = QrReader.read_qr(img)

        metadata = ImagePreprocessor._metadata(
            width=final_w,
            height=final_h,
            original_width=orig_w,
            original_height=orig_h,
            aspect_ratio_after_crop=aspect_after,
            deskew_angle=round(float(angle), 2),
            is_cropped=True,
            crop_failed=False,
            contour_area_ratio=contour_area_ratio,
            image_for_sharpness=cropped_img,
            qr_result=qr_res.to_dict() if qr_res.has_qr else None,
        )
        return buffer.tobytes(), metadata

    @staticmethod
    def _fallback_response(
        image_bytes: bytes,
        img: np.ndarray,
        width: int,
        height: int,
        contour_area_ratio: float,
    ) -> Tuple[bytes, Dict[str, Any]]:
        aspect = round(width / float(height), 4) if height else 0.0
        qr_res = QrReader.read_qr(img)
        metadata = ImagePreprocessor._metadata(
            width=width,
            height=height,
            original_width=width,
            original_height=height,
            aspect_ratio_after_crop=aspect,
            deskew_angle=0.0,
            is_cropped=False,
            crop_failed=True,
            contour_area_ratio=contour_area_ratio,
            image_for_sharpness=img,
            qr_result=qr_res.to_dict() if qr_res.has_qr else None,
        )
        return image_bytes, metadata

    @staticmethod
    def _metadata(
            *,
            width: int,
            height: int,
            original_width: int,
            original_height: int,
            aspect_ratio_after_crop: float,
            deskew_angle: float,
            is_cropped: bool,
            crop_failed: bool,
            contour_area_ratio: float,
            image_for_sharpness: Optional[np.ndarray],
            qr_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "width": width,
            "height": height,
            "original_width": original_width,
            "original_height": original_height,
            "aspect_ratio": aspect_ratio_after_crop,
            "aspect_ratio_after_crop": aspect_ratio_after_crop,
            "deskew_angle": deskew_angle,
            "is_cropped": is_cropped,
            "crop_failed": crop_failed,
            "contour_area_ratio": contour_area_ratio,
            "laplacian_variance": ImagePreprocessor._laplacian_variance(image_for_sharpness),
            "mean_brightness": ImagePreprocessor._mean_brightness(image_for_sharpness),
            "qr_result": qr_result,
        }

    @staticmethod
    def _laplacian_variance(image: Optional[np.ndarray]) -> float:
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4)

    @staticmethod
    def _mean_brightness(image: Optional[np.ndarray]) -> float:
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return round(float(cv2.mean(gray)[0]), 4)

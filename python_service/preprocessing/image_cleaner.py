import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class ImageCleaner:
    """
    OpenCV Pipeline: Ảnh gốc -> Ảnh sạch (Deskew, Denoise, CLAHE, Shadow Removal, Sharpen)
    """

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không thể giải mã dữ liệu ảnh (Invalid image format)")
        return img

    @staticmethod
    def encode_image(image: np.ndarray, format: str = ".jpg") -> bytes:
        success, encoded = cv2.imencode(format, image)
        if not success:
            raise ValueError("Không thể mã hóa ảnh sang byte stream")
        return encoded.tobytes()

    @staticmethod
    def resize_if_needed(image: np.ndarray, max_dim: int = 2500) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            logger.info(f"Resize ảnh từ ({w}x{h}) sang ({new_w}x{new_h})")
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    @staticmethod
    def compute_skew_angle(gray: np.ndarray) -> float:
        """
        Tính góc nghiêng của văn bản bằng minAreaRect trên text pixels
        """
        try:
            # Nhị phân hóa nghịch đảo (chữ trắng nền đen)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Tìm tọa độ các pixel văn bản
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) < 100:
                return 0.0

            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            else:
                angle = -angle

            # Nếu góc quá lớn (ảnh dọc/ngang bình thường), có thể là do nhầm hướng
            if abs(angle) > 45:
                return 0.0
            return angle
        except Exception as e:
            logger.warning(f"Lỗi khi tính góc nghiêng: {e}")
            return 0.0

    @classmethod
    def deskew(cls, image: np.ndarray) -> np.ndarray:
        """
        Tự động xoay thẳng văn bản
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        angle = cls.compute_skew_angle(gray)

        if abs(angle) < 0.5:
            return image

        logger.info(f"Phát hiện ảnh nghiêng {angle:.2f} độ. Đang tự động xoay thẳng...")
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Tính kích thước mới để không bị cắt xén góc
        cos = np.abs(rot_matrix[0, 0])
        sin = np.abs(rot_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        rot_matrix[0, 2] += (new_w / 2) - center[0]
        rot_matrix[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            image, rot_matrix, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        return rotated

    @staticmethod
    def remove_shadow_and_enhance(image: np.ndarray) -> np.ndarray:
        """
        Khử bóng đổ, cân bằng sáng và tăng tương phản ký tự bằng CLAHE
        """
        if len(image.shape) == 3:
            # Chuyển sang không gian màu LAB để xử lý kênh sáng L
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Khử bóng đổ: Morphological closing ước lượng nền
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            background = cv2.morphologyEx(l_channel, cv2.MORPH_CLOSE, kernel)
            
            # Trừ nền để làm đều độ sáng
            normalized_l = cv2.divide(l_channel, background, scale=255)
            
            # Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(normalized_l)
            
            # Gộp lại và chuyển về BGR
            enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            return enhanced_bgr
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    @staticmethod
    def denoise_and_sharpen(image: np.ndarray) -> np.ndarray:
        """
        Khử nhiễu nhẹ và làm nét viền ký tự (Sharpening)
        """
        # Bilateral filter giữ cạnh sắc nét trong khi làm mịn vùng nhiễu
        denoised = cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)

        # Kernel làm nét
        kernel_sharpen = np.array([
            [0, -0.5, 0],
            [-0.5, 3.0, -0.5],
            [0, -0.5, 0]
        ], dtype=np.float32)

        sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
        return sharpened

    @classmethod
    def clean(cls, image_bytes: bytes, max_dim: int = 2500) -> Tuple[np.ndarray, bytes]:
        """
        Quy trình xử lý hoàn chỉnh: Ảnh gốc -> Ảnh sạch
        """
        raw_img = cls.decode_image(image_bytes)
        img_resized = cls.resize_if_needed(raw_img, max_dim=max_dim)
        
        # 1. Deskew (Xoay thẳng)
        deskewed = cls.deskew(img_resized)
        
        # 2. Khử bóng & Tăng tương phản
        enhanced = cls.remove_shadow_and_enhance(deskewed)
        
        # 3. Khử nhiễu & Làm nét ký tự
        cleaned = cls.denoise_and_sharpen(enhanced)
        
        cleaned_bytes = cls.encode_image(cleaned, format=".jpg")
        return cleaned, cleaned_bytes

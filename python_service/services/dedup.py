import hashlib
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def slugify(text: Optional[str]) -> str:
    """Chuẩn hóa chuỗi văn bản thành slug không dấu dùng làm khóa định danh."""
    if not text:
        return ""
    s = text.lower().strip()
    s = re.sub(r"[àáạảãâầấậẩẫăằắặẳẵ]", "a", s)
    s = re.sub(r"[èéẹẻẽêềếệểễ]", "e", s)
    s = re.sub(r"[ìíịỉĩ]", "i", s)
    s = re.sub(r"[òóọỏõôồốộổỗơờớợởỡ]", "o", s)
    s = re.sub(r"[ùúụủũưừứựửữ]", "u", s)
    s = re.sub(r"[ỳýỵỷỹ]", "y", s)
    s = re.sub(r"[đ]", "d", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", "_", s)


class ImageHasher:
    """
    Tính mã băm nhận thức hình ảnh (Perceptual Hash - dHash/pHash) và SHA-256.
    Phát hiện việc cùng một danh thiếp được chụp lại hoặc tải lên nhiều lần.
    """

    @staticmethod
    def sha256(image_bytes: bytes) -> str:
        """Mã băm SHA-256 đối soát trùng lặp tệp tuyệt đối."""
        return hashlib.sha256(image_bytes).hexdigest()

    @staticmethod
    def dhash(image: Union[np.ndarray, bytes], hash_size: int = 8) -> str:
        """
        Tính difference hash (dHash 64-bit):
        Thu nhỏ ảnh về kích thước (hash_size + 1, hash_size), so sánh pixel lân cận theo hàng.
        """
        try:
            if isinstance(image, bytes):
                nparr = np.frombuffer(image, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            elif image.ndim == 3:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                img = image

            if img is None or img.size == 0:
                return "0" * 16

            # Resize ảnh về (hash_size + 1, hash_size)
            resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)

            # So sánh pixel cột liền kề
            diff = resized[:, 1:] > resized[:, :-1]
            # Chuyển boolean array thành chuỗi hex
            decimal_value = 0
            for index, value in enumerate(diff.flatten()):
                if value:
                    decimal_value += 1 << index

            return f"{decimal_value:016x}"
        except Exception as e:
            logger.warning(f"Lỗi tính dHash: {e}")
            return "0" * 16

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Tính khoảng cách Hamming giữa hai mã băm 64-bit (số bit khác biệt)."""
        try:
            val1 = int(hash1, 16)
            val2 = int(hash2, 16)
            return bin(val1 ^ val2).count("1")
        except Exception:
            return 64


class CardDedupService:
    """
    Dịch vụ phát hiện thẻ trùng lặp (Dedup Service):
    - Tầng 1: pHash / dHash ảnh (Hamming distance <= 5).
    - Tầng 2: Khóa định danh thực thể (Số điện thoại, Email, Cặp Công ty + Họ tên).
    """

    def __init__(self):
        self._lock = threading.Lock()
        # image_hashes: {card_id: dhash_hex}
        self.image_hashes: Dict[str, str] = {}
        # phone_index: {normalized_phone: card_id}
        self.phone_index: Dict[str, str] = {}
        # email_index: {normalized_email: card_id}
        self.email_index: Dict[str, str] = {}
        # entity_index: {slug_company_person: card_id}
        self.entity_index: Dict[str, str] = {}

    def register_card(
        self,
        card_id: str,
        image_hash: Optional[str] = None,
        phones: Optional[List[str]] = None,
        emails: Optional[List[str]] = None,
        company: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """Đăng ký thông tin thẻ vào chỉ mục dedup."""
        with self._lock:
            if image_hash:
                self.image_hashes[card_id] = image_hash

            for p in phones or []:
                clean_p = re.sub(r"[^\d+]", "", p)
                if clean_p:
                    self.phone_index[clean_p] = card_id

            for em in emails or []:
                clean_em = em.lower().strip()
                if clean_em:
                    self.email_index[clean_em] = card_id

            if company and name:
                key = f"{slugify(company)}::{slugify(name)}"
                if key != "::":
                    self.entity_index[key] = card_id

    def check_duplicate(
        self,
        image_hash: Optional[str] = None,
        phones: Optional[List[str]] = None,
        emails: Optional[List[str]] = None,
        company: Optional[str] = None,
        name: Optional[str] = None,
        max_hash_dist: int = 5,
    ) -> Dict[str, Any]:
        """
        Kiểm tra xem danh thiếp đã tồn tại hay chưa.
        """
        with self._lock:
            # 1. Kiểm tra qua Số điện thoại
            for p in phones or []:
                clean_p = re.sub(r"[^\d+]", "", p)
                if clean_p and clean_p in self.phone_index:
                    matched_id = self.phone_index[clean_p]
                    return {
                        "isDuplicate": True,
                        "reason": f"Trùng số điện thoại ({clean_p}) với thẻ đã lưu",
                        "matchedId": matched_id,
                        "identityKey": f"phone:{clean_p}",
                        "imageHash": image_hash,
                    }

            # 2. Kiểm tra qua Email
            for em in emails or []:
                clean_em = em.lower().strip()
                if clean_em and clean_em in self.email_index:
                    matched_id = self.email_index[clean_em]
                    return {
                        "isDuplicate": True,
                        "reason": f"Trùng địa chỉ email ({clean_em}) với thẻ đã lưu",
                        "matchedId": matched_id,
                        "identityKey": f"email:{clean_em}",
                        "imageHash": image_hash,
                    }

            # 3. Kiểm tra qua cặp Tên công ty + Họ tên người đại diện
            if company and name:
                key = f"{slugify(company)}::{slugify(name)}"
                if key in self.entity_index:
                    matched_id = self.entity_index[key]
                    return {
                        "isDuplicate": True,
                        "reason": f"Trùng tên công ty và người đại diện ({company} - {name})",
                        "matchedId": matched_id,
                        "identityKey": key,
                        "imageHash": image_hash,
                    }

            # 4. Kiểm tra qua Perceptual Image Hash
            if image_hash:
                for existing_id, existing_hash in self.image_hashes.items():
                    dist = ImageHasher.hamming_distance(image_hash, existing_hash)
                    if dist <= max_hash_dist:
                        return {
                            "isDuplicate": True,
                            "reason": f"Hình ảnh tương đồng cao (Khoảng cách Hamming = {dist})",
                            "matchedId": existing_id,
                            "identityKey": f"hash:{image_hash}",
                            "imageHash": image_hash,
                        }

        return {
            "isDuplicate": False,
            "reason": None,
            "matchedId": None,
            "identityKey": None,
            "imageHash": image_hash,
        }


dedup_service = CardDedupService()

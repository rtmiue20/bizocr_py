import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class QrCodeResult:
    def __init__(
        self,
        has_qr: bool = False,
        raw_text: Optional[str] = None,
        qr_type: Optional[str] = None,
        vcard_data: Optional[Dict[str, Any]] = None,
        points: Optional[List[List[float]]] = None,
    ):
        self.has_qr = has_qr
        self.raw_text = raw_text
        self.qr_type = qr_type
        self.vcard_data = vcard_data or {}
        self.points = points or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hasQr": self.has_qr,
            "qrText": self.raw_text,
            "qrType": self.qr_type,
            "vcardData": self.vcard_data,
        }


class QrReader:
    """
    Module quét và giải mã mã QR Code / vCard trên danh thiếp.
    Hỗ trợ phát hiện qua OpenCV QRCodeDetector và phân tích cú pháp vCard 2.1/3.0/4.0.
    """

    _detector = None

    @classmethod
    def _get_detector(cls) -> cv2.QRCodeDetector:
        if cls._detector is None:
            cls._detector = cv2.QRCodeDetector()
        return cls._detector

    @classmethod
    def parse_vcard(cls, vcard_text: str) -> Dict[str, Any]:
        """
        Phân tích cú pháp vCard (hỗ trợ vCard 2.1, 3.0, 4.0).
        Trích xuất Họ tên, Công ty, Chức vụ, Số điện thoại, Email, Website, Địa chỉ.
        """
        data: Dict[str, Any] = {
            "name": None,
            "company": None,
            "title": None,
            "phones": [],
            "emails": [],
            "websites": [],
            "address": None,
            "note": None,
        }

        # Gộp các dòng vCard bị ngắt dòng (folding lines bắt đầu bằng space hoặc tab)
        unfolded_lines: List[str] = []
        for line in vcard_text.splitlines():
            line_str = line.strip("\r\n")
            if not line_str:
                continue
            if line_str.startswith(" ") or line_str.startswith("\t"):
                if unfolded_lines:
                    unfolded_lines[-1] += line_str[1:]
            else:
                unfolded_lines.append(line_str)

        for line in unfolded_lines:
            if not line or ":" not in line:
                continue

            header, value = line.split(":", 1)
            header_clean = header.upper().strip()
            # Bỏ param sau dấu chấm phẩy, ví dụ: TEL;TYPE=WORK -> TEL
            prop_name = header_clean.split(";")[0].strip()
            val_clean = value.strip()

            if prop_name in ("FN", "NAME") and not data["name"]:
                # Họ và tên đầy đủ
                data["name"] = val_clean
            elif prop_name == "N" and not data["name"]:
                # N:Họ;Tên;Đệm;Tiền tố;Hậu tố
                parts = [p.strip() for p in val_clean.split(";") if p.strip()]
                if parts:
                    # Đảo lại theo thứ tự Họ Tên tự nhiên nếu có nhiều phần tử
                    data["name"] = " ".join(reversed(parts))
            elif prop_name in ("ORG", "COMPANY"):
                # Tên tổ chức / công ty (có thể có phòng ban sau dấu ;)
                org_parts = [p.strip() for p in val_clean.split(";") if p.strip()]
                if org_parts and not data["company"]:
                    data["company"] = org_parts[0]
            elif prop_name in ("TITLE", "ROLE") and not data["title"]:
                data["title"] = val_clean
            elif prop_name in ("TEL", "PHONE", "CELL"):
                # Làm sạch số điện thoại
                clean_phone = re.sub(r"[^\d+]", "", val_clean)
                if clean_phone and clean_phone not in data["phones"]:
                    data["phones"].append(clean_phone)
            elif prop_name in ("EMAIL", "MAIL"):
                if val_clean and val_clean not in data["emails"]:
                    data["emails"].append(val_clean)
            elif prop_name in ("URL", "WEBSITE"):
                if val_clean and val_clean not in data["websites"]:
                    data["websites"].append(val_clean)
            elif prop_name in ("ADR", "ADDRESS") and not data["address"]:
                # ADR:;;Phố;Thành phố;Tỉnh;Mã;Quốc gia
                adr_parts = [p.strip() for p in val_clean.split(";") if p.strip()]
                if adr_parts:
                    data["address"] = ", ".join(adr_parts)
            elif prop_name == "NOTE" and not data["note"]:
                data["note"] = val_clean

        return data

    @classmethod
    def read_qr(cls, image: Union[np.ndarray, bytes]) -> QrCodeResult:
        """
        Nhận diện và giải mã mã QR từ ảnh numpy hoặc bytes.
        """
        try:
            if isinstance(image, bytes):
                nparr = np.frombuffer(image, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = image

            if img is None or img.size == 0:
                return QrCodeResult(has_qr=False)

            detector = cls._get_detector()

            # 1. Thử giải mã trên ảnh gốc
            decoded_text, points, _ = detector.detectAndDecode(img)

            # 2. Nếu chưa tìm thấy, thử trên ảnh xám hoặc tăng cường tương phản
            if not decoded_text:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                decoded_text, points, _ = detector.detectAndDecode(gray)

            # 3. Thử ngưỡng nhị phân Otsu nếu vẫn chưa thấy
            if not decoded_text:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                decoded_text, points, _ = detector.detectAndDecode(thresh)

            if not decoded_text or not decoded_text.strip():
                return QrCodeResult(has_qr=False)

            decoded_text = decoded_text.strip()
            pts_list = points.tolist() if points is not None else []

            # Phân loại định dạng nội dung
            upper_text = decoded_text.upper()
            if "BEGIN:VCARD" in upper_text:
                vcard = cls.parse_vcard(decoded_text)
                logger.info(f"Đã phát hiện mã vCard QR: {vcard.get('name')} | {vcard.get('company')}")
                return QrCodeResult(
                    has_qr=True,
                    raw_text=decoded_text,
                    qr_type="VCARD",
                    vcard_data=vcard,
                    points=pts_list,
                )
            elif decoded_text.startswith("http://") or decoded_text.startswith("https://"):
                logger.info(f"Đã phát hiện mã URL QR: {decoded_text}")
                return QrCodeResult(
                    has_qr=True,
                    raw_text=decoded_text,
                    qr_type="URL",
                    vcard_data={"websites": [decoded_text]},
                    points=pts_list,
                )
            else:
                logger.info(f"Đã phát hiện mã QR Text: {decoded_text[:50]}...")
                return QrCodeResult(
                    has_qr=True,
                    raw_text=decoded_text,
                    qr_type="TEXT",
                    vcard_data={},
                    points=pts_list,
                )

        except Exception as e:
            logger.warning(f"Lỗi khi đọc mã QR: {e}")
            return QrCodeResult(has_qr=False)

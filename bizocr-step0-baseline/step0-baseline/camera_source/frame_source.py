"""
Nguồn khung hình dùng chung cho camera agent (Bước 4) và test.

Camera agent chỉ phụ thuộc interface FrameSource, nên chạy được với:
  - FolderFrameSource / VideoFileFrameSource : test, CI, demo không cần phần cứng
  - CameraFrameSource                        : camera thật qua cổng C (điền sau khi có header/SDK)

Quy ước khung hình: numpy uint8, shape (H, W, 3), thứ tự kênh BGR (giống OpenCV).
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


class FrameSource(ABC):
    def open(self) -> None:  # noqa: D401
        """Mở nguồn. Mặc định không làm gì."""

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Trả về khung hình kế tiếp (BGR) hoặc None khi hết / lỗi."""

    def close(self) -> None:
        """Giải phóng tài nguyên. Mặc định không làm gì."""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


def _letterbox(img: np.ndarray, size: tuple[int, int], bg: tuple[int, int, int]) -> np.ndarray:
    """Đặt ảnh vào giữa khung (W, H), giữ tỉ lệ, phần còn lại là nền."""
    W, H = size
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((H, W, 3), bg, np.uint8)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


class FolderFrameSource(FrameSource):
    """
    Giả lập camera cố định: với mỗi ảnh trong thư mục sẽ phát
        empty_frames khung bàn trống -> hold_frames khung có tài liệu (thêm nhiễu nhẹ) -> ...
    và cuối cùng thêm empty_frames khung trống rồi kết thúc.
    Dùng để test máy trạng thái auto-capture: đúng 1 lần chụp cho mỗi tài liệu.
    """

    def __init__(self, folder, empty_frames=15, hold_frames=30, frame_size=(1280, 720),
                 noise_sigma=2.0, background_bgr=(70, 70, 70), seed=0,
                 extensions=(".jpg", ".jpeg", ".png")):
        self.paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in extensions)
        self.empty_frames, self.hold_frames = empty_frames, hold_frames
        self.frame_size, self.noise_sigma, self.bg = frame_size, noise_sigma, background_bgr
        self._rng = np.random.default_rng(seed)
        self._it: Optional[Iterator[np.ndarray]] = None

    def open(self) -> None:
        self._it = self._frames()

    def _noisy(self, frame: np.ndarray) -> np.ndarray:
        if self.noise_sigma <= 0:
            return frame.copy()
        noise = self._rng.normal(0, self.noise_sigma, frame.shape)
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _frames(self) -> Iterator[np.ndarray]:
        W, H = self.frame_size
        empty = np.full((H, W, 3), self.bg, np.uint8)
        for path in self.paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            doc = _letterbox(img, self.frame_size, self.bg)
            for _ in range(self.empty_frames):
                yield self._noisy(empty)
            for _ in range(self.hold_frames):
                yield self._noisy(doc)
        for _ in range(self.empty_frames):
            yield self._noisy(empty)

    def read(self) -> Optional[np.ndarray]:
        if self._it is None:
            self.open()
        return next(self._it, None)


class VideoFileFrameSource(FrameSource):
    def __init__(self, path):
        self.path, self._cap = str(path), None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise IOError(f"Không mở được video: {self.path}")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


class CameraFrameSource(FrameSource):
    """
    Kết nối với Webcam USB (như Ugreen 4K) thông qua OpenCV.
    Không cần C-binding/SDK vì thiết bị dùng chuẩn UVC (USB Video Class).
    """

    def __init__(self, camera_index=0, width=3840, height=2160):
        self.camera_index = camera_index
        # Mặc định ép lấy độ phân giải 4K (8MP) của Ugreen để OCR rõ nét nhất
        self.width = width
        self.height = height
        self._cap = None

    def open(self) -> None:
        # Mở kết nối camera (Trên Windows có thể thêm cv2.CAP_DSHOW nếu khởi động chậm)
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise IOError(f"Không thể kết nối với webcam ở index {self.camera_index}")

        # Ép độ phân giải phần cứng
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Đọc bỏ 5-10 khung hình đầu tiên để webcam lấy nét (Auto-Focus)
        # và cân bằng sáng (Auto-Exposure) ổn định.
        for _ in range(10):
            self._cap.read()

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            self.open()

        ok, frame = self._cap.read()
        if not ok:
            return None

        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

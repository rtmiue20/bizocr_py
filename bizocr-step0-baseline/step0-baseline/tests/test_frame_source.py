import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "camera_source"))
from frame_source import FolderFrameSource  # noqa: E402


def test_folder_source_yields_empty_hold_empty_pattern(tmp_path):
    for i in range(2):
        img = np.full((300, 400, 3), 240, np.uint8)
        cv2.rectangle(img, (20, 20), (380, 280), (0, 0, 0), 3)
        cv2.imwrite(str(tmp_path / f"doc{i}.png"), img)

    src = FolderFrameSource(tmp_path, empty_frames=3, hold_frames=5, frame_size=(640, 360), noise_sigma=0)
    with src:
        frames = []
        while (f := src.read()) is not None:
            frames.append(f)

    # 2 tài liệu x (3 trống + 5 giữ) + 3 trống cuối
    assert len(frames) == 2 * (3 + 5) + 3
    assert frames[0].shape == (360, 640, 3)
    mean = [f.mean() for f in frames]
    assert mean[0] < mean[3]          # khung trống tối hơn khung có giấy trắng
    assert mean[-1] == mean[0]        # kết thúc bằng bàn trống

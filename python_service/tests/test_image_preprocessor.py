import cv2
import numpy as np

from python_service.preprocessing.image_preprocessor import ImagePreprocessor


def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_document_on_desk_is_cropped_and_reports_post_crop_aspect():
    frame = np.full((720, 1280, 3), 40, np.uint8)
    # Giấy A4 ngang trên bàn: 900x640 → aspect sau crop ~ 1.4
    cv2.rectangle(frame, (190, 40), (1090, 680), (245, 245, 245), -1)
    cv2.rectangle(frame, (190, 40), (1090, 680), (10, 10, 10), 4)

    _, meta = ImagePreprocessor.process(_encode(frame))

    assert meta["is_cropped"] is True
    assert meta["contour_area_ratio"] > 0.4
    assert meta["aspect_ratio_after_crop"] == meta["aspect_ratio"]
    assert 1.2 < meta["aspect_ratio_after_crop"] < 1.6
    assert meta["original_width"] == 1280
    assert meta["original_height"] == 720
    assert meta["width"] < 1280
    assert meta["height"] < 720
    assert meta["laplacian_variance"] > 0


def test_tiny_blob_does_not_crop_and_reports_low_contour_ratio():
    frame = np.full((400, 400, 3), 30, np.uint8)
    cv2.rectangle(frame, (190, 190), (210, 210), (250, 250, 250), -1)

    _, meta = ImagePreprocessor.process(_encode(frame))

    assert meta["is_cropped"] is False
    assert meta["contour_area_ratio"] < 0.08
    assert meta["aspect_ratio_after_crop"] == meta["aspect_ratio"]
    assert meta["original_width"] == 400
    assert meta["original_height"] == 400

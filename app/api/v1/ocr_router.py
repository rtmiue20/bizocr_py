from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ocr_service import extract_text
from app.models.ocr_dto import OcrResponse
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ocr", response_model=OcrResponse)
async def process_ocr(file: UploadFile = File(...)):
    # Hỗ trợ cả định dạng ảnh lẫn PDF ứng với ứng dụng kế toán
    valid_types = ["image/", "application/pdf"]
    if not any(file.content_type.startswith(t) for t in valid_types):
        raise HTTPException(status_code=400, detail="Tệp tải lên phải là hình ảnh hoặc tài liệu PDF")

    try:
        contents = await file.read()

        # 1. Xử lý trường hợp là file PDF
        if file.content_type.startswith("application/pdf"):
            # TODO: Cần sử dụng thư viện như pdf2image để chuyển byte PDF thành mảng ảnh OpenCV
            # Tạm thời trả về lỗi 501 nếu chưa cài đặt pdf2image
            raise HTTPException(status_code=501, detail="Chức năng trích xuất PDF đang được cập nhật.")

        # 2. Xử lý trường hợp là hình ảnh
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Không thể giải mã dữ liệu hình ảnh tải lên.")

        # 3. Gọi engine xử lý (Xoay, Nắn 3D và OCR)
        result = extract_text(img)

        # Kết quả trả về tự động được Pydantic parse theo OcrResponse model
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi xử lý OCR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống khi xử lý OCR: {str(e)}")
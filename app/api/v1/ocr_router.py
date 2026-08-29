from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ocr_service import extract_text
from app.models.ocr_dto import OcrResponse
import logging

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
        result = extract_text(contents)
        return result
    except Exception as e:
        logger.error(f"Lỗi khi xử lý OCR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống khi xử lý OCR: {str(e)}")
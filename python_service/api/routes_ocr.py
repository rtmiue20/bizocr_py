import io
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import Response

from python_service.api.models import OcrResponseDto, BoundingBoxDto
from python_service.preprocessing.image_cleaner import ImageCleaner
from python_service.ocr.paddle_ocr_engine import PaddleOcrEngine
from python_service.normalizer.qwen_normalizer import QwenTextNormalizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["OCR & Vision Pipeline"])

ocr_engine = PaddleOcrEngine()
qwen_normalizer = QwenTextNormalizer()

@router.post("/ocr", response_model=OcrResponseDto)
async def process_ocr(
    file: UploadFile = File(...),
    normalize_with_llm: bool = Query(default=True, description="Chuẩn hóa văn bản bằng Qwen2.5:3B")
):
    """
    Workflow chuẩn:
    [Ảnh gốc] -> (OpenCV Preprocessing) -> [Ảnh sạch] 
              -> (PaddleOCR) -> [OCR thô] 
              -> (Qwen2.5:3B LLM) -> [Văn bản chuẩn]
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File tải lên rỗng (Empty file)")

        logger.info(f"===> [BƯỚC 1] Nhận file '{file.filename}' ({len(content)} bytes). Tiền xử lý OpenCV...")
        # 1. Ảnh gốc -> OpenCV -> Ảnh sạch
        cleaned_img, _ = ImageCleaner.clean(content)

        logger.info("===> [BƯỚC 2] Chạy PaddleOCR trên ảnh sạch để nhận diện bounding boxes và text thô...")
        # 2. Ảnh sạch -> PaddleOCR -> OCR thô
        raw_ocr_text, confidence, boxes, engine_name = ocr_engine.recognize(cleaned_img)

        # 3. OCR thô -> Qwen2.5:3B -> Văn bản chuẩn
        final_text = raw_ocr_text
        if normalize_with_llm and raw_ocr_text:
            logger.info("===> [BƯỚC 3] Gửi OCR thô sang mô hình Qwen2.5:3B để chuẩn hóa văn bản & sửa lỗi chính tả...")
            final_text = await qwen_normalizer.normalize_text(raw_ocr_text)

        formatted_boxes = [
            BoundingBoxDto(text=b["text"], points=b["points"]) for b in boxes
        ]

        logger.info(f"===> [HOÀN TẤT] Trả về văn bản chuẩn ({len(final_text)} ký tự), confidence={confidence:.2f}")

        return OcrResponseDto(
            rawText=final_text,
            rawOcrText=raw_ocr_text,
            confidence=round(confidence, 4),
            boxes=formatted_boxes,
            engine=engine_name,
            normalized=normalize_with_llm
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi xử lý OCR pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý OCR: {str(e)}")

@router.post("/ocr/preprocess")
async def preprocess_image_only(file: UploadFile = File(...)):
    """
    Endpoint trả về ảnh sạch sau khi qua bộ lọc OpenCV (Deskew, CLAHE, Shadow Removal, Denoise)
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File tải lên rỗng")
        
        _, cleaned_bytes = ImageCleaner.clean(content)
        return Response(content=cleaned_bytes, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Lỗi khi tiền xử lý ảnh: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ocr/normalize-text")
async def normalize_raw_text(payload: dict):
    """
    Chuẩn hóa trực tiếp chuỗi text OCR thô bằng Qwen2.5:3B
    """
    raw_text = payload.get("rawText", "")
    if not raw_text:
        return {"normalizedText": ""}
    
    normalized = await qwen_normalizer.normalize_text(raw_text)
    return {"normalizedText": normalized}

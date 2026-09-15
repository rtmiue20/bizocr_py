import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import Response

from python_service.api.models import OcrResponseDto, BoundingBoxDto
from python_service.preprocessing.image_preprocessor import ImagePreprocessor
from python_service.preprocessing.image_cleaner import ImageCleaner
from python_service.preprocessing.structural_heuristics import StructuralHeuristics
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
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File tải lên rỗng")

        # TẦNG 0: Deskew + Auto-Crop lấy Metadata
        processed_bytes, img_meta = ImagePreprocessor.process(content)

        # BƯỚC 1: Làm sạch ảnh
        cleaned_img, _ = ImageCleaner.clean(processed_bytes)

        # BƯỚC 2: PaddleOCR nhận diện thô
        raw_ocr_text, confidence, boxes, engine_name = ocr_engine.recognize(cleaned_img)

        # BƯỚC 2.5: Phân loại Candidate nhanh bằng Aspect Ratio & Cấu trúc
        candidate_info = StructuralHeuristics.classify_candidate(
            aspect_ratio=img_meta["aspect_ratio"],
            raw_text=raw_ocr_text,
            boxes=boxes
        )

        # BƯỚC 3: Qwen LLM (Bỏ qua nếu candidate là Namecard để tiết kiệm tài nguyên)
        final_text = raw_ocr_text
        should_skip_llm = candidate_info["candidateType"] == "NAME_CARD"

        if normalize_with_llm and raw_ocr_text and not should_skip_llm:
            final_text = await qwen_normalizer.normalize_text(raw_ocr_text)

        formatted_boxes = [
            BoundingBoxDto(text=b["text"], points=b["points"]) for b in boxes
        ]

        return OcrResponseDto(
            rawText=final_text,
            rawOcrText=raw_ocr_text,
            confidence=round(confidence, 4),
            boxes=formatted_boxes,
            engine=engine_name,
            normalized=(normalize_with_llm and not should_skip_llm),
            width=img_meta["width"],
            height=img_meta["height"],
            aspectRatio=img_meta["aspect_ratio"],
            deskewAngle=img_meta["deskew_angle"],
            isCropped=img_meta["is_cropped"],
            candidateDocumentType=candidate_info["candidateType"],
            candidateConfidence=candidate_info["confidence"]
        )

    except Exception as e:
        logger.error(f"Lỗi xử lý OCR pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr/preprocess")
async def preprocess_image_only(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File tải lên rỗng")

        processed_bytes, _ = ImagePreprocessor.process(content)
        _, cleaned_bytes = ImageCleaner.clean(processed_bytes)
        return Response(content=cleaned_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
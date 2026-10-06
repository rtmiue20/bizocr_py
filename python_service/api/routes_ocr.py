import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import Response

from python_service.api.models import OcrResponseDto, BoundingBoxDto, QrResultDto, DedupResultDto
from python_service.config import settings
from python_service.preprocessing.image_preprocessor import ImagePreprocessor
from python_service.preprocessing.image_cleaner import ImageCleaner
from python_service.preprocessing.quality_gate import (
    QualityGateConfig,
    evaluate as evaluate_quality,
    metrics_snapshot,
)
from python_service.preprocessing.structural_heuristics import StructuralHeuristics
from python_service.ocr.paddle_ocr_engine import PaddleOcrEngine
from python_service.normalizer.qwen_normalizer import QwenTextNormalizer
from python_service.services.company_resolver import company_resolver
from python_service.ml.namecard_field_tagger import namecard_tagger
from python_service.ml.industry_classifier import industry_classifier
from python_service.services.dedup import dedup_service, ImageHasher

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["OCR & Vision Pipeline"])

ocr_engine = PaddleOcrEngine()
qwen_normalizer = QwenTextNormalizer()


def _quality_config() -> QualityGateConfig:
    return QualityGateConfig(
        mode=settings.QUALITY_GATE_MODE,
        min_contour_area_ratio=settings.QUALITY_GATE_MIN_CONTOUR_AREA_RATIO,
        min_width=settings.QUALITY_GATE_MIN_WIDTH,
        min_height=settings.QUALITY_GATE_MIN_HEIGHT,
        min_laplacian_variance=settings.QUALITY_GATE_MIN_LAPLACIAN,
        min_aspect_ratio=settings.QUALITY_GATE_MIN_ASPECT,
        max_aspect_ratio=settings.QUALITY_GATE_MAX_ASPECT,
        require_crop=settings.QUALITY_GATE_REQUIRE_CROP,
    )


@router.post("/ocr", response_model=OcrResponseDto)
async def process_ocr(
        file: UploadFile = File(...),
        normalize_with_llm: bool = Query(default=True, description="Chuẩn hóa văn bản bằng Qwen2.5:3B-instruct")
):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File tải lên rỗng")

        # TẦNG 0: Deskew + Auto-Crop lấy Metadata
        processed_bytes, img_meta = ImagePreprocessor.process(content)
        quality = evaluate_quality(img_meta, _quality_config())
        if not quality.passed and quality.mode == "enforce":
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "QUALITY_GATE_FAILED",
                    "reasons": quality.reasons,
                    "metadata": {
                        "contourAreaRatio": img_meta.get("contour_area_ratio"),
                        "aspectRatioAfterCrop": img_meta.get("aspect_ratio_after_crop"),
                        "width": img_meta.get("width"),
                        "height": img_meta.get("height"),
                        "meanBrightness": img_meta.get("mean_brightness"),
                        "cropFailed": img_meta.get("crop_failed")
                    },
                },
            )

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

        is_namecard = candidate_info["candidateType"] in ["NAMECARD", "CARD", "BUSINESS_CARD"] or (0.5 <= img_meta["aspect_ratio"] <= 2.2)

        # BƯỚC 3: Chuẩn hóa LLM
        final_text = raw_ocr_text
        if normalize_with_llm and raw_ocr_text:
            if is_namecard:
                final_text = await qwen_normalizer.normalize_namecard(raw_ocr_text)
            else:
                final_text = await qwen_normalizer.normalize_text(raw_ocr_text)

        # BƯỚC 4: Bóc tách thực thể Namecard (Nếu là Namecard hoặc có dấu hiệu thẻ)
        qr_info = img_meta.get("qr_result")
        qr_dto = QrResultDto(**qr_info) if qr_info else None
        extracted_fields_data = None
        dedup_dto = None

        if is_namecard or qr_info:
            qr_vcard = qr_info.get("vcardData") if qr_info else None
            tag_result = await namecard_tagger.tag_fields(
                raw_text=final_text or raw_ocr_text,
                qr_vcard=qr_vcard,
            )
            extracted_fields = tag_result.get("extractedFields", {})

            # Phân loại ngành nghề từ Tên công ty
            comp_name = extracted_fields.get("company")
            if comp_name:
                ind_res = industry_classifier.predict_industry(comp_name, text_snippet=final_text)
                extracted_fields["industry"] = ind_res.get("industry")
                extracted_fields["industryCode"] = ind_res.get("industryCode")
                extracted_fields["industryConfidence"] = ind_res.get("confidence")

            extracted_fields_data = extracted_fields

            # Kiểm tra thẻ trùng lặp (Dedup)
            img_hash = ImageHasher.dhash(processed_bytes)
            dup_res = dedup_service.check_duplicate(
                image_hash=img_hash,
                phones=extracted_fields.get("phones"),
                emails=extracted_fields.get("emails"),
                company=extracted_fields.get("company"),
                name=extracted_fields.get("name"),
            )
            dedup_dto = DedupResultDto(**dup_res)

        # BƯỚC 4.5: Hợp nhất thông tin công ty qua DuckDuckGo / CompanyResolver
        company_info = None
        company_target = (extracted_fields_data.get("company") if extracted_fields_data else None) or (final_text.split("\n")[0] if final_text else "")
        if company_target and len(company_target) > 5:
            company_info = await company_resolver.resolve_company(company_name=company_target)

        formatted_boxes = [BoundingBoxDto(text=b["text"], points=b["points"]) for b in boxes]

        return OcrResponseDto(
            rawText=final_text,
            rawOcrText=raw_ocr_text,
            confidence=round(confidence, 4),
            boxes=formatted_boxes,
            engine=engine_name,
            normalized=normalize_with_llm,
            width=img_meta["width"],
            height=img_meta["height"],
            aspectRatio=img_meta["aspect_ratio"],
            aspectRatioAfterCrop=img_meta["aspect_ratio_after_crop"],
            originalWidth=img_meta["original_width"],
            originalHeight=img_meta["original_height"],
            contourAreaRatio=img_meta["contour_area_ratio"],
            laplacianVariance=img_meta["laplacian_variance"],
            meanBrightness=img_meta["mean_brightness"],
            isCropped=img_meta["is_cropped"],
            cropFailed=img_meta["crop_failed"],
            qualityPassed=quality.passed,
            qualityReasons=quality.reasons,
            qualityMode=quality.mode.upper(),
            candidateDocumentType="NAMECARD" if is_namecard else candidate_info["candidateType"],
            candidateConfidence=candidate_info["confidence"],
            companyInfo=company_info,
            qrResult=qr_dto,
            extractedFields=extracted_fields_data,
            dedupInfo=dedup_dto,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi xử lý OCR pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr/preprocess")
async def preprocess_image_only(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File tải lên rỗng")

        processed_bytes, img_meta = ImagePreprocessor.process(content)
        quality = evaluate_quality(img_meta, _quality_config())
        if not quality.passed and quality.mode == "enforce":
            raise HTTPException(
                status_code=422,
                detail={"code": "QUALITY_GATE_FAILED", "reasons": quality.reasons},
            )
        _, cleaned_bytes = ImageCleaner.clean(processed_bytes)
        return Response(content=cleaned_bytes, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ocr/quality-metrics")
async def quality_metrics():
    """Bộ đếm in-process cho shadow mode — không ghi MariaDB."""
    snap = metrics_snapshot(_quality_config())
    snap["mode"] = settings.QUALITY_GATE_MODE
    return snap

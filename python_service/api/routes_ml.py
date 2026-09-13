import logging
from fastapi import APIRouter, HTTPException
from python_service.api.models import (
    PredictRequestDto, PredictResponseDto,
    CollectRequestDto,
    ExtractRequestDto, ExtractResponseDto,
    ClusterRequestDto,
    AnomalyRequestDto
)
from python_service.ml.classifier_ml import document_ml_service
from python_service.normalizer.qwen_normalizer import QwenTextNormalizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml", tags=["Machine Learning & Analytics"])

@router.post("/predict", response_model=PredictResponseDto)
async def predict_document_type(request: PredictRequestDto):
    """
    Dự đoán loại chứng từ từ raw text bằng mô hình ML nội bộ (TF-IDF + LogisticRegression)
    """
    res = document_ml_service.predict(request.rawText)
    return PredictResponseDto(
        predictedType=res.get("predictedType", "UNKNOWN"),
        confidence=res.get("confidence", 0.0)
    )

@router.post("/collect")
async def collect_training_data(request: CollectRequestDto):
    """
    Thu thập dữ liệu nhãn được duyệt từ kế toán để self-learning huấn luyện mô hình
    """
    document_ml_service.collect(request.rawText, request.label)
    return {"status": "success", "message": f"Collected sample for label {request.label}"}

@router.post("/extract", response_model=ExtractResponseDto)
async def extract_fields_ml(request: ExtractRequestDto):
    """
    Trích xuất các trường thông tin cơ bản bằng ML/Pattern hỗ trợ Java Core
    """
    text = request.rawText or ""
    doc_type = request.documentType or "UNKNOWN"
    
    extracted = {}
    import re
    # Match MST
    mst_match = re.search(r"(?:Mã số thuế|MST|Tax code)[:\s]*([0-9]{10}(?:-[0-9]{3})?)", text, re.IGNORECASE)
    if mst_match:
        extracted["tax_code"] = mst_match.group(1).strip()
        
    # Match Tổng tiền
    amount_match = re.search(r"(?:Tổng tiền|Tổng cộng|Thành tiền|Tổng thanh toán)[:\s]*([0-9\.,]+)", text, re.IGNORECASE)
    if amount_match:
        extracted["total_amount"] = amount_match.group(1).strip()
        
    return ExtractResponseDto(extractedFields=extracted)

@router.post("/cluster-unknowns")
async def cluster_unknown_documents(request: ClusterRequestDto):
    """
    Gom cụm các chứng từ chưa phân loại được bằng K-Means để phát hiện nhóm chứng từ mới
    """
    res = document_ml_service.cluster_unknowns(request.n_clusters)
    return res

@router.post("/analyze-anomaly")
async def analyze_anomaly(request: AnomalyRequestDto):
    """
    Phân tích phát hiện số tiền bất thường (Z-Score / 3-sigma)
    """
    res = document_ml_service.analyze_anomaly(request.current_amount, request.historical_amounts)
    return res

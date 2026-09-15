import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

# Import service ML đã định nghĩa trong file classifier_ml.py
from python_service.ml.classifier_ml import document_ml_service

logger = logging.getLogger(__name__)

# Đặt prefix chuẩn khớp với Java Client gọi sang
router = APIRouter(prefix="/api/v1/ml", tags=["Machine Learning Classification"])

# --- Các class định nghĩa dữ liệu đầu vào (Payload) ---
class PredictRequest(BaseModel):
    rawText: str

class CollectRequest(BaseModel):
    rawText: str
    label: str

class ExtractRequest(BaseModel):
    rawText: str
    documentType: str

class ClusterRequest(BaseModel):
    n_clusters: int

class AnomalyRequest(BaseModel):
    current_amount: float
    historical_amounts: List[float]


# --- Các API Endpoints ---

@router.post("/predict")
async def predict_document(payload: PredictRequest):
    """API dự đoán loại chứng từ (Predict)"""
    try:
        return document_ml_service.predict(payload.rawText)
    except Exception as e:
        logger.error(f"Lỗi endpoint predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/collect")
async def collect_data(payload: CollectRequest):
    """API thu thập dữ liệu huấn luyện (Collect)"""
    try:
        document_ml_service.collect(payload.rawText, payload.label)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Lỗi endpoint collect: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract")
async def extract_fields(payload: ExtractRequest):
    """API bóc tách dữ liệu (Extract)"""
    try:
        # Hiện tại DocumentMlService chưa có hàm xử lý extract riêng.
        # Trả về JSON rỗng "{}" để Java Client phân tích an toàn, không bị lỗi.
        return {"extractedFields": "{}"}
    except Exception as e:
        logger.error(f"Lỗi endpoint extract: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cluster-unknowns")
async def cluster_unknowns(payload: ClusterRequest):
    """API phân cụm K-Means cho tài liệu chưa biết"""
    try:
        return document_ml_service.cluster_unknowns(payload.n_clusters)
    except Exception as e:
        logger.error(f"Lỗi endpoint cluster-unknowns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-anomaly")
async def analyze_anomaly(payload: AnomalyRequest):
    """API phân tích dị thường (Anomaly Detection)"""
    try:
        return document_ml_service.analyze_anomaly(payload.current_amount, payload.historical_amounts)
    except Exception as e:
        logger.error(f"Lỗi endpoint analyze-anomaly: {e}")
        raise HTTPException(status_code=500, detail=str(e))
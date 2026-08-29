import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.data_pipeline.cleaner import process_and_clean_data
from app.data_pipeline.extractor import extract_all_raw_data, save_raw_to_json
from app.data_pipeline.trainer import train_system_model
from app.services.extractor_service import extractor_service
from app.services.system_ml_service import system_ml
from app.services.clustering_service import clustering_service
from app.services.anomaly_service import anomaly_service
from typing import List

logger = logging.getLogger(__name__)
router = APIRouter()

# --- DTO Models (Request Body) ---
class PredictRequest(BaseModel):
    rawText: str


class CollectDataRequest(BaseModel):
    rawText: str
    label: Optional[str] = "UNKNOWN"


class ExtractRequest(BaseModel):
    rawText: str
    documentType: Optional[str] = "UNKNOWN"

class ClusterRequest(BaseModel):
    n_clusters: int = 3

class AnomalyRequest(BaseModel):
    current_amount: float
    historical_amounts: List[float]

# --- API Endpoints ---
@router.post("/predict")
async def predict_document_type(req: PredictRequest):
    """Spring Boot gọi API này để dự đoán phân loại chứng từ bằng ML"""
    return system_ml.predict(req.rawText)


@router.post("/collect")
async def collect_raw_data(req: CollectDataRequest):
    """Lưu văn bản OCR thô và nhãn (label) đã duyệt vào data/raw/ để làm dữ liệu train"""
    try:
        # Hàm save_raw_to_json giờ đã hỗ trợ nhận tham số label một cách an toàn
        filepath = save_raw_to_json(req.rawText, req.label)
        return {"status": "success", "file": filepath}
    except Exception as e:
        logger.error(f"Lỗi khi thu thập dữ liệu thô: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract")
async def extract_document_fields(req: ExtractRequest):
    """Spring Boot gọi API này để bóc tách thông tin chi tiết (MST, Tổng tiền...) từ văn bản OCR"""
    try:
        fields = extractor_service.extract_fields(req.rawText, req.documentType)
        return {
            "status": "success",
            "extractedFields": fields
        }
    except Exception as e:
        logger.error(f"Lỗi khi bóc tách thông tin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-pipeline")
async def trigger_full_pipeline():
    """Kích hoạt trích xuất -> làm sạch -> chia dữ liệu 8/2 -> huấn luyện model mới"""
    try:
        # Bước 1: Trích xuất toàn bộ dữ liệu thô
        records = extract_all_raw_data()
        if not records:
            return {"status": "warning", "message": "Chưa có dữ liệu thô trong data/raw/"}

        # Bước 2: Làm sạch & Chuẩn hóa dữ liệu
        cleaned_file = process_and_clean_data(records)
        if not cleaned_file:
            return {"status": "error", "message": "Quá trình làm sạch dữ liệu thất bại"}

        # Bước 3: Chia 8/2 và Train model
        model_path = train_system_model(cleaned_file, test_size=0.2)
        if not model_path:
            return {"status": "warning", "message": "Dữ liệu chưa đủ số lượng hoặc quá trình train gặp lỗi"}

        return {"status": "success", "message": "Đã hoàn tất pipeline và cập nhật model mới", "model": model_path}
    except Exception as e:
        logger.error(f"Lỗi pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cluster-unknowns")
async def cluster_unknowns(req: ClusterRequest):
    """(Unsupervised Learning) Dùng K-Means gom cụm các chứng từ chưa phân loại để gợi ý nhãn"""
    return clustering_service.cluster_unknown_documents(req.n_clusters)

@router.post("/analyze-anomaly")
async def analyze_anomaly(req: AnomalyRequest):
    """(Anomaly Detection) Dùng Mean, Median, Regression để phát hiện số tiền bất thường"""
    return anomaly_service.detect_amount_anomaly(req.current_amount, req.historical_amounts)
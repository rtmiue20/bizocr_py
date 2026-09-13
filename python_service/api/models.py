from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class BoundingBoxDto(BaseModel):
    text: str
    points: List[int] # [x1, y1, x2, y2, x3, y3, x4, y4]

class OcrResponseDto(BaseModel):
    rawText: str # Văn bản đã chuẩn hóa qua Qwen2.5:3b (hoặc OCR thô nếu chưa qua)
    rawOcrText: Optional[str] = None # Văn bản OCR thô ban đầu trước khi sửa lỗi
    confidence: float
    boxes: List[BoundingBoxDto] = []
    engine: str = "paddleocr"
    normalized: bool = True

class PredictRequestDto(BaseModel):
    rawText: str

class PredictResponseDto(BaseModel):
    predictedType: str
    confidence: float

class CollectRequestDto(BaseModel):
    rawText: str
    label: str

class ExtractRequestDto(BaseModel):
    rawText: str
    documentType: Optional[str] = "UNKNOWN"

class ExtractResponseDto(BaseModel):
    extractedFields: Dict[str, Any]

class ClusterRequestDto(BaseModel):
    n_clusters: int = 5

class AnomalyRequestDto(BaseModel):
    current_amount: float
    historical_amounts: List[float]

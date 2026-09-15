from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class BoundingBoxDto(BaseModel):
    text: str
    points: List[int]

class OcrResponseDto(BaseModel):
    rawText: str
    rawOcrText: Optional[str] = None
    confidence: float
    boxes: List[BoundingBoxDto] = []
    engine: str = "paddleocr"
    normalized: bool = True
    width: Optional[int] = 0
    height: Optional[int] = 0
    aspectRatio: Optional[float] = 0.0
    deskewAngle: Optional[float] = 0.0
    isCropped: Optional[bool] = False
    candidateDocumentType: Optional[str] = "UNKNOWN"
    candidateConfidence: Optional[float] = 0.0
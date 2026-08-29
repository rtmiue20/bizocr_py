from pydantic import BaseModel
from typing import List, Optional

class BoundingBox(BaseModel):
    text: str
    points: List[int]

class OcrResponse(BaseModel):
    rawText: str
    confidence: float
    boxes: Optional[List[BoundingBox]] = []
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class BoundingBoxDto(BaseModel):
    text: str
    points: List[int]


class QrResultDto(BaseModel):
    hasQr: bool = False
    qrText: Optional[str] = None
    qrType: Optional[str] = None  # VCARD | URL | TEXT
    vcardData: Optional[Dict[str, Any]] = None


class NamecardFieldsDto(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    phones: List[str] = []
    emails: List[str] = []
    websites: List[str] = []
    address: Optional[str] = None
    taxCode: Optional[str] = None
    industry: Optional[str] = None
    industryConfidence: Optional[float] = 0.0
    extractionMethod: Optional[str] = "REGEX_HEURISTICS"  # QR_VCARD | REGEX_HEURISTICS | HYBRID_LLM


class DedupResultDto(BaseModel):
    isDuplicate: bool = False
    reason: Optional[str] = None
    imageHash: Optional[str] = None
    identityKey: Optional[str] = None
    matchedId: Optional[str] = None


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
    aspectRatioAfterCrop: Optional[float] = 0.0
    originalWidth: Optional[int] = 0
    originalHeight: Optional[int] = 0
    contourAreaRatio: Optional[float] = 0.0
    laplacianVariance: Optional[float] = 0.0
    meanBrightness: Optional[float] = 0.0
    deskewAngle: Optional[float] = 0.0
    isCropped: Optional[bool] = False
    cropFailed: Optional[bool] = True
    qualityPassed: Optional[bool] = True
    qualityReasons: List[str] = []
    qualityMode: Optional[str] = "SHADOW"
    candidateDocumentType: Optional[str] = "UNKNOWN"
    candidateConfidence: Optional[float] = 0.0
    # Thông tin bổ sung
    companyInfo: Optional[dict] = None
    qrResult: Optional[QrResultDto] = None
    extractedFields: Optional[Dict[str, Any]] = None
    dedupInfo: Optional[DedupResultDto] = None


class ExtractRequest(BaseModel):
    rawText: str
    documentType: Optional[str] = "NAMECARD"
    ownerCompanyId: Optional[str] = None
    ocrBoxes: Optional[List[Dict[str, Any]]] = None


class ExtractResponse(BaseModel):
    status: str = "ACCEPTED"
    extractedFields: Dict[str, Any] = {}
    completenessScore: float = 1.0
    llmInvoked: bool = False
    extractionMethod: str = "REGEX_HEURISTICS"
    errors: List[str] = []


class IndustryPredictRequest(BaseModel):
    companyName: str
    textSnippet: Optional[str] = None


class IndustryPredictResponse(BaseModel):
    industry: str
    industryCode: str
    confidence: float
    explanation: Optional[str] = None


class SemanticSearchItem(BaseModel):
    cardId: Optional[str] = None
    score: float
    company: Optional[str] = None
    name: Optional[str] = None
    industry: Optional[str] = None
    matchedSnippet: Optional[str] = None


class SemanticSearchRequest(BaseModel):
    query: str
    topK: Optional[int] = 10
    candidateCards: Optional[List[Dict[str, Any]]] = None


class SemanticSearchResponse(BaseModel):
    query: str
    results: List[SemanticSearchItem] = []
    total: int = 0


class DedupCheckRequest(BaseModel):
    phone: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    name: Optional[str] = None
    imageHash: Optional[str] = None
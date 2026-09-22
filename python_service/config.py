import os
from pydantic import BaseModel


class Settings(BaseModel):
    PROJECT_NAME: str = "BizOCR Vision & Text Normalization Engine"
    API_V1_STR: str = "/api/v1"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Ollama LLM Configuration for Qwen2.5:3B text normalization
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
    OLLAMA_TIMEOUT_SECONDS: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30"))

    # OCR Settings
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "paddleocr")
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() in ("true", "1", "t")
    OCR_LANG: str = os.getenv("OCR_LANG", "vi")  # Vietnamese + Latin
    OCR_MAX_IMAGE_DIM: int = int(os.getenv("OCR_MAX_IMAGE_DIM", "4000"))

    # Preprocessing Settings
    AUTO_DESKEW: bool = True
    ENHANCE_CONTRAST: bool = True
    DENOISE: bool = True

    # Quality gate after crop/deskew. shadow = attach + log, still OCR.
    # enforce = HTTP 422 trước OCR nếu không đạt. Đổi khi đã đo tỉ lệ false-positive.
    QUALITY_GATE_MODE: str = os.getenv("QUALITY_GATE_MODE", "shadow").lower()
    QUALITY_GATE_MIN_CONTOUR_AREA_RATIO: float = float(
        os.getenv("QUALITY_GATE_MIN_CONTOUR_AREA_RATIO", "0.08")
    )
    QUALITY_GATE_MIN_WIDTH: int = int(os.getenv("QUALITY_GATE_MIN_WIDTH", "200"))
    QUALITY_GATE_MIN_HEIGHT: int = int(os.getenv("QUALITY_GATE_MIN_HEIGHT", "200"))
    QUALITY_GATE_MIN_LAPLACIAN: float = float(
        os.getenv("QUALITY_GATE_MIN_LAPLACIAN", "40")
    )
    QUALITY_GATE_MIN_ASPECT: float = float(os.getenv("QUALITY_GATE_MIN_ASPECT", "0.30"))
    QUALITY_GATE_MAX_ASPECT: float = float(os.getenv("QUALITY_GATE_MAX_ASPECT", "3.50"))
    QUALITY_GATE_REQUIRE_CROP: bool = os.getenv(
        "QUALITY_GATE_REQUIRE_CROP", "false"
    ).lower() in ("true", "1", "t")


settings = Settings()
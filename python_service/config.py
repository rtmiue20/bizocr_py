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


settings = Settings()
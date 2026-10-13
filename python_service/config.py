import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

# --- Directory Configuration ---
# Đảm bảo đường dẫn tuyệt đối dựa trên vị trí thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parents[1]
CLASSIFIER_DIR = BASE_DIR / "saved_models" / "classifier"
CORRECTION_DIR = BASE_DIR / "saved_models" / "corrections"
GOLDEN_DIR = BASE_DIR / "golden"
LEARNING_DIR = BASE_DIR / "saved_models" / "learning"
ACTIVE_MODEL_DIR = BASE_DIR / "saved_models" / "active_model"
DATASETS_DIR = BASE_DIR / "datasets"
INDUSTRY_DIR = BASE_DIR / "saved_models" / "industry"
FEWSHOT_DIR = BASE_DIR / "saved_models" / "few_shot"


# Đọc tự động file .env nếu có (không phụ thuộc thư viện ngoài)
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    try:
        with open(_env_file, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    except Exception:
        pass


def detect_ollama_model(base_url: str = "http://localhost:11434", default_model: str = "qwen2.5:3b-instruct") -> str:
    """
    Tự động dò tìm tag model Qwen có sẵn trong Ollama để tránh lỗi 404 Not Found.
    Ưu tiên: qwen2.5:3b-instruct -> qwen2.5:3b -> qwen2.5:latest -> default_model.
    """
    try:
        import httpx
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{base_url.rstrip('/')}/api/tags")
            if resp.status_code == 200:
                tags_data = resp.json()
                models = [m.get("name", "") for m in tags_data.get("models", [])]
                # 1. Khớp chính xác
                for preferred in [default_model, "qwen2.5:3b-instruct", "qwen2.5:3b", "qwen2.5:latest"]:
                    if preferred in models:
                        return preferred
                # 2. Khớp tiền tố/chứa qwen2.5
                for m in models:
                    if "qwen2.5" in m:
                        return m
    except Exception:
        pass
    return default_model


class Settings(BaseModel):
    PROJECT_NAME: str = "BizOCR Vision & Text Normalization Engine"
    API_V1_STR: str = "/api/v1"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Security & Database Configuration (Đồng bộ mặc định với Java Core: dev-internal-token / dev-admin-token)
    INTERNAL_TOKEN: Optional[str] = os.getenv("INTERNAL_TOKEN", "dev-internal-token")
    ADMIN_TOKEN: Optional[str] = os.getenv("ADMIN_TOKEN", "dev-admin-token")

    DB_HOST: Optional[str] = os.getenv("DB_HOST")
    DB_PORT: Optional[int] = int(os.getenv("DB_PORT", "3306")) if os.getenv("DB_HOST") else None
    DB_USER: Optional[str] = os.getenv("DB_USER")
    DB_PASSWORD: Optional[str] = os.getenv("DB_PASSWORD")
    DB_NAME: Optional[str] = os.getenv("DB_NAME")

    # Standardized Directory Paths
    BASE_DIR: Path = BASE_DIR
    CLASSIFIER_DIR: Path = CLASSIFIER_DIR
    CORRECTION_DIR: Path = CORRECTION_DIR
    GOLDEN_DIR: Path = GOLDEN_DIR
    LEARNING_DIR: Path = LEARNING_DIR
    ACTIVE_MODEL_DIR: Path = ACTIVE_MODEL_DIR
    DATASETS_DIR: Path = DATASETS_DIR
    INDUSTRY_DIR: Path = INDUSTRY_DIR
    FEWSHOT_DIR: Path = FEWSHOT_DIR

    # Ollama LLM Configuration for Qwen2.5:3B text normalization
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
    OLLAMA_TIMEOUT_SECONDS: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30"))

    # OCR Settings
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "paddleocr")
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() in ("true", "1", "t")
    OCR_LANG: str = os.getenv("OCR_LANG", "vi")
    OCR_MAX_IMAGE_DIM: int = int(os.getenv("OCR_MAX_IMAGE_DIM", "1800"))

    # Preprocessing Settings
    AUTO_DESKEW: bool = True
    ENHANCE_CONTRAST: bool = True
    DENOISE: bool = True

    # Quality Gate Settings
    # Mặc định 'shadow': vẫn cho phép OCR xử lý để không chặn các ảnh webcam hơi mờ, chỉ gắn cờ đánh dấu
    QUALITY_GATE_MODE: str = os.getenv("QUALITY_GATE_MODE", "shadow").lower()
    QUALITY_GATE_MIN_CONTOUR_AREA_RATIO: float = float(
        os.getenv("QUALITY_GATE_MIN_CONTOUR_AREA_RATIO", "0.08")
    )
    QUALITY_GATE_MIN_WIDTH: int = int(os.getenv("QUALITY_GATE_MIN_WIDTH", "200"))
    QUALITY_GATE_MIN_HEIGHT: int = int(os.getenv("QUALITY_GATE_MIN_HEIGHT", "200"))
    # Ngưỡng Laplacian 20.0 phù hợp với cả ảnh chụp webcam laptop và điện thoại
    QUALITY_GATE_MIN_LAPLACIAN: float = float(
        os.getenv("QUALITY_GATE_MIN_LAPLACIAN", "20.0")
    )
    QUALITY_GATE_MIN_ASPECT: float = float(os.getenv("QUALITY_GATE_MIN_ASPECT", "0.30"))
    QUALITY_GATE_MAX_ASPECT: float = float(os.getenv("QUALITY_GATE_MAX_ASPECT", "3.50"))
    QUALITY_GATE_REQUIRE_CROP: bool = os.getenv(
        "QUALITY_GATE_REQUIRE_CROP", "false"
    ).lower() in ("true", "1", "t")

    # OCR Thresholds
    OCR_LINE_MERGE_THRESHOLD_RATIO: float = float(os.getenv("OCR_LINE_MERGE_THRESHOLD_RATIO", "0.6"))

    # Document Heuristics
    HEURISTIC_ASPECT_RATIO_A4_MIN: float = float(os.getenv("HEURISTIC_ASPECT_RATIO_A4_MIN", "0.65"))
    HEURISTIC_ASPECT_RATIO_A4_MAX: float = float(os.getenv("HEURISTIC_ASPECT_RATIO_A4_MAX", "1.45"))

    # Excel Export Template Path
    EXCEL_TEMPLATE_PATH: str = os.getenv("EXCEL_TEMPLATE_PATH", "data/templates/export_template.xlsx")

    def create_required_directories(self) -> None:
        """Tự động tạo đầy đủ hệ thống thư mục cấu hình nếu chưa tồn tại."""
        for directory in (
            self.CLASSIFIER_DIR,
            self.CORRECTION_DIR,
            self.GOLDEN_DIR,
            self.LEARNING_DIR,
            self.ACTIVE_MODEL_DIR,
            self.DATASETS_DIR,
            self.INDUSTRY_DIR,
            self.FEWSHOT_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
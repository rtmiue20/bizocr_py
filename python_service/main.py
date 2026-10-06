import hmac
import os
import logging
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Depends
from python_service.config import settings
from python_service.api.routes_ocr import router as ocr_router
from python_service.api.routes_ml import router as ml_router, alias_router as ml_alias_router
from python_service.api.routes_search import router as search_router
from python_service.api.routes_export import router as export_router

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("bizocr-python-service")


def require_internal_token(x_internal_token: str | None = Header(None, alias="X-Internal-Token")):
    """Xác thực token nội bộ giữa Java Core và Python Service."""
    expected = settings.INTERNAL_TOKEN or os.getenv("INTERNAL_TOKEN", "dev-internal-token")
    if not expected:
        logger.error("INTERNAL_TOKEN chưa được cấu hình trong môi trường!")
        raise HTTPException(
            status_code=500,
            detail="Cấu hình bảo mật hệ thống chưa hoàn tất."
        )

    if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
        raise HTTPException(
            status_code=403,
            detail="X-Internal-Token không hợp lệ hoặc bị thiếu."
        )


from contextlib import asynccontextmanager

# Mặc định tắt tài liệu OpenAPI/Swagger công khai trong môi trường Production
ENV = os.getenv("ENVIRONMENT", "production").lower()
is_production = ENV in ("production", "prod")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.create_required_directories()
    logger.info("BizOCR Service đã khởi tạo thành công hệ thống thư mục cấu hình.")
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="2.0.0",
    description="Microservice xử lý ảnh OpenCV, PaddleOCR, Chuẩn hóa văn bản Qwen2.5:3B và Machine Learning",
    docs_url=None if is_production else "/docs",
    redoc_url=None if is_production else "/redoc",
    openapi_url=None if is_production else "/openapi.json",
    lifespan=lifespan,
)

# Đăng ký Routers với lớp bảo vệ token nội bộ
app.include_router(ocr_router, dependencies=[Depends(require_internal_token)])
app.include_router(ml_router, dependencies=[Depends(require_internal_token)])
app.include_router(ml_alias_router, dependencies=[Depends(require_internal_token)])
app.include_router(search_router, dependencies=[Depends(require_internal_token)])
app.include_router(export_router, dependencies=[Depends(require_internal_token)])


@app.get("/")
@app.get("/health")
@app.get("/api/v1/health")
async def health_check():
    """Endpoint kiểm tra trạng thái dịch vụ (không yêu cầu token)."""
    return {
        "status": "UP",
        "service": settings.PROJECT_NAME,
        "ocr_engine": settings.OCR_ENGINE,
        "llm_normalizer": settings.OLLAMA_MODEL,
        "workflow": "Ảnh gốc -> OpenCV -> Ảnh sạch -> PaddleOCR -> OCR thô -> Qwen2.5:3B -> Văn bản chuẩn -> Java Core"
    }


if __name__ == "__main__":
    logger.info(f"Khởi động {settings.PROJECT_NAME} trên {settings.HOST}:{settings.PORT}")
    # Đã sửa: Nạp theo chuẩn module path để chạy ổn định từ root folder trong Docker
    uvicorn.run("python_service.main:app", host=settings.HOST, port=settings.PORT, reload=False)
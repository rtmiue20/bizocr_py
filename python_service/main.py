import torch

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from python_service.config import settings
from python_service.api.routes_ocr import router as ocr_router
from python_service.api.routes_ml import router as ml_router
from python_service.api.routes_search import router as search_router
from python_service.api.routes_export import router as export_router


# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("bizocr-python-service")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="2.0.0",
    description="Microservice xử lý ảnh OpenCV, PaddleOCR, Chuẩn hóa văn bản Qwen2.5:3B và Machine Learning"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký Routers
app.include_router(ocr_router)
app.include_router(ml_router)
app.include_router(search_router)
app.include_router(export_router)

@app.get("/")
@app.get("/health")
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "UP",
        "service": settings.PROJECT_NAME,
        "ocr_engine": settings.OCR_ENGINE,
        "llm_normalizer": settings.OLLAMA_MODEL,
        "workflow": "Ảnh gốc -> OpenCV -> Ảnh sạch -> PaddleOCR -> OCR thô -> Qwen2.5:3B -> Văn bản chuẩn -> Java Core"
    }

if __name__ == "__main__":
    logger.info(f"Khởi động {settings.PROJECT_NAME} trên {settings.HOST}:{settings.PORT}")
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=False)

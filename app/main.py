from fastapi import FastAPI
import uvicorn
from app.api.v1.ocr_router import router as ocr_router
from app.api.v1.ml_router import router as ml_router

app = FastAPI(
    title="BizOCR Vision Engine",
    description="Microservice xử lý ảnh và OCR",
    version="1.0.0"
)

# Route trực tiếp /ocr và /api/v1/ocr để khớp endpoint từ Java
app.include_router(ocr_router)
app.include_router(ml_router, prefix="/api/v1/ml")
app.include_router(ocr_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
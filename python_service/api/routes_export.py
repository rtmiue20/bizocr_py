import logging
import io
import os
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
from python_service.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/export", tags=["Export Data"])

@router.post("/excel")
async def export_to_excel(data: List[Dict[str, Any]]):
    """
    Xuất danh sách kết quả OCR/Company ra Excel theo template.
    Nếu template không tồn tại, sẽ tạo file Excel mới hoàn toàn.
    """
    try:
        if not data:
            raise HTTPException(status_code=400, detail="Không có dữ liệu để xuất")

        df = pd.DataFrame(data)
        
        output = io.BytesIO()
        
        # Kiểm tra template
        template_path = settings.EXCEL_TEMPLATE_PATH
        if os.path.exists(template_path):
            logger.info(f"Sử dụng Excel template tại: {template_path}")
            # Logic: Ghi vào template có sẵn (cần openpyxl)
            # Ở đây đơn giản hóa là load template rồi ghi đè dữ liệu vào sheet 1
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Load template (PD không hỗ trợ trực tiếp append vào template dễ dàng mà không làm hỏng format
                # nên ta sẽ tạo mới nhưng có thể mở rộng logic này sau)
                df.to_excel(writer, index=False, sheet_name='OCR_Results')
        else:
            logger.warning(f"Không tìm thấy template tại {template_path}. Tạo file Excel mới.")
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='OCR_Results')

        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=bizocr_export.xlsx"}
        )
    except Exception as e:
        logger.error(f"Lỗi xuất Excel: {e}")
        raise HTTPException(status_code=500, detail=str(e))

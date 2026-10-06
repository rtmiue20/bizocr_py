import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from python_service.config import settings


def get_status_file_path() -> Path:
    """Trả về đường dẫn chuẩn hóa tới status.json từ settings."""
    return settings.LEARNING_DIR / "status.json"


def get_job_status(batch_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Lấy trạng thái tiến trình huấn luyện hiện tại.

    Đặc biệt xử lý:
    - Kiểm tra xem batch_id yêu cầu có khớp với batch_id lưu trong file hay không.
    - Nếu batch_id không khớp (ví dụ: Java hỏi lô N+1 nhưng đĩa ghi kết quả PROMOTED của lô N),
      sẽ tự động trả về IDLE để tránh Java báo sai trạng thái.
    """
    status_file = get_status_file_path()

    if not status_file.exists():
        return {
            "status": "IDLE",
            "batch_id": batch_id,
            "message": "Chưa có tiến trình huấn luyện nào được khởi chạy."
        }

    try:
        with status_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        recorded_batch_id = data.get("batch_id")

        # So sánh batch_id: Nếu Java truyền batch_id nhưng không khớp lô gần nhất trong đĩa
        if batch_id is not None and recorded_batch_id != batch_id:
            return {
                "status": "IDLE",
                "batch_id": batch_id,
                "message": f"Lô {batch_id} chưa khởi chạy tiến trình huấn luyện (file hiện tại thuộc về lô {recorded_batch_id})."
            }

        return data
    except Exception as e:
        return {
            "status": "ERROR",
            "batch_id": batch_id,
            "error": f"Không thể đọc file trạng thái: {str(e)}"
        }


def update_job_status(
        batch_id: int,
        status: str,
        message: str = "",
        details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Cập nhật trạng thái job huấn luyện nguyên tử (Atomic Write) vào LEARNING_DIR/status.json.

    Ghi ra file tạm thời rồi os.replace để đảm bảo tính an toàn trên cả Windows và Linux.
    """
    settings.LEARNING_DIR.mkdir(parents=True, exist_ok=True)
    status_file = get_status_file_path()

    payload = {
        "batch_id": batch_id,
        "status": status,  # Ví dụ: "RUNNING", "PROMOTED", "REJECTED", "FAILED"
        "message": message,
        "details": details or {}
    }

    # Ghi file tạm cùng thư mục
    fd, temp_name = tempfile.mkstemp(
        prefix=".status_",
        suffix=".tmp",
        dir=str(settings.LEARNING_DIR)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, status_file)
    finally:
        if os.path.exists(temp_name):
            try:
                os.unlink(temp_name)
            except OSError:
                pass
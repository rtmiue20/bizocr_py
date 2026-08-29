import os
import json
import uuid

RAW_DATA_DIR = "data/raw/"
os.makedirs(RAW_DATA_DIR, exist_ok=True)


def save_raw_to_json(raw_text: str, label: str = "UNKNOWN") -> str:
    """Lưu văn bản thô từ OCR và nhãn thành file JSON vào thư mục data/raw/"""
    file_id = str(uuid.uuid4())
    filepath = os.path.join(RAW_DATA_DIR, f"doc_{file_id}.json")

    data = {
        "raw_content": raw_text,
        "label": label
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return filepath


def extract_all_raw_data() -> list:
    """Quét toàn bộ thư mục data/raw/ để lấy rawText và label phục vụ huấn luyện"""
    extracted_records = []

    for filename in os.listdir(RAW_DATA_DIR):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(RAW_DATA_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            raw_text = data.get("raw_content", "")
            label = data.get("label", "UNKNOWN")

            # Chỉ lấy các file có chứa text
            if not raw_text.strip():
                continue

            # Map đúng tên cột để cleaner.py và trainer.py sử dụng
            record = {
                "doc_id": filename,
                "rawText": raw_text,
                "label": label
            }
            extracted_records.append(record)
        except Exception as e:
            print(f"Lỗi đọc file {filepath}: {e}")

    return extracted_records
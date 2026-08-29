from app.data_pipeline.extractor import extract_all_raw_data
from app.data_pipeline.cleaner import process_and_clean_data
from app.data_pipeline.trainer import train_system_model

if __name__ == "__main__":
    print("=== BẮT ĐẦU CHẠY DATA PIPELINE ===")

    print("[1/3] Trích xuất dữ liệu thô từ data/raw/...")
    records = extract_all_raw_data()
    print(f"-> Tìm thấy {len(records)} bản ghi.")

    if records:
        print("[2/3] Làm sạch và chuẩn hóa dữ liệu...")
        cleaned_path = process_and_clean_data(records)

        print("[3/3] Chia tập dữ liệu (8/2) và Huấn luyện Mô hình ML...")
        train_system_model(cleaned_path, test_size=0.2)

    print("=== HOÀN THÀNH PIPELINE ===")
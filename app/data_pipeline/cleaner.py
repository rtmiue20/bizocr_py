import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)
PROCESSED_DATA_DIR = "data/processed/"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def process_and_clean_data(records: list):
    """Làm sạch, chuẩn hóa và lưu dữ liệu đã xử lý"""
    if not records:
        logger.warning("Không có dữ liệu để làm sạch.")
        return None

    df = pd.DataFrame(records)

    # 1. Loại bỏ trùng lặp linh hoạt theo các cột hiện có
    subset_cols = [col for col in ["rawText", "label", "Ma_CTY", "MST", "Tong_Tien"] if col in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)
    else:
        df = df.drop_duplicates()

    # 2. Xử lý giá trị bị thiếu (Fix lỗi median trên kiểu String)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
        else:
            df[col] = df[col].fillna('Unknown')

    # 3. Xử lý ngoại lệ (Outliers) bằng IQR cho các cột số
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # 4. Chuẩn hóa & Mã hóa
    # Mã hóa cột Categorical nếu có
    if 'Ma_CTY' in df.columns:
        encoder = LabelEncoder()
        df['Ma_CTY_Encoded'] = encoder.fit_transform(df['Ma_CTY'].astype(str))

    # Chuẩn hóa các cột số về khoảng (0 -> 1)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Lưu file CSV đã làm sạch vào data/processed/
    output_path = os.path.join(PROCESSED_DATA_DIR, "cleaned_dataset.csv")
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Đã làm sạch và lưu {len(df)} bản ghi vào {output_path}")

    return output_path
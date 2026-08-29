import os
import pandas as pd
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
MODEL_PATH = "saved_models/bizocr_classifier.pkl"
MIN_SAMPLES = 2  # Để test nhanh


def train_system_model(csv_path: str, test_size: float = 0.2):
    """Huấn luyện mô hình NLP phân loại văn bản OCR"""
    try:
        df = pd.read_csv(csv_path)

        # Kiểm tra điều kiện dữ liệu
        if 'rawText' not in df.columns or 'label' not in df.columns:
            logger.error("Dữ liệu thiếu cột 'rawText' hoặc 'label'")
            return None

        df = df.dropna(subset=['rawText', 'label'])

        # Bỏ qua các nhãn UNKNOWN không có giá trị phân loại
        df = df[df['label'] != 'UNKNOWN']

        if len(df) < MIN_SAMPLES:
            logger.warning(f"Dữ liệu chưa đủ số lượng tối thiểu để train (cần {MIN_SAMPLES})")
            return None

        X = df['rawText']
        y = df['label']

        # Xây dựng quy trình học máy (Pipeline)
        ml_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, lowercase=True, stop_words=None)),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # Huấn luyện mô hình (fit toàn bộ data để test nhanh)
        ml_pipeline.fit(X, y)

        # Lưu toàn bộ Pipeline (bao gồm cả Tfidf và Model)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(ml_pipeline, MODEL_PATH)

        logger.info(f"Đã huấn luyện và lưu mô hình mới tại {MODEL_PATH}")

        # Trả về đường dẫn file mô hình nếu thành công
        return MODEL_PATH

    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện: {e}")
        return None
import os
import re
import joblib
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
MODEL_PATH = "saved_models/bizocr_classifier.pkl"


class SystemMlService:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Tải mô hình đã huấn luyện từ ổ đĩa nếu tồn tại"""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                logger.info(f"[ML Service] Đã nạp thành công mô hình từ {MODEL_PATH}")
            except Exception as e:
                logger.error(f"[ML Service] Lỗi khi nạp mô hình: {e}")
                self.model = None
        else:
            logger.warning(
                f"[ML Service] Chưa tìm thấy file mô hình tại {MODEL_PATH}. Mô hình sẽ trả về 0.0 confidence.")

    def _extract_features_from_text(self, raw_text: str) -> pd.DataFrame:
        """Trích xuất các đặc trưng số/mã hóa từ văn bản thô OCR để đưa vào model"""
        tong_tien = re.search(r'Tổng tiền:\s*([\d\.,]+)', raw_text, re.IGNORECASE)
        tien_val = float(tong_tien.group(1).replace(',', '').replace('.', '')) if tong_tien else 0.0

        # Tạo DataFrame chứa các feature khớp với các cột đã train trong trainer.py
        data = {
            "Tong_Tien": [tien_val],
            "Ma_CTY_Encoded": [0]  # Mặc định hoặc bóc tách mã CTY nếu có
        }
        return pd.DataFrame(data)

    def predict(self, raw_text: str):
        """Dự đoán loại chứng từ dựa trên văn bản thô OCR"""
        # Nếu model chưa được train hoặc text rỗng -> Trả về độ tin cậy 0.0 để Spring Boot fallback sang Ollama
        if not os.path.exists(MODEL_PATH):
            self.load_model()

        if self.model is None or not raw_text:
            return {
                "predictedType": "UNKNOWN",
                "confidence": 0.0,
                "matchedBy": "ML_SYSTEM"
            }

        try:
            features = self._extract_features_from_text(raw_text)

            # Dự đoán lớp (Class) và xác suất (Probability)
            predicted_type = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))

            return {
                "predictedType": str(predicted_type),
                "confidence": round(confidence, 2),
                "matchedBy": "ML_SYSTEM"
            }
        except Exception as e:
            logger.error(f"[ML Service] Lỗi thực thi dự đoán: {e}")
            return {
                "predictedType": "UNKNOWN",
                "confidence": 0.0,
                "matchedBy": "ML_SYSTEM_ERROR"
            }


# Khởi tạo instance dùng chung cho toàn hệ thống
system_ml = SystemMlService()
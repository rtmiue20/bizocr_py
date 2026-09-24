import os
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class DocumentMlService:
    """
    Quản lý mô hình Machine Learning: Phân loại văn bản, Huấn luyện tự động, Clustering và Phân tích dị thường
    """

    def __init__(self, data_file: str = "training_data.jsonl"):
        self.data_file = data_file
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=500, class_weight='balanced')
        self.is_trained = False
        self.collected_data: List[Dict[str, str]] = []
        self._load_data_and_train()

    def _load_data_and_train(self):
        # Nạp dữ liệu huấn luyện mặc định nếu có
        default_samples = [
            {"label": "VAT_INVOICE", "text": "HÓA ĐƠN GIÁ TRỊ GIA TĂNG HÓA ĐƠN ĐIỆN TỬ MÃ SỐ THUẾ CỘNG TIỀN HÀNG THUẾ SUẤT GTGT TỔNG TIỀN THANH TOÁN"},
            {"label": "SERVICE_PAYMENT_VOUCHER", "text": "PHIẾU THANH TOÁN DỊCH VỤ PHIẾU THU PHIẾU CHI BIÊN LAI THÀNH TIỀN TỔNG CHI PHÍ TIỀN MẶT"},
            {"label": "ELECTRICITY_BILL", "text": "TỔNG CÔNG TY ĐIỆN LỰC HÓA ĐƠN TIỀN ĐIỆN EVN MÃ KHÁCH HÀNG CHỈ SỐ CŨ CHỈ SỐ MỚI ĐIỆN NĂNG TIÊU THỤ"},
            {"label": "WATER_BILL", "text": "CÔNG TY CỔ PHẦN CẤP NƯỚC HÓA ĐƠN TIỀN NƯỚC MÃ KHÁCH HÀNG CHỈ SỐ ĐỒNG HỒ TIÊU THỤ NƯỚC SINH HOẠT"},
            {"label": "CONTRACT", "text": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM HỢP ĐỒNG KINH TẾ BÊN A BÊN B ĐIỀU KHOẢN NGHIỆM THU THANH TOÁN"},
            {"label": "BANK_STATEMENT", "text": "ỦY NHIỆM CHI SAO KÊ TÀI KHOẢN NGÂN HÀNG SỐ TÀI KHOẢN SỐ DƯ TÀI KHOẢN GIAO DỊCH CHUYỂN TIỀN"},
            {"label": "ID_CARD", "text": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM CĂN CƯỚC CÔNG DÂN HỌ VÀ TÊN NGÀY SINH NƠI THƯỜNG TRÚ GIÁ TRỊ ĐẾN"},
            {"label": "NAMECARD", "text": "CARD VISIT GIÁM ĐỐC KINH DOANH EMAIL MOBILE PHONE TEL WEBSITE ADDRESS COMPANY"}
        ]
        
        self.collected_data.extend(default_samples)

        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.collected_data.append(json.loads(line))
            except Exception as e:
                logger.error(f"Lỗi khi đọc file training_data.jsonl: {e}")

        self._train()

    def _train(self):
        if len(self.collected_data) < 2:
            return
        
        try:
            texts = [d["text"] for d in self.collected_data]
            labels = [d["label"] for d in self.collected_data]
            
            # Chỉ train nếu có ít nhất 2 classes khác nhau
            unique_labels = set(labels)
            if len(unique_labels) >= 2:
                X = self.vectorizer.fit_transform(texts)
                self.model.fit(X, labels)
                self.is_trained = True
                logger.info(f"Đã huấn luyện TF-IDF + LogisticRegression với {len(texts)} mẫu trên {len(unique_labels)} nhãn.")
        except Exception as e:
            logger.error(f"Lỗi trong quá trình huấn luyện: {e}")

    def predict(self, raw_text: str) -> Dict[str, Any]:
        if not raw_text or not self.is_trained:
            return {"predictedType": "UNKNOWN", "confidence": 0.0}

        try:
            X = self.vectorizer.transform([raw_text])
            probabilities = self.model.predict_proba(X)[0]
            max_idx = int(np.argmax(probabilities))
            predicted_class = self.model.classes_[max_idx]
            confidence = float(probabilities[max_idx])

            return {
                "predictedType": predicted_class,
                "confidence": round(confidence, 4)
            }
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán loại chứng từ: {e}")
            return {"predictedType": "UNKNOWN", "confidence": 0.0}

    def collect(self, raw_text: str, label: str):
        if not raw_text or not label:
            return

        record = {"label": label, "text": raw_text}
        self.collected_data.append(record)
        
        # Ghi vào file
        try:
            with open(self.data_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"Đã lưu mẫu huấn luyện mới: label={label}")
            
            # Huấn luyện lại theo chu kỳ
            if len(self.collected_data) % 5 == 0:
                self._train()
        except Exception as e:
            logger.error(f"Lỗi khi lưu dữ liệu collect: {e}")

    def cluster_unknowns(self, n_clusters: int = 5) -> Dict[str, Any]:
        if len(self.collected_data) < n_clusters:
            return {"clusters": [], "message": "Không đủ dữ liệu để phân cụm"}

        try:
            texts = [d["text"] for d in self.collected_data]
            X = self.vectorizer.transform(texts)
            k_clusters = min(n_clusters, len(texts))
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            kmeans.fit(X)

            results = []
            for i, label in enumerate(kmeans.labels_):
                results.append({
                    "sample_index": i,
                    "cluster": int(label),
                    "original_label": self.collected_data[i]["label"]
                })
            return {"clusters": results, "k": k_clusters}
        except Exception as e:
            logger.error(f"Lỗi khi phân cụm: {e}")
            return {"clusters": [], "error": str(e)}

    @staticmethod
    def analyze_anomaly(current_amount: float, historical_amounts: List[float]) -> Dict[str, Any]:
        if not historical_amounts or len(historical_amounts) < 3:
            return {
                "is_anomaly": False,
                "z_score": 0.0,
                "reason": "Chưa đủ dữ liệu lịch sử để tính độ lệch chuẩn"
            }

        arr = np.array(historical_amounts)
        mean = float(np.mean(arr))
        std = float(np.std(arr))

        if std == 0.0:
            is_anomaly = abs(current_amount - mean) > 0.01
            z_score = 999.0 if is_anomaly else 0.0
        else:
            z_score = abs(current_amount - mean) / std
            is_anomaly = z_score > 3.0 # Ngưỡng 3-sigma

        return {
            "is_anomaly": bool(is_anomaly),
            "z_score": round(float(z_score), 2),
            "mean": round(mean, 2),
            "std": round(std, 2),
            "current_amount": current_amount
        }

document_ml_service = DocumentMlService()

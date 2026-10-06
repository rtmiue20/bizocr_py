import os
import json
import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class DocumentMlService:
    """
    Quản lý mô hình Machine Learning: Phân loại văn bản, Huấn luyện tự động, Clustering, Phân tích dị thường.
    Bảo vệ suy luận: Hoán đổi atomic dựa trên đánh giá tập eval độc lập. Đồng bộ Uvicorn worker qua mtime.
    """

    DEFAULT_SAMPLES = [
        {"label": "VAT_INVOICE",
         "text": "HÓA ĐƠN GIÁ TRỊ GIA TĂNG HÓA ĐƠN ĐIỆN TỬ MÃ SỐ THUẾ CỘNG TIỀN HÀNG THUẾ SUẤT GTGT TỔNG TIỀN THANH TOÁN"},
        {"label": "SERVICE_PAYMENT_VOUCHER",
         "text": "PHIẾU THANH TOÁN DỊCH VỤ PHIẾU THU PHIẾU CHI BIÊN LAI THÀNH TIỀN TỔNG CHI PHÍ TIỀN MẶT"},
        {"label": "ELECTRICITY_BILL",
         "text": "TỔNG CÔNG TY ĐIỆN LỰC HÓA ĐƠN TIỀN ĐIỆN EVN MÃ KHÁCH HÀNG CHỈ SỐ CŨ CHỈ SỐ MỚI ĐIỆN NĂNG TIÊU THỤ"},
        {"label": "WATER_BILL",
         "text": "CÔNG TY CỔ PHẦN CẤP NƯỚC HÓA ĐƠN TIỀN NƯỚC MÃ KHÁCH HÀNG CHỈ SỐ ĐỒNG HỒ TIÊU THỤ NƯỚC SINH HOẠT"},
        {"label": "CONTRACT",
         "text": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM HỢP ĐỒNG KINH TẾ BÊN A BÊN B ĐIỀU KHOẢN NGHIỆM THU THANH TOÁN"},
        {"label": "BANK_STATEMENT",
         "text": "ỦY NHIỆM CHI SAO KÊ TÀI KHOẢN NGÂN HÀNG SỐ TÀI KHOẢN SỐ DƯ TÀI KHOẢN GIAO DỊCH CHUYỂN TIỀN"},
        {"label": "ID_CARD",
         "text": "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM CĂN CƯỚC CÔNG DÂN HỌ VÀ TÊN NGÀY SINH NƠI THƯỜNG TRÚ GIÁ TRỊ ĐẾN"},
        {"label": "NAMECARD", "text": "CARD VISIT GIÁM ĐỐC KINH DOANH EMAIL MOBILE PHONE TEL WEBSITE ADDRESS COMPANY"}
    ]

    def __init__(self, data_file: str = "datasets/train_source.jsonl", model_dir: str = "saved_models/classifier"):
        self.data_file = Path(data_file)
        self.legacy_file = Path("training_data.legacy.jsonl")
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=500, class_weight='balanced')
        self.is_trained = False
        self.model_version = "v0"
        self.classes_: List[str] = []
        self.collected_data: List[Dict[str, str]] = []
        self._last_manifest_mtime: float = 0.0

        if not self.reload():
            self._load_data_and_train_initial()

    def _load_data_and_train_initial(self):
        """Khởi tạo bootstrap mô hình trên 100% dữ liệu mẫu ban đầu (không chia 80/20 để tránh mất lớp)."""
        self.collected_data.extend(self.DEFAULT_SAMPLES)

        if self.data_file.exists():
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        label = (
                            record.get("metadata", {}).get("document_type")
                            or record.get("label")
                            or record.get("document_type")
                        )
                        # Hỗ trợ đọc cả thuộc tính "input" từ dataset export
                        text = (
                            record.get("input")
                            or record.get("raw_ocr_text")
                            or record.get("text")
                        )
                        if label and text:
                            self.collected_data.append({"label": str(label), "text": str(text)})
            except Exception as e:
                logger.error(f"Lỗi khi đọc file {self.data_file}: {e}")

        # Khởi tạo Bootstrap: Train trực tiếp trên toàn bộ mẫu thu thập được không qua cổng so sánh
        bundle_res = self.build_candidate(self.collected_data)
        if bundle_res.get("status") == "SUCCESS":
            bundle = bundle_res["bundle"]
            acc = self.score_bundle(bundle, self.collected_data)
            self.promote_bundle(bundle, accuracy=acc)
            logger.info(f"Khởi tạo bootstrap mô hình thành công với {len(self.collected_data)} mẫu.")
        else:
            logger.error(f"Khởi tạo bootstrap mô hình thất bại: {bundle_res.get('reason')}")

    def build_candidate(self, train_samples: List[Dict[str, str]]) -> Dict[str, Any]:
        """Tạo ứng viên huấn luyện trong bộ nhớ mà không ghi đè active model."""
        if not train_samples or len(train_samples) < 2:
            return {"status": "ERROR", "reason": "Số lượng mẫu quá ít (< 2)"}

        train_texts = [d["text"] for d in train_samples if d.get("text")]
        train_labels = [d["label"] for d in train_samples if d.get("label")]

        unique_labels = set(train_labels)
        if len(unique_labels) < 2:
            return {"status": "ERROR", "reason": "Cần ít nhất 2 lớp khác nhau để huấn luyện"}

        try:
            candidate_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            candidate_model = LogisticRegression(max_iter=500, class_weight='balanced')

            X_train = candidate_vectorizer.fit_transform(train_texts)
            candidate_model.fit(X_train, train_labels)

            now_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            new_version = f"cls-{now_str}"

            bundle = {
                "vectorizer": candidate_vectorizer,
                "model": candidate_model,
                "version": new_version,
                "classes": list(candidate_model.classes_),
                "trained_at": now_str,
                "sample_count": len(train_texts),
            }
            return {"status": "SUCCESS", "bundle": bundle}
        except Exception as e:
            logger.error(f"Lỗi khi build_candidate: {e}")
            return {"status": "ERROR", "reason": str(e)}

    def score_bundle(self, bundle: Dict[str, Any], eval_samples: List[Dict[str, str]]) -> float:
        """Chấm điểm một bundle trên tập eval."""
        if not eval_samples:
            return 0.0

        eval_texts = [d["text"] for d in eval_samples if d.get("text")]
        eval_labels = [d["label"] for d in eval_samples if d.get("label")]

        if not eval_texts:
            return 0.0

        try:
            cand_vectorizer = bundle["vectorizer"]
            cand_model = bundle["model"]
            X_eval = cand_vectorizer.transform(eval_texts)
            preds = cand_model.predict(X_eval)
            correct = sum(1 for p, y in zip(preds, eval_labels) if p == y)
            return correct / len(eval_labels)
        except Exception as e:
            logger.error(f"Lỗi khi score_bundle: {e}")
            return 0.0

    def promote_bundle(self, bundle: Dict[str, Any], accuracy: float) -> Dict[str, Any]:
        """Lưu bundle thành mô hình active chính thức và hoán đổi atomic."""
        now_str = bundle["trained_at"]
        new_version = bundle["version"]
        joblib_path = self.model_dir / f"classifier_{new_version}.joblib"
        manifest_path = self.model_dir / "classifier_manifest.json"

        bundle_to_save = {
            **bundle,
            "accuracy": accuracy
        }
        joblib.dump(bundle_to_save, joblib_path)

        manifest_data = {
            "active_version": new_version,
            "active_joblib": str(joblib_path.name),
            "accuracy": round(accuracy, 4),
            "classes": bundle["classes"],
            "updated_at": now_str,
        }

        manifest_temp = self.model_dir / "classifier_manifest.json.tmp"
        with open(manifest_temp, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2)
        manifest_temp.replace(manifest_path)

        with self._lock:
            self.vectorizer = bundle["vectorizer"]
            self.model = bundle["model"]
            self.model_version = new_version
            self.classes_ = bundle["classes"]
            self.is_trained = True
            self._last_manifest_mtime = manifest_path.stat().st_mtime

        logger.info(f"Đã nâng cấp mô hình active thành công -> {new_version} (Acc: {accuracy:.2%})")
        return {"status": "PROMOTED", "version": new_version, "accuracy": accuracy}

    def retrain(self, train_samples: List[Dict[str, str]], eval_samples: List[Dict[str, str]]) -> Dict[str, Any]:
        """Huấn luyện ứng viên, kiểm tra chống mất lớp, chấm điểm và hoán đổi nếu đạt."""
        # 1. Build Candidate
        res = self.build_candidate(train_samples)
        if res.get("status") != "SUCCESS":
            return {"status": "SKIPPED", "reason": res.get("reason", "Cannot build candidate")}

        bundle = res["bundle"]
        candidate_classes = set(bundle["classes"])

        # 2. Kiểm tra chống mất lớp (Class Loss Guard)
        active_classes = set(self.classes_) if self.is_trained else set()
        if active_classes and not active_classes.issubset(candidate_classes):
            missing = active_classes - candidate_classes
            msg = f"Candidate bị mất các lớp so với Active: {missing}"
            logger.warning(f"[REJECTED] {msg}")
            return {"status": "REJECTED", "reason": msg, "missing_classes": list(missing)}

        # 3. Đánh giá Candidate vs Active
        if not eval_samples:
            eval_samples = train_samples

        cand_acc = self.score_bundle(bundle, eval_samples)

        curr_acc = 0.0
        if self.is_trained:
            active_bundle = {"vectorizer": self.vectorizer, "model": self.model}
            curr_acc = self.score_bundle(active_bundle, eval_samples)

        # 4. Kiểm tra thoái lui (Regression Check)
        if cand_acc < curr_acc:
            msg = f"Candidate acc ({cand_acc:.4f}) thấp hơn Active acc ({curr_acc:.4f})"
            logger.warning(f"[REJECTED] {msg}")
            return {
                "status": "REJECTED",
                "reason": msg,
                "candidate_accuracy": cand_acc,
                "current_accuracy": curr_acc
            }

        # 5. Nâng cấp (Promote)
        promote_res = self.promote_bundle(bundle, accuracy=cand_acc)
        promote_res["previous_accuracy"] = curr_acc
        return promote_res

    def reload(self, version: Optional[str] = None) -> bool:
        """Nạp lại mô hình từ đĩa."""
        try:
            manifest_path = self.model_dir / "classifier_manifest.json"
            target_joblib: Optional[Path] = None

            if version:
                specific_path = self.model_dir / f"classifier_{version}.joblib"
                if specific_path.is_file():
                    target_joblib = specific_path
            elif manifest_path.is_file():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        fname = data.get("active_joblib")
                        if fname:
                            p = self.model_dir / fname
                            if p.is_file():
                                target_joblib = p
                except Exception:
                    pass

            if target_joblib is None:
                joblib_files = sorted(self.model_dir.glob("classifier_*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
                if joblib_files:
                    target_joblib = joblib_files[0]

            if target_joblib and target_joblib.is_file():
                bundle = joblib.load(target_joblib)
                with self._lock:
                    self.vectorizer = bundle["vectorizer"]
                    self.model = bundle["model"]
                    self.model_version = bundle.get("version", target_joblib.stem)
                    self.classes_ = bundle.get("classes", list(getattr(self.model, "classes_", [])))
                    self.is_trained = True
                    if manifest_path.is_file():
                        self._last_manifest_mtime = manifest_path.stat().st_mtime
                logger.info(f"Đã nạp mô hình phân loại: {self.model_version}")
                return True
        except Exception as e:
            logger.error(f"Không thể nạp mô hình classifier: {e}")
        return False

    def _sync_worker_model(self):
        """Đồng bộ mô hình worker qua mtime."""
        manifest_path = self.model_dir / "classifier_manifest.json"
        if manifest_path.is_file():
            current_mtime = manifest_path.stat().st_mtime
            if current_mtime > self._last_manifest_mtime:
                logger.info("Phát hiện mô hình mới trên đĩa, tiến hành đồng bộ worker...")
                self.reload()

    def predict(self, raw_text: str) -> Dict[str, Any]:
        self._sync_worker_model()

        if not raw_text or not self.is_trained:
            return {"predictedType": "UNKNOWN", "confidence": 0.0, "modelVersion": self.model_version}

        try:
            with self._lock:
                X = self.vectorizer.transform([raw_text])
                probabilities = self.model.predict_proba(X)[0]
                classes = self.model.classes_

            max_idx = int(np.argmax(probabilities))
            predicted_class = classes[max_idx]
            confidence = float(probabilities[max_idx])

            return {
                "predictedType": predicted_class,
                "confidence": round(confidence, 4),
                "modelVersion": self.model_version,
            }
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán loại chứng từ: {e}")
            return {"predictedType": "UNKNOWN", "confidence": 0.0, "modelVersion": self.model_version}

    def collect(self, raw_text: str, label: str):
        if not raw_text or not label:
            return
        record = {"label": label, "text": raw_text}
        try:
            with open(self.legacy_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Lỗi khi lưu dữ liệu collect: {e}")

    def cluster_unknowns(self, raw_texts: List[str], n_clusters: int = 5) -> Dict[str, Any]:
        """Gom nhóm các văn bản chưa nhận dạng (UNKNOWN) sử dụng KMeans."""
        if not raw_texts:
            return {"clusters": {}, "n_clusters": 0, "message": "Không có dữ liệu để gom nhóm."}

        n_samples = len(raw_texts)
        actual_clusters = min(n_clusters, n_samples)
        if actual_clusters < 1:
            return {"clusters": {}, "n_clusters": 0, "message": "Số lượng mẫu quá ít."}

        try:
            vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            X = vec.fit_transform(raw_texts)

            kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)

            grouped = defaultdict(list)
            for text, cluster_id in zip(raw_texts, cluster_labels):
                grouped[int(cluster_id)].append(text)

            return {
                "clusters": dict(grouped),
                "n_clusters": actual_clusters,
                "total_samples": n_samples
            }
        except Exception as e:
            logger.error(f"Lỗi khi gom nhóm UNKNOWN: {e}")
            return {"clusters": {}, "n_clusters": 0, "error": str(e)}

    @staticmethod
    def analyze_anomaly(current_amount: float, historical_amounts: List[float]) -> Dict[str, Any]:
        """Phân tích dị thường giá trị số (ví dụ: tổng tiền hóa đơn) bằng phương pháp Z-Score."""
        if current_amount is None:
            return {"is_anomaly": False, "score": 0.0, "reason": "Thiếu giá trị hiện tại"}

        if not historical_amounts or len(historical_amounts) < 3:
            return {
                "is_anomaly": False,
                "score": 0.0,
                "reason": "Chưa đủ dữ liệu lịch sử để phân tích dị thường (cần >= 3 bản ghi)"
            }

        try:
            amounts = np.array(historical_amounts, dtype=float)
            mean = float(np.mean(amounts))
            std = float(np.std(amounts))

            if std == 0:
                is_anomaly = current_amount != mean
                return {
                    "is_anomaly": is_anomaly,
                    "score": 99.0 if is_anomaly else 0.0,
                    "mean": mean,
                    "std": 0.0,
                    "reason": f"Dữ liệu lịch sử cố định. Giá trị hiện tại {'khác' if is_anomaly else 'bằng'} lịch sử."
                }

            z_score = abs(current_amount - mean) / std
            is_anomaly = bool(z_score > 3.0)

            return {
                "is_anomaly": is_anomaly,
                "score": round(float(z_score), 2),
                "mean": round(mean, 2),
                "std": round(std, 2),
                "reason": f"Mức độ dị thường Z-Score = {z_score:.2f} (> 3.0)" if is_anomaly else "Giá trị nằm trong dải bình thường"
            }
        except Exception as e:
            logger.error(f"Lỗi khi phân tích dị thường: {e}")
            return {"is_anomaly": False, "score": 0.0, "error": str(e)}


document_ml_service = DocumentMlService()
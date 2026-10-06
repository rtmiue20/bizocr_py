import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

from python_service.config import settings

logger = logging.getLogger(__name__)


# Danh mục 12 ngành nghề kinh tế trọng tâm
INDUSTRY_CATALOG = {
    "CONSTRUCTION_REALESTATE": {
        "name_vi": "Xây dựng & Bất động sản",
        "description": "Xây dựng công trình, bất động sản, kiến trúc, vật liệu xây dựng, nội thất",
    },
    "IT_SOFTWARE": {
        "name_vi": "Công nghệ thông tin & Phần mềm",
        "description": "Phần mềm, viễn thông, chuyển đổi số, trí tuệ nhân tạo, thiết bị số",
    },
    "LOGISTICS_TRANSPORT": {
        "name_vi": "Logistics & Vận tải",
        "description": "Vận chuyển hàng hóa, kho bãi, giao nhận chuyển phát, xuất nhập khẩu vận tải",
    },
    "MANUFACTURING_INDUSTRIAL": {
        "name_vi": "Sản xuất & Công nghiệp",
        "description": "Cơ khí, chế tạo, luyện kim, hóa chất, bao bì, gia công công nghiệp",
    },
    "FINANCE_BANKING_INSURANCE": {
        "name_vi": "Tài chính, Ngân hàng & Bảo hiểm",
        "description": "Ngân hàng, chứng khoán, bảo hiểm, quỹ đầu tư, fintech, tài chính doanh nghiệp",
    },
    "COMMERCE_RETAIL": {
        "name_vi": "Thương mại & Bán lẻ",
        "description": "Bán buôn, bán lẻ, xuất nhập khẩu tổng hợp, phân phối hàng tiêu dùng",
    },
    "HEALTHCARE_PHARMA": {
        "name_vi": "Y tế & Dược phẩm",
        "description": "Bệnh viện, phòng khám, dược phẩm, mỹ phẩm, thiết bị y tế",
    },
    "EDUCATION_TRAINING": {
        "name_vi": "Giáo dục & Đào tạo",
        "description": "Trường học, trung tâm ngoại ngữ, đào tạo nghề, tư vấn du học, giáo dục trực tuyến",
    },
    "MARKETING_MEDIA_PRINTING": {
        "name_vi": "Quảng cáo, Truyền thông & In ấn",
        "description": "Quảng cáo truyền thông, tổ chức sự kiện, in ấn bao bì ấn phẩm, thiết kế đồ họa",
    },
    "HOSPITALITY_FOOD_BEVERAGE": {
        "name_vi": "F&B, Nhà hàng & Du lịch",
        "description": "Dịch vụ ăn uống, nhà hàng, khách sạn, lữ hành du lịch, khu nghỉ dưỡng",
    },
    "LEGAL_CONSULTING": {
        "name_vi": "Pháp lý & Tư vấn",
        "description": "Văn phòng luật sư, kiểm toán kế toán thuế, tư vấn quản trị doanh nghiệp",
    },
    "AGRICULTURE": {
        "name_vi": "Nông nghiệp & Thủy hải sản",
        "description": "Trồng trọt, chăn nuôi, chế biến thủy hải sản, thức ăn chăn nuôi, nông sản",
    },
}

# Tập mẫu hạt giống (Seed Samples) khởi tạo ban đầu cho mô hình
SEED_SAMPLES: List[Tuple[str, str]] = [
    # CONSTRUCTION_REALESTATE
    ("Công ty Cổ phần Tập đoàn Xây dựng Hòa Bình", "CONSTRUCTION_REALESTATE"),
    ("Công ty Cổ phần Vinhomes Bất động sản Đô thị", "CONSTRUCTION_REALESTATE"),
    ("Công ty TNHH Tư vấn Thiết kế Kiến trúc và Nội thất Xanh", "CONSTRUCTION_REALESTATE"),
    ("Tổng Công ty Xi măng Vật liệu Xây dựng Việt Nam", "CONSTRUCTION_REALESTATE"),
    ("Công ty CP Đầu tư Xây dựng Cầu đường Giao thông", "CONSTRUCTION_REALESTATE"),

    # IT_SOFTWARE
    ("Công ty Cổ phần Viễn thông FPT Phần mềm và Chuyển đổi số", "IT_SOFTWARE"),
    ("BizOCR AI Solutions Trí tuệ Nhân tạo và Phần mềm", "IT_SOFTWARE"),
    ("Công ty TNHH Giải pháp Công nghệ Thông tin Viettel", "IT_SOFTWARE"),
    ("Tập đoàn Công nghệ CMC Phần mềm và An ninh mạng", "IT_SOFTWARE"),
    ("Công ty Cổ phần Dịch vụ Điện toán Đám mây và Phần cứng", "IT_SOFTWARE"),

    # LOGISTICS_TRANSPORT
    ("Công ty Cổ phần Giao Hàng Tiết Kiệm Chuyển phát Nhanh", "LOGISTICS_TRANSPORT"),
    ("Công ty TNHH Logistics Toàn Cầu Vận tải Quốc tế Kho bãi", "LOGISTICS_TRANSPORT"),
    ("Tổng Công ty Hàng hải Việt Nam Cảng biển Vận chuyển", "LOGISTICS_TRANSPORT"),
    ("Công ty Vận tải Giao nhận Container Đường bộ Hải phòng", "LOGISTICS_TRANSPORT"),

    # MANUFACTURING_INDUSTRIAL
    ("Tập đoàn Thép Hòa Phát Sản xuất Luyện kim Cơ khí", "MANUFACTURING_INDUSTRIAL"),
    ("Công ty TNHH Cơ khí Chế tạo Máy Công nghiệp Nặng", "MANUFACTURING_INDUSTRIAL"),
    ("Nhà máy Sản xuất Nhựa Bao bì Công nghiệp Chế xuất", "MANUFACTURING_INDUSTRIAL"),
    ("Công ty CP Thiết bị Điện và Gia công Kim loại", "MANUFACTURING_INDUSTRIAL"),

    # FINANCE_BANKING_INSURANCE
    ("Ngân hàng Thương mại Cổ phần Ngoại thương Việt Nam Vietcombank", "FINANCE_BANKING_INSURANCE"),
    ("Công ty Cổ phần Chứng khoán SSI Đầu tư Tài chính", "FINANCE_BANKING_INSURANCE"),
    ("Tổng Công ty Bảo hiểm Bảo Việt Nhân thọ Phi nhân thọ", "FINANCE_BANKING_INSURANCE"),
    ("Quỹ Đầu tư Khởi nghiệp và Quản lý Tài sản Doanh nghiệp", "FINANCE_BANKING_INSURANCE"),

    # COMMERCE_RETAIL
    ("Công ty Cổ phần Đầu tư Bán lẻ Thế Giới Di Động", "COMMERCE_RETAIL"),
    ("Công ty TNHH Thương mại Dịch vụ Xuất Nhập khẩu Hàng Tiêu dùng", "COMMERCE_RETAIL"),
    ("Hệ thống Chuỗi Siêu thị và Phân phối Thực phẩm", "COMMERCE_RETAIL"),

    # HEALTHCARE_PHARMA
    ("Công ty Cổ phần Dược Hậu Giang Dược phẩm Y tế", "HEALTHCARE_PHARMA"),
    ("Bệnh viện Đa khoa Quốc tế Vinmec Khám chữa bệnh", "HEALTHCARE_PHARMA"),
    ("Công ty TNHH Thiết bị Y tế và Hóa chất Xét nghiệm", "HEALTHCARE_PHARMA"),
    ("Phòng khám Nha khoa Thẩm mỹ Răng hàm mặt", "HEALTHCARE_PHARMA"),

    # EDUCATION_TRAINING
    ("Trường Đại học Quốc tế và Viện Đào tạo Doanh nhân", "EDUCATION_TRAINING"),
    ("Hệ thống Giáo dục Trực tuyến và Trung tâm Ngoại ngữ", "EDUCATION_TRAINING"),
    ("Công ty Tư vấn Du học và Đào tạo Kỹ năng Nghề nghiệp", "EDUCATION_TRAINING"),

    # MARKETING_MEDIA_PRINTING
    ("Công ty TNHH Quảng cáo Truyền thông Tiếp thị Kỹ thuật số", "MARKETING_MEDIA_PRINTING"),
    ("Công ty CP In ấn Bao bì Quảng cáo và Ấn phẩm Hội chợ", "MARKETING_MEDIA_PRINTING"),
    ("Tổ chức Sự kiện Event Media và Thiết kế Đồ họa Thương hiệu", "MARKETING_MEDIA_PRINTING"),

    # HOSPITALITY_FOOD_BEVERAGE
    ("Chuỗi Nhà hàng Ẩm thực Lẩu Nướng BBQ Buffet", "HOSPITALITY_FOOD_BEVERAGE"),
    ("Công ty Cổ phần Khách sạn và Du lịch Lữ hành Saigontourist", "HOSPITALITY_FOOD_BEVERAGE"),
    ("Công ty TNHH Đồ uống Cafe Trà sữa Giải khát", "HOSPITALITY_FOOD_BEVERAGE"),

    # LEGAL_CONSULTING
    ("Văn phòng Luật sư Tranh tụng và Tư vấn Pháp lý Doanh nghiệp", "LEGAL_CONSULTING"),
    ("Công ty TNHH Dịch vụ Kế toán Thuế và Kiểm toán Độc lập", "LEGAL_CONSULTING"),
    ("Tư vấn Chiến lược Quản trị Doanh nghiệp và Tái cấu trúc", "LEGAL_CONSULTING"),

    # AGRICULTURE
    ("Công ty Cổ phần Thủy sản Minh Phú Tôm Cá Xuất khẩu", "AGRICULTURE"),
    ("Tập đoàn Lộc Trời Giống cây trồng Phân bón Nông nghiệp", "AGRICULTURE"),
    ("Công ty TNHH Chăn nuôi Gia súc Gia cầm Nông sản Sạch", "AGRICULTURE"),
]


class IndustryClassifier:
    """
    Phân loại ngành nghề từ tên công ty và đoạn mô tả dịch vụ bằng TF-IDF + k-Nearest Neighbors (kNN).
    Hỗ trợ học tăng cường liên tục (Online Continuous Learning) khi người dùng gán nhãn ngành mới.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or settings.INDUSTRY_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "industry_knn.joblib"
        self.samples_path = self.model_dir / "industry_samples.jsonl"
        self._lock = threading.Lock()

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=4000,
            sublinear_tf=True,
            analyzer="word"
        )
        self.knn = KNeighborsClassifier(n_neighbors=3, metric="cosine", weights="distance")
        self.samples: List[Tuple[str, str]] = []
        self.is_trained = False

        self._initialize_or_load()

    def _initialize_or_load(self):
        """Khởi tạo từ đĩa hoặc nạp tập seed ban đầu."""
        loaded = self._load()
        if not loaded:
            logger.info("Chưa có mô hình kNN ngành nghề trên đĩa. Khởi tạo từ bộ mẫu hạt giống...")
            self.samples = list(SEED_SAMPLES)
            self._fit_and_save()

    def _fit(self):
        if not self.samples or len(self.samples) < 3:
            return

        texts = [s[0] for s in self.samples]
        labels = [s[1] for s in self.samples]

        X = self.vectorizer.fit_transform(texts)
        k = min(3, len(self.samples))
        self.knn.set_params(n_neighbors=k)
        self.knn.fit(X, labels)
        self.is_trained = True

    def _save(self):
        bundle = {
            "vectorizer": self.vectorizer,
            "knn": self.knn,
            "sample_count": len(self.samples),
        }
        joblib.dump(bundle, self.model_path)

        with open(self.samples_path, "w", encoding="utf-8") as f:
            for text, lbl in self.samples:
                f.write(json.dumps({"text": text, "label": lbl}, ensure_ascii=False) + "\n")

    def _fit_and_save(self):
        self._fit()
        self._save()
        logger.info(f"Đã huấn luyện và lưu kNN phân loại ngành nghề với {len(self.samples)} mẫu.")

    def _load(self) -> bool:
        if self.model_path.is_file() and self.samples_path.is_file():
            try:
                bundle = joblib.load(self.model_path)
                with self._lock:
                    self.vectorizer = bundle["vectorizer"]
                    self.knn = bundle["knn"]
                    self.is_trained = True

                self.samples = []
                with open(self.samples_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rec = json.loads(line)
                            self.samples.append((rec["text"], rec["label"]))
                logger.info(f"Đã nạp kNN phân loại ngành nghề ({len(self.samples)} mẫu) từ {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Lỗi khi nạp kNN industry: {e}")
        return False

    def predict_industry(self, company_name: str, text_snippet: Optional[str] = None) -> Dict[str, Any]:
        """
        Dự đoán ngành nghề từ tên công ty và đoạn trích xuất trên danh thiếp.
        """
        if not company_name and not text_snippet:
            return {
                "industry": "Khác",
                "industryCode": "OTHER",
                "confidence": 0.0,
                "explanation": "Thiếu thông tin tên công ty để phân loại",
            }

        comp_clean = (company_name or "").strip().lower()
        with self._lock:
            # Ưu tiên khớp chính xác với ngân hàng mẫu đã biết
            for s_text, s_code in self.samples:
                if s_text.lower() == comp_clean:
                    return {
                        "industry": INDUSTRY_CATALOG.get(s_code, {}).get("name_vi", s_code),
                        "industryCode": s_code,
                        "confidence": 1.0,
                        "explanation": f"Khớp chính xác với công ty đã xác nhận: {s_text}",
                    }

            if not self.is_trained:
                return {
                    "industry": "Khác",
                    "industryCode": "OTHER",
                    "confidence": 0.0,
                    "explanation": "Mô hình kNN chưa sẵn sàng",
                }

            combined_text = f"{company_name or ''} {text_snippet or ''}".strip()
            X = self.vectorizer.transform([combined_text])
            probs = self.knn.predict_proba(X)[0]
            classes = self.knn.classes_

        max_idx = int(np.argmax(probs))
        pred_code = classes[max_idx]
        conf = float(probs[max_idx])

        # Chuẩn hóa nếu confidence thấp
        if conf < 0.25:
            pred_code = "OTHER"
            name_vi = "Khác"
        else:
            name_vi = INDUSTRY_CATALOG.get(pred_code, {}).get("name_vi", pred_code)

        return {
            "industry": name_vi,
            "industryCode": pred_code,
            "confidence": round(conf, 4),
            "explanation": INDUSTRY_CATALOG.get(pred_code, {}).get("description"),
        }

    def add_sample(self, company_name: str, industry_code: str, persist: bool = True) -> bool:
        """
        Bổ sung ngay 1 mẫu công ty đã duyệt vào kNN memory.
        """
        if not company_name or not industry_code or industry_code not in INDUSTRY_CATALOG:
            return False

        with self._lock:
            self.samples.append((company_name.strip(), industry_code.strip()))
            self._fit()
            if persist:
                self._save()
        return True


industry_classifier = IndustryClassifier()

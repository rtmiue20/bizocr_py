import os
import json
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)
RAW_DATA_DIR = "data/raw/"


class ClusteringService:
    def cluster_unknown_documents(self, n_clusters: int = 3):
        """Gom cụm các chứng từ chưa xác định (UNKNOWN) bằng K-Means"""
        unknown_docs = []
        doc_ids = []

        # 1. Thu thập các document chưa được dán nhãn
        if os.path.exists(RAW_DATA_DIR):
            for filename in os.listdir(RAW_DATA_DIR):
                if filename.endswith(".json"):
                    filepath = os.path.join(RAW_DATA_DIR, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if data.get("label", "UNKNOWN") == "UNKNOWN":
                                text = data.get("raw_content", "").strip()
                                if text:
                                    unknown_docs.append(text)
                                    doc_ids.append(filename)
                    except Exception as e:
                        logger.error(f"Lỗi đọc file clustering {filepath}: {e}")

        if len(unknown_docs) < n_clusters:
            return {"status": "warning",
                    "message": f"Chưa đủ tài liệu UNKNOWN để chia thành {n_clusters} cụm. (Có {len(unknown_docs)} mẫu)"}

        try:
            # 2. Xử lý NLP (TF-IDF)
            vectorizer = TfidfVectorizer(max_features=1000, lowercase=True)
            X = vectorizer.fit_transform(unknown_docs)

            # 3. Áp dụng K-Means Clustering (Học không giám sát)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(X)

            # 4. Phân nhóm kết quả
            clusters = {f"Cluster_{i}": [] for i in range(n_clusters)}
            for doc_id, label in zip(doc_ids, kmeans.labels_):
                clusters[f"Cluster_{int(label)}"].append(doc_id)

            return {
                "status": "success",
                "message": "Đã gom cụm thành công các chứng từ lạ",
                "n_clusters": n_clusters,
                "clusters": clusters
            }
        except Exception as e:
            logger.error(f"Lỗi K-Means Clustering: {e}")
            return {"status": "error", "message": str(e)}


clustering_service = ClusteringService()
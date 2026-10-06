import logging
import re
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def normalize_search_text(text: Optional[str]) -> str:
    """Chuẩn hóa chuỗi văn bản phục vụ tìm kiếm ngữ nghĩa."""
    if not text:
        return ""
    # Giữ lại các chữ cái tiếng Việt, dấu cách, số
    cleaned = re.sub(r"[^\w\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


class EmbeddingService:
    """
    Dịch vụ tìm kiếm ngữ nghĩa (Semantic Search Service):
    Sử dụng bộ biểu diễn vector đa đặc trưng (Word + Character n-grams + TF-IDF)
    cho phép tìm kiếm mờ và tìm kiếm ngữ nghĩa tự nhiên tiếng Việt không bị lệch từ khóa.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            analyzer="word",
            sublinear_tf=True,
            min_df=1,
        )
        self.char_vectorizer = TfidfVectorizer(
            ngram_range=(2, 4),
            analyzer="char_wb",
            sublinear_tf=True,
            min_df=1,
        )

    def encode_text(self, text: str) -> str:
        return normalize_search_text(text)

    def semantic_search(
        self,
        query: str,
        candidate_cards: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm các danh thiếp phù hợp nhất với câu truy vấn ngữ nghĩa.
        Ví dụ query: 'công ty xây dựng', 'logistics chuyển phát', 'in ấn quảng cáo', 'nha khoa'...
        """
        if not query or not query.strip() or not candidate_cards:
            return []

        norm_query = normalize_search_text(query)

        # Xây dựng văn bản đại diện cho từng danh thiếp
        card_corpus: List[str] = []
        for card in candidate_cards:
            parts = [
                card.get("company", ""),
                card.get("name", ""),
                card.get("title", ""),
                card.get("industry", ""),
                card.get("address", ""),
                card.get("rawText", ""),
            ]
            card_repr = normalize_search_text(" ".join(filter(None, parts)))
            card_corpus.append(card_repr)

        all_texts = [norm_query] + card_corpus

        try:
            # 1. Tính toán điểm tương đồng theo từ (Word n-grams)
            word_vec = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", sublinear_tf=True)
            X_word = word_vec.fit_transform(all_texts)
            word_sims = cosine_similarity(X_word[0:1], X_word[1:])[0]

            # 2. Tính toán điểm tương đồng theo cụm ký tự (Char n-grams) để bắt lỗi gõ mờ/từ ghép
            char_vec = TfidfVectorizer(ngram_range=(2, 4), analyzer="char_wb", sublinear_tf=True)
            X_char = char_vec.fit_transform(all_texts)
            char_sims = cosine_similarity(X_char[0:1], X_char[1:])[0]

            # Trọng số kết hợp: 60% từ + 40% ký tự
            combined_scores = 0.6 * word_sims + 0.4 * char_sims

            # Đóng gói và sắp xếp kết quả
            ranked_results = []
            for idx, score in enumerate(combined_scores):
                if score > 0.05:  # Ngưỡng tương quan tối thiểu
                    card = candidate_cards[idx]
                    card_id = str(card.get("id") or card.get("cardId") or f"card_{idx}")
                    ranked_results.append({
                        "cardId": card_id,
                        "score": round(float(score), 4),
                        "company": card.get("company"),
                        "name": card.get("name"),
                        "industry": card.get("industry"),
                        "matchedSnippet": card_corpus[idx][:150] + ("..." if len(card_corpus[idx]) > 150 else ""),
                    })

            ranked_results.sort(key=lambda x: x["score"], reverse=True)
            return ranked_results[:top_k]

        except Exception as e:
            logger.error(f"Lỗi khi thực hiện semantic search: {e}", exc_info=True)
            return []


embedding_service = EmbeddingService()

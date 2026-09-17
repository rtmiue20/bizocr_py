"""
api/routes_search.py

Endpoint tra cứu thông tin doanh nghiệp (MST, mã ngành) bằng DuckDuckGo,
thay thế cho Google Custom Search.

Gọi bởi: com.company.bizocr.service.ExternalTaxSearchService (Java)
    POST /api/tax-search
    body: {"companyName": "...", "domain": "..."}   # domain optional
"""

import logging
import re
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("python_service.search.duckduckgo_search")

router = APIRouter(prefix="/api", tags=["tax-search"])

# --- Cấu hình tìm kiếm --------------------------------------------------

TRUSTED_DOMAINS = [
    "masothue.com",
    "hosocongty.vn",
    "thongtindoanhnghiep.co",
    "tratencongty.com",
    "vietnambiz.vn",
]

MAX_RESULTS_PER_QUERY = 8
MAX_INDUSTRY_CANDIDATES = 5

TAX_CODE_PATTERN = re.compile(r"\b\d{10}(?:-\d{3})?\b")
INDUSTRY_KEYWORD_PATTERN = re.compile(
    r"(ngành\s*nghề|mã\s*ngành|nganh\s*nghe|ma\s*nganh)", re.IGNORECASE
)
INDUSTRY_CODE_PATTERN = re.compile(r"\b(\d{4})\b")


# --- DTO -----------------------------------------------------------------

class TaxSearchRequest(BaseModel):
    companyName: str = ""
    domain: Optional[str] = None


class TaxSearchResponse(BaseModel):
    taxCode: Optional[str] = None
    industryCodes: List[str] = []
    source: Optional[str] = None
    matchedUrl: Optional[str] = None


# --- Endpoint --------------------------------------------------------------

@router.post("/tax-search", response_model=TaxSearchResponse)
def tax_search(payload: TaxSearchRequest) -> TaxSearchResponse:
    company_name = (payload.companyName or "").strip()
    domain = (payload.domain or "").strip() or None

    if not company_name and not domain:
        logger.warning("[TaxSearch] Bỏ qua tìm kiếm vì companyName và domain đều rỗng")
        return TaxSearchResponse()

    # Làm sạch tên công ty: Bỏ dấu gạch ngang '-' và khoảng trắng thừa để tránh lỗi toán tử DDG
    clean_company = re.sub(r"\s*-\s*", " ", company_name)
    clean_company = re.sub(r"\s+", " ", clean_company).strip()

    logger.info(
        "[TaxSearch] Đang tra cứu DuckDuckGo: (gốc='%s', sạch='%s', domain=%s)",
        company_name,
        clean_company,
        domain,
    )

    try:
        results = _run_search(clean_company, domain)
    except Exception as error:  # DDGS lỗi mạng/rate-limit -> graceful fallback
        logger.error("[TaxSearch] Lỗi gọi DuckDuckGo: %s", error, exc_info=True)
        return TaxSearchResponse()

    if not results:
        logger.warning("[TaxSearch] DuckDuckGo không trả về kết quả nào")
        return TaxSearchResponse()

    # Ưu tiên trang uy tín
    ranked = sorted(results, key=lambda r: _trust_rank(r.get("href", "")))
    matched_url = ranked[0].get("href") if ranked else None

    combined_text = "\n".join(
        f"{r.get('title', '')} {r.get('body', '')}" for r in ranked
    )

    tax_code = _extract_tax_code(combined_text)
    industry_codes = _extract_industry_codes(combined_text)

    if tax_code:
        logger.info("[TaxSearch] Tìm được MST ứng viên: %s", tax_code)
    else:
        logger.warning("[TaxSearch] Không tìm thấy MST hợp lệ trong kết quả")

    return TaxSearchResponse(
        taxCode=tax_code,
        industryCodes=industry_codes,
        source="duckduckgo",
        matchedUrl=matched_url,
    )


# --- Helpers -----------------------------------------------------------

def _run_search(company_name: str, domain: Optional[str]) -> List[dict]:
    from ddgs import DDGS  # Đổi sang tên package mới

    query_parts = [company_name, "mã số thuế"]
    if domain:
        query_parts.append(domain)
    query = " ".join(part for part in query_parts if part)

    results = []
    try:
        with DDGS() as ddgs:
            # 1. Thử tìm kiếm mở rộng (bỏ bớt ràng buộc region cứng)
            results = list(ddgs.text(query, max_results=MAX_RESULTS_PER_QUERY))

            # 2. Fallback nếu không có kết quả: Ép ngoặc kép tên công ty
            if not results:
                query_quoted = f'"{company_name}" mã số thuế'
                results = list(ddgs.text(query_quoted, max_results=MAX_RESULTS_PER_QUERY))
    except Exception as e:
        logger.error("[TaxSearch] Lỗi truy vấn DDGS: %s", e)

    return results


def _trust_rank(url: str) -> int:
    url = (url or "").lower()
    for i, trusted in enumerate(TRUSTED_DOMAINS):
        if trusted in url:
            return i
    return len(TRUSTED_DOMAINS)


def _extract_tax_code(text: str) -> Optional[str]:
    match = TAX_CODE_PATTERN.search(text)
    return match.group() if match else None


def _extract_industry_codes(text: str) -> List[str]:
    candidates: List[str] = []
    for keyword_match in INDUSTRY_KEYWORD_PATTERN.finditer(text):
        window = text[keyword_match.end(): keyword_match.end() + 200]
        for code_match in INDUSTRY_CODE_PATTERN.finditer(window):
            code = code_match.group(1)
            if code not in candidates:
                candidates.append(code)
            if len(candidates) >= MAX_INDUSTRY_CANDIDATES:
                break
        if len(candidates) >= MAX_INDUSTRY_CANDIDATES:
            break
    return candidates
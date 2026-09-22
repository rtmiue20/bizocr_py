import logging
import re
from typing import Dict, Any, Optional, List
from python_service.api.routes_search import tax_search, TaxSearchRequest, TaxSearchResponse

logger = logging.getLogger(__name__)

class CompanyResolver:
    """
    Hợp nhất logic xử lý công ty từ nhiều nguồn:
    1. Kết quả OCR (đã qua Heuristics & Qwen Normalizer)
    2. DuckDuckGo Tax Search (Tra cứu thông tin MST, mã ngành)
    3. (Optional) Database local - Hiện tại đang pass qua để Java xử lý
    """

    @staticmethod
    async def resolve_company(
        company_name: str,
        tax_code: Optional[str] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Quy trình hợp nhất:
        - Nếu có MST -> Coi như đã xác định.
        - Nếu chỉ có tên -> Gọi Tax Search để tìm MST và mã ngành.
        - Trả về payload hợp nhất để Java Service thực hiện upsertCompany.
        """
        logger.info(f"[CompanyResolver] Resolving: {company_name}, tax_code={tax_code}, domain={domain}")
        
        resolved_tax_code = tax_code
        industry_codes = []
        source = "OCR"
        matched_url = None

        # Nếu thiếu MST, thử tra cứu
        if not resolved_tax_code and company_name:
            try:
                search_req = TaxSearchRequest(companyName=company_name, domain=domain)
                search_res: TaxSearchResponse = tax_search(search_req)
                
                if search_res.taxCode:
                    resolved_tax_code = search_res.taxCode
                    industry_codes = search_res.industryCodes
                    source = "duckduckgo"
                    matched_url = search_res.matchedUrl
                    logger.info(f"[CompanyResolver] Found tax code via search: {resolved_tax_code}")
            except Exception as e:
                logger.error(f"[CompanyResolver] Search failed: {e}")

        return {
            "name": company_name,
            "taxCode": resolved_tax_code,
            "industryCodes": industry_codes,
            "source": source,
            "matchedUrl": matched_url,
            "resolved": bool(resolved_tax_code)
        }

company_resolver = CompanyResolver()

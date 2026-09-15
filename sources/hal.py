from typing import Optional

import requests

from sources.base import DocumentMetadata, DocumentSource, RawDocument

HAL_SEARCH_URL = "https://api.archives-ouvertes.fr/search/"
HAL_FIELDS = "docid,title_s,publicationDate_s,uri_s,domain_s,files_s"
DEFAULT_LIMIT = 100
MAX_ROWS = 10000


class HALSource(DocumentSource):
    def fetch(self, **filters) -> list[RawDocument]:
        domain = filters.get("domain")
        date_start = filters.get("date_start")
        date_end = filters.get("date_end")
        only_with_pdf = filters.get("only_with_pdf", False)
        limit = filters.get("limit", DEFAULT_LIMIT) or MAX_ROWS

        query_params = self._build_query_params(domain, date_start, date_end, only_with_pdf, limit)
        response = requests.get(HAL_SEARCH_URL, params=query_params, timeout=30)
        response.raise_for_status()

        documents = response.json()["response"]["docs"]
        return [self._to_raw_document(document) for document in documents]

    def _build_query_params(
        self,
        domain: Optional[str],
        date_start: Optional[str],
        date_end: Optional[str],
        only_with_pdf: bool,
        limit: int,
    ) -> dict:
        filter_queries = []

        if domain:
            filter_queries.append(f"domain_s:{domain}")

        if date_start or date_end:
            start = date_start or "*"
            end = date_end or "*"
            filter_queries.append(f"producedDate_s:[{start} TO {end}]")

        if only_with_pdf:
            filter_queries.append("submitType_s:file")

        return {
            "q": "*:*",
            "fq": filter_queries,
            "fl": HAL_FIELDS,
            "rows": limit,
            "sort": "producedDate_s desc",
            "wt": "json",
        }

    def _to_raw_document(self, document: dict) -> RawDocument:
        pdf_files = document.get("files_s") or []
        titles = document.get("title_s") or [""]
        domains = document.get("domain_s") or [None]

        return RawDocument(
            id=document["docid"],
            text=None,
            pdf_path=pdf_files[0] if pdf_files else None,
            metadata=DocumentMetadata(
                title=titles[0],
                date=document.get("publicationDate_s"),
                url=document.get("uri_s"),
                domain=domains[0],
            ),
        )

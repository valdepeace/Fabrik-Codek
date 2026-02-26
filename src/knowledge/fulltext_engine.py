"""Full-text search engine using Meilisearch.

Provides BM25-style keyword search as complement to vector (LanceDB) and graph (NetworkX).
Uses httpx directly — no additional SDK dependency required.
Fully optional: degrades gracefully when Meilisearch is unavailable.
"""

import hashlib

import httpx
import structlog

from src.config import settings

logger = structlog.get_logger()


class FullTextEngine:
    """Async full-text search via Meilisearch REST API.

    Design:
        - Uses httpx.AsyncClient for HTTP calls (already a project dependency).
        - All methods return empty results on connection failure (no exceptions).
        - Documents indexed with: id, text, source, category, project.
        - Search returns same dict format as RAGEngine for RRF compatibility.
    """

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        index: str | None = None,
    ):
        self._url = url or settings.meilisearch_url
        self._key = key or settings.meilisearch_key
        self._index = index or settings.meilisearch_index
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        self._client = httpx.AsyncClient(
            base_url=self._url,
            headers=headers,
            timeout=10.0,
        )

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Check if Meilisearch is reachable."""
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_documents(self, documents: list[dict]) -> int:
        """Add documents to the Meilisearch index.

        Each document must have: id, text, source, category.
        Returns number of documents submitted.
        """
        if not documents:
            return 0

        try:
            resp = await self._client.post(
                f"/indexes/{self._index}/documents",
                json=documents,
            )
            if resp.status_code in (200, 202):
                logger.info("fulltext_index_ok", count=len(documents))
                return len(documents)
            logger.warning("fulltext_index_error", status=resp.status_code, body=resp.text[:200])
            return 0
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning("fulltext_index_unavailable", error=str(exc))
            return 0

    async def ensure_index(self) -> bool:
        """Create the index if it doesn't exist. Configure filterable/searchable attributes."""
        try:
            resp = await self._client.post(
                "/indexes",
                json={"uid": self._index, "primaryKey": "id"},
            )
            # 409 = already exists, which is fine
            if resp.status_code not in (200, 201, 202, 409):
                logger.warning("fulltext_create_index_error", status=resp.status_code)
                return False

            await self._client.put(
                f"/indexes/{self._index}/settings",
                json={
                    "searchableAttributes": ["text", "source"],
                    "filterableAttributes": ["category", "project"],
                    "sortableAttributes": ["source"],
                },
            )
            return True
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    @staticmethod
    def make_doc_id(text: str, source: str) -> str:
        """Deterministic document ID from content + source."""
        return hashlib.md5(f"{source}:{text[:200]}".encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 5,
        category: str | None = None,
    ) -> list[dict]:
        """Full-text search. Returns results compatible with RAGEngine format.

        Returns:
            List of dicts with keys: text, source, category, score, origin.
        """
        body: dict = {
            "q": query,
            "limit": limit,
            "attributesToRetrieve": ["id", "text", "source", "category"],
        }
        if category:
            body["filter"] = f"category = '{category}'"

        try:
            resp = await self._client.post(
                f"/indexes/{self._index}/search",
                json=body,
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            results = []
            for rank, hit in enumerate(data.get("hits", [])):
                results.append(
                    {
                        "text": hit.get("text", ""),
                        "source": hit.get("source", ""),
                        "category": hit.get("category", ""),
                        "score": 1.0 / (1.0 + rank),
                        "origin": "fulltext",
                    }
                )
            return results

        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.debug("fulltext_search_unavailable", error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict:
        """Get index statistics."""
        try:
            resp = await self._client.get(f"/indexes/{self._index}/stats")
            if resp.status_code != 200:
                return {"document_count": 0, "is_indexing": False, "available": False}

            data = resp.json()
            return {
                "document_count": data.get("numberOfDocuments", 0),
                "is_indexing": data.get("isIndexing", False),
                "available": True,
            }
        except (httpx.ConnectError, httpx.TimeoutException):
            return {"document_count": 0, "is_indexing": False, "available": False}

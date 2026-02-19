# Full-Text Search with Meilisearch - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Meilisearch as a third retrieval source alongside LanceDB (vector) and NetworkX (graph), exposed via MCP tool and REST API.

**Architecture:** FullTextEngine wraps Meilisearch REST API via httpx (already a dependency). HybridRAGEngine extends RRF fusion to 3 sources. Meilisearch is fully optional — system degrades gracefully when unavailable. No new Python dependencies required.

**Tech Stack:** Meilisearch (external binary), httpx (existing dep), FastAPI (existing), FastMCP (existing)

---

## Pre-requisites

Meilisearch must be installed and running. The engine treats it like Ollama — optional external service.

```bash
# Install Meilisearch binary
curl -L https://install.meilisearch.com | sh
# Or via package manager
sudo apt install meilisearch  # if available

# Run (dev mode, no persistence required for testing)
meilisearch --master-key="fabrik-dev-key"
# Default: http://localhost:7700
```

---

### Task 1: Add Meilisearch config to Settings

**Files:**
- Modify: `src/config/settings.py`

**Step 1: Write the failing test**

File: `tests/test_fulltext_engine.py`

```python
"""Tests for FullTextEngine (Meilisearch integration)."""

import pytest

from src.config import settings


class TestFullTextConfig:
    """Verify Meilisearch settings exist with correct defaults."""

    def test_meilisearch_url_default(self):
        assert settings.meilisearch_url == "http://localhost:7700"

    def test_meilisearch_key_default(self):
        assert settings.meilisearch_key is None

    def test_meilisearch_index_default(self):
        assert settings.meilisearch_index == "fabrik_knowledge"

    def test_fulltext_weight_default(self):
        assert settings.fulltext_weight == 0.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fulltext_engine.py::TestFullTextConfig -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'meilisearch_url'`

**Step 3: Write minimal implementation**

Add to `src/config/settings.py` after the Knowledge Graph section (line ~65):

```python
    # Full-text search (Meilisearch) - optional
    meilisearch_url: str = "http://localhost:7700"
    meilisearch_key: str | None = None
    meilisearch_index: str = "fabrik_knowledge"
    fulltext_weight: float = 0.0  # 0.0 = disabled in RRF fusion
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fulltext_engine.py::TestFullTextConfig -v`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add src/config/settings.py tests/test_fulltext_engine.py
git commit -m "FEAT: Add Meilisearch config to settings"
```

---

### Task 2: Create FullTextEngine core

**Files:**
- Create: `src/knowledge/fulltext_engine.py`
- Modify: `tests/test_fulltext_engine.py`

**Step 1: Write the failing tests**

Append to `tests/test_fulltext_engine.py`:

```python
import json
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

from src.knowledge.fulltext_engine import FullTextEngine


@pytest.fixture
def engine():
    """FullTextEngine with default config, no real Meilisearch needed."""
    return FullTextEngine()


class TestFullTextEngineInit:
    """Verify engine initializes with correct defaults."""

    def test_default_url(self, engine):
        assert engine._url == "http://localhost:7700"

    def test_default_index(self, engine):
        assert engine._index == "fabrik_knowledge"

    def test_custom_config(self):
        e = FullTextEngine(url="http://meili:7700", key="secret", index="custom")
        assert e._url == "http://meili:7700"
        assert e._index == "custom"


class TestFullTextHealthCheck:
    """Verify health_check calls Meilisearch /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_ok(self, engine):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "available"}

        with patch.object(engine, "_client") as mock_client:
            mock_client.get = AsyncMock(return_value=mock_resp)
            result = await engine.health_check()
            assert result is True
            mock_client.get.assert_called_once_with("/health")

    @pytest.mark.asyncio
    async def test_health_unavailable(self, engine):
        with patch.object(engine, "_client") as mock_client:
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            result = await engine.health_check()
            assert result is False


class TestFullTextSearch:
    """Verify search calls Meilisearch search endpoint."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, engine):
        meili_response = {
            "hits": [
                {"id": "doc1", "text": "FastAPI uses Pydantic", "source": "training.jsonl",
                 "category": "training"},
                {"id": "doc2", "text": "Retry with backoff", "source": "errors.jsonl",
                 "category": "learning"},
            ],
            "estimatedTotalHits": 2,
            "processingTimeMs": 1,
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = meili_response

        with patch.object(engine, "_client") as mock_client:
            mock_client.post = AsyncMock(return_value=mock_resp)
            results = await engine.search("FastAPI", limit=5)

            assert len(results) == 2
            assert results[0]["text"] == "FastAPI uses Pydantic"
            assert results[0]["source"] == "training.jsonl"
            assert results[0]["origin"] == "fulltext"
            assert "score" in results[0]

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self, engine):
        meili_response = {
            "hits": [
                {"id": "doc1", "text": "match", "source": "s", "category": "training"},
            ],
            "estimatedTotalHits": 1,
            "processingTimeMs": 1,
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = meili_response

        with patch.object(engine, "_client") as mock_client:
            mock_client.post = AsyncMock(return_value=mock_resp)
            results = await engine.search("query", limit=5, category="training")

            # Verify filter was passed in the request body
            call_args = mock_client.post.call_args
            body = call_args[1].get("json", call_args[0][1] if len(call_args[0]) > 1 else {})
            assert "filter" in body

    @pytest.mark.asyncio
    async def test_search_connection_error_returns_empty(self, engine):
        with patch.object(engine, "_client") as mock_client:
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            results = await engine.search("query")
            assert results == []


class TestFullTextIndexing:
    """Verify document indexing calls Meilisearch documents endpoint."""

    @pytest.mark.asyncio
    async def test_index_documents(self, engine):
        # Meilisearch returns a task on document add
        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_resp.json.return_value = {"taskUid": 1, "status": "enqueued"}

        docs = [
            {"id": "1", "text": "doc one", "source": "a.jsonl", "category": "training"},
            {"id": "2", "text": "doc two", "source": "b.jsonl", "category": "learning"},
        ]

        with patch.object(engine, "_client") as mock_client:
            mock_client.post = AsyncMock(return_value=mock_resp)
            count = await engine.index_documents(docs)
            assert count == 2

    @pytest.mark.asyncio
    async def test_index_documents_empty_list(self, engine):
        count = await engine.index_documents([])
        assert count == 0


class TestFullTextStats:
    """Verify stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats(self, engine):
        meili_response = {
            "numberOfDocuments": 150,
            "isIndexing": False,
            "fieldDistribution": {"text": 150, "source": 150},
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = meili_response

        with patch.object(engine, "_client") as mock_client:
            mock_client.get = AsyncMock(return_value=mock_resp)
            stats = await engine.get_stats()

            assert stats["document_count"] == 150
            assert stats["is_indexing"] is False
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fulltext_engine.py -v -k "not TestFullTextConfig"`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.knowledge.fulltext_engine'`

**Step 3: Write minimal implementation**

Create `src/knowledge/fulltext_engine.py`:

```python
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
        headers = {"Content-Type": "application/json"}
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
            # Create index (idempotent — Meilisearch ignores if exists)
            resp = await self._client.post(
                "/indexes",
                json={"uid": self._index, "primaryKey": "id"},
            )
            if resp.status_code not in (200, 201, 202):
                # 409 = already exists, which is fine
                if resp.status_code != 409:
                    logger.warning("fulltext_create_index_error", status=resp.status_code)
                    return False

            # Configure searchable and filterable attributes
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
                results.append({
                    "text": hit.get("text", ""),
                    "source": hit.get("source", ""),
                    "category": hit.get("category", ""),
                    "score": 1.0 / (1.0 + rank),  # Rank-based score for RRF
                    "origin": "fulltext",
                })
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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fulltext_engine.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/knowledge/fulltext_engine.py tests/test_fulltext_engine.py
git commit -m "FEAT: Add FullTextEngine with Meilisearch"
```

---

### Task 3: Integrate FullTextEngine into HybridRAG (three-tier RRF)

**Files:**
- Modify: `src/knowledge/hybrid_rag.py`
- Modify: `tests/test_hybrid_rag.py`

**Step 1: Write the failing tests**

Append to `tests/test_hybrid_rag.py`:

```python
class TestThreeTierRRF:
    """Verify RRF fusion works with three sources: vector + graph + fulltext."""

    def test_rrf_with_fulltext_results(self, graph_engine):
        """Full-text results contribute to fusion score."""
        mock_rag = AsyncMock()
        engine = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
            vector_weight=0.4,
            graph_weight=0.3,
            fulltext_weight=0.3,
        )

        vector_results = [
            {"text": "FastAPI uses Pydantic for validation", "source": "a", "category": "training", "score": 0.9},
        ]
        graph_results = [
            {"text": "FastAPI uses Pydantic for validation", "source": "a", "category": "training", "score": 0.8},
        ]
        fulltext_results = [
            {"text": "FastAPI uses Pydantic for validation", "source": "a", "category": "training",
             "score": 1.0, "origin": "fulltext"},
        ]

        fused = engine._rrf_fusion(vector_results, graph_results, fulltext_results, limit=5)
        assert len(fused) == 1
        # Should be "hybrid" since it appeared in multiple sources
        assert fused[0]["origin"] == "hybrid"
        # Score should be higher than with just 2 sources
        assert fused[0]["score"] > 0

    def test_rrf_fulltext_only_results(self, graph_engine):
        """Results only in fulltext still appear."""
        mock_rag = AsyncMock()
        engine = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
            fulltext_weight=0.3,
        )

        fulltext_results = [
            {"text": "exact error match", "source": "err", "category": "learning",
             "score": 1.0, "origin": "fulltext"},
        ]

        fused = engine._rrf_fusion([], [], fulltext_results, limit=5)
        assert len(fused) == 1
        assert fused[0]["origin"] == "fulltext"

    def test_rrf_no_fulltext_weight_disables(self, graph_engine):
        """When fulltext_weight=0, fulltext results don't contribute."""
        mock_rag = AsyncMock()
        engine = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
            fulltext_weight=0.0,
        )

        fulltext_results = [
            {"text": "should not appear", "source": "x", "category": "x",
             "score": 1.0, "origin": "fulltext"},
        ]

        fused = engine._rrf_fusion([], [], fulltext_results, limit=5)
        assert len(fused) == 0


class TestHybridRetrieveWithFullText:
    """Verify retrieve() includes fulltext results when engine is available."""

    @pytest.mark.asyncio
    async def test_retrieve_with_fulltext_engine(self, graph_engine):
        mock_rag = AsyncMock()
        mock_rag.retrieve = AsyncMock(return_value=[
            {"text": "vector result", "source": "v", "category": "training", "score": 0.9},
        ])

        mock_fulltext = AsyncMock()
        mock_fulltext.search = AsyncMock(return_value=[
            {"text": "fulltext result", "source": "f", "category": "training",
             "score": 1.0, "origin": "fulltext"},
        ])

        engine = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
            fulltext_engine=mock_fulltext,
            fulltext_weight=0.3,
        )

        results = await engine.retrieve("test query", limit=5)
        # Both sources should contribute
        assert len(results) >= 1
        mock_fulltext.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_without_fulltext_engine(self, graph_engine):
        """When no fulltext engine, works as before (vector + graph only)."""
        mock_rag = AsyncMock()
        mock_rag.retrieve = AsyncMock(return_value=[
            {"text": "vector result", "source": "v", "category": "training", "score": 0.9},
        ])

        engine = HybridRAGEngine(
            rag_engine=mock_rag,
            graph_engine=graph_engine,
        )

        results = await engine.retrieve("test query", limit=5)
        assert len(results) >= 1
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_hybrid_rag.py::TestThreeTierRRF -v`
Expected: FAIL — `_rrf_fusion` doesn't accept 3 args, `fulltext_weight` not a param, etc.

**Step 3: Modify HybridRAGEngine**

Changes to `src/knowledge/hybrid_rag.py`:

1. Add `DEFAULT_FULLTEXT_WEIGHT = 0.0` to module constants.

2. Update `__init__` signature:
```python
def __init__(
    self,
    rag_engine: RAGEngine | None = None,
    graph_engine: GraphEngine | None = None,
    fulltext_engine=None,  # FullTextEngine | None, lazy import to avoid circular
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    graph_weight: float = DEFAULT_GRAPH_WEIGHT,
    fulltext_weight: float = DEFAULT_FULLTEXT_WEIGHT,
):
    self._rag = rag_engine
    self._graph = graph_engine
    self._fulltext = fulltext_engine
    self.vector_weight = vector_weight
    self.graph_weight = graph_weight
    self.fulltext_weight = fulltext_weight
    self._owns_rag = rag_engine is None
```

3. Update `retrieve()` to include fulltext:
```python
async def retrieve(self, query, limit=5, graph_depth=DEFAULT_GRAPH_DEPTH, min_weight=DEFAULT_MIN_WEIGHT):
    # ... existing vector + graph retrieval ...

    # 3. Full-text search (when available and weighted)
    fulltext_results = []
    if self._fulltext and self.fulltext_weight > 0:
        try:
            fulltext_results = await self._fulltext.search(query, limit=limit * 2)
        except Exception as exc:
            logger.debug("fulltext_retrieve_failed", error=str(exc))

    # 4. RRF fusion (three sources)
    fused = self._rrf_fusion(vector_results, graph_results, fulltext_results, limit=limit)
    # ... rest unchanged ...
```

4. Update `_rrf_fusion()` to accept three sources:
```python
def _rrf_fusion(
    self,
    vector_results: list[dict],
    graph_results: list[dict],
    fulltext_results: list[dict] | None = None,
    limit: int = 5,
) -> list[dict]:
    """Reciprocal Rank Fusion of two or three result sets."""
    k = DEFAULT_RRF_K
    scores: dict[str, float] = {}
    result_map: dict[str, dict] = {}
    sources_seen: dict[str, set[str]] = {}  # key -> set of source names

    # Score vector results
    for rank, r in enumerate(vector_results):
        key = r["text"][:100]
        scores[key] = scores.get(key, 0) + self.vector_weight / (k + rank)
        if key not in result_map:
            result_map[key] = {**r, "origin": "vector"}
            sources_seen[key] = {"vector"}
        else:
            sources_seen[key].add("vector")

    # Score graph results
    for rank, r in enumerate(graph_results):
        key = r["text"][:100]
        scores[key] = scores.get(key, 0) + self.graph_weight / (k + rank)
        if key not in result_map:
            result_map[key] = {**r, "origin": "graph"}
            sources_seen[key] = {"graph"}
        else:
            sources_seen[key].add("graph")

    # Score fulltext results
    if fulltext_results and self.fulltext_weight > 0:
        for rank, r in enumerate(fulltext_results):
            key = r["text"][:100]
            scores[key] = scores.get(key, 0) + self.fulltext_weight / (k + rank)
            if key not in result_map:
                result_map[key] = {**r, "origin": "fulltext"}
                sources_seen[key] = {"fulltext"}
            else:
                sources_seen[key].add("fulltext")

    # Mark multi-source results as "hybrid"
    for key, srcs in sources_seen.items():
        if len(srcs) > 1 and key in result_map:
            result_map[key]["origin"] = "hybrid"

    # Sort by fused score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for key, score in ranked[:limit]:
        if key in result_map:
            result = result_map[key]
            result["score"] = score
            results.append(result)

    return results
```

**Step 4: Run ALL hybrid_rag tests**

Run: `python -m pytest tests/test_hybrid_rag.py -v`
Expected: ALL PASS (existing tests + new three-tier tests)

**Step 5: Commit**

```bash
git add src/knowledge/hybrid_rag.py tests/test_hybrid_rag.py
git commit -m "FEAT: Three-tier RRF fusion (vector+graph+fulltext)"
```

---

### Task 4: Add MCP tool and API endpoint

**Files:**
- Modify: `src/interfaces/mcp_server.py`
- Modify: `src/interfaces/api.py`
- Modify: `tests/test_mcp_server.py`
- Modify: `tests/test_api.py`

**Step 1: Write the failing tests**

Append to `tests/test_mcp_server.py`:

```python
class TestFabrikFulltextSearch:
    """Test fabrik_fulltext_search MCP tool."""

    @pytest.mark.asyncio
    async def test_fulltext_search_returns_results(self):
        mock_fulltext = AsyncMock()
        mock_fulltext.search = AsyncMock(return_value=[
            {"text": "exact match result", "source": "training.jsonl",
             "category": "training", "score": 1.0, "origin": "fulltext"},
        ])

        from src.interfaces.mcp_server import _state, fabrik_fulltext_search
        _state["fulltext"] = mock_fulltext
        try:
            result = await fabrik_fulltext_search("exact error message", limit=5)
            data = json.loads(result)
            assert data["count"] == 1
            assert data["results"][0]["text"] == "exact match result"
            assert data["results"][0]["origin"] == "fulltext"
        finally:
            _state.pop("fulltext", None)

    @pytest.mark.asyncio
    async def test_fulltext_search_unavailable(self):
        from src.interfaces.mcp_server import _state, fabrik_fulltext_search
        _state["fulltext"] = None
        try:
            result = await fabrik_fulltext_search("query")
            data = json.loads(result)
            assert data["count"] == 0
            assert "error" in data
        finally:
            _state.pop("fulltext", None)
```

Append to `tests/test_api.py` (similar pattern for the REST endpoint):

```python
class TestFulltextSearchEndpoint:
    """Test POST /fulltext/search endpoint."""

    @pytest.mark.asyncio
    async def test_fulltext_search(self, client):
        # Mock the fulltext engine on app.state
        mock_ft = AsyncMock()
        mock_ft.search = AsyncMock(return_value=[
            {"text": "keyword match", "source": "s.jsonl", "category": "training",
             "score": 1.0, "origin": "fulltext"},
        ])
        client.app.state.fulltext = mock_ft

        resp = client.post("/fulltext/search", json={"query": "keyword", "limit": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_fulltext_search_unavailable(self, client):
        client.app.state.fulltext = None
        resp = client.post("/fulltext/search", json={"query": "keyword"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mcp_server.py::TestFabrikFulltextSearch -v`
Expected: FAIL — `fabrik_fulltext_search` doesn't exist

**Step 3: Add MCP tool**

Add to `src/interfaces/mcp_server.py` after `fabrik_graph_stats`:

```python
@mcp.tool(
    name="fabrik_fulltext_search",
    description=(
        "Full-text keyword search in the knowledge base via Meilisearch. "
        "Best for exact matches: error messages, function names, config keys. "
        "Complements semantic vector search (fabrik_search) for precision queries."
    ),
)
async def fabrik_fulltext_search(
    query: str, limit: int = 5, category: str | None = None,
) -> str:
    """Full-text search using Meilisearch."""
    limit = max(1, min(limit, 50))

    fulltext = _state.get("fulltext")
    if fulltext is None:
        return json.dumps({"results": [], "count": 0, "error": "Full-text engine not available"})

    try:
        results = await fulltext.search(query, limit=limit, category=category)
    except Exception as exc:
        return json.dumps({"results": [], "count": 0, "error": str(exc)})

    output = {
        "results": [
            {
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "category": r.get("category", ""),
                "score": r.get("score", 0.0),
                "origin": "fulltext",
            }
            for r in results
        ],
        "count": len(results),
    }
    return json.dumps(output)
```

Update `lifespan()` to init FullTextEngine (after graph init, before hybrid):

```python
    # 4b. Full-text engine (Meilisearch) - optional
    try:
        from src.knowledge.fulltext_engine import FullTextEngine

        ft = FullTextEngine()
        if await ft.health_check():
            await ft.ensure_index()
            _state["fulltext"] = ft
        else:
            await ft.close()
            _state["fulltext"] = None
            logger.info("mcp_fulltext_unavailable", msg="Meilisearch not running, skipping")
    except Exception as exc:
        logger.warning("mcp_fulltext_init_failed", error=str(exc))
        _state["fulltext"] = None
```

Also pass fulltext to HybridRAGEngine init:

```python
    # 5. Hybrid RAG — now with optional fulltext
    if _state.get("rag") is not None:
        try:
            from src.knowledge.hybrid_rag import HybridRAGEngine

            hybrid = HybridRAGEngine(
                rag_engine=_state["rag"],
                graph_engine=_state["graph"],
                fulltext_engine=_state.get("fulltext"),
                fulltext_weight=settings.fulltext_weight,
            )
            hybrid._owns_rag = False
            _state["hybrid"] = hybrid
        except Exception as exc:
            logger.warning("mcp_hybrid_init_failed", error=str(exc))
            _state["hybrid"] = None
```

Update shutdown to close fulltext:

```python
    # Shutdown
    if _state.get("fulltext"):
        await _state["fulltext"].close()
    if _state.get("rag"):
        await _state["rag"].close()
```

**Step 4: Add API endpoint**

Add to `src/interfaces/api.py`:

Request/response models (after GraphStatsResponse):
```python
class FulltextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    category: str | None = None
    limit: int = Field(default=5, ge=1, le=50)


class FulltextSearchResult(BaseModel):
    text: str
    source: str
    category: str
    score: float
    origin: str = "fulltext"


class FulltextSearchResponse(BaseModel):
    results: list[FulltextSearchResult]
    count: int
```

Endpoint (after `/graph/stats`):
```python
@app.post("/fulltext/search", response_model=FulltextSearchResponse)
async def fulltext_search(req: FulltextSearchRequest, request: Request):
    """Full-text keyword search via Meilisearch."""
    state = request.app.state
    if getattr(state, "fulltext", None) is None:
        return FulltextSearchResponse(results=[], count=0)

    results = await state.fulltext.search(req.query, limit=req.limit, category=req.category)

    return FulltextSearchResponse(
        results=[
            FulltextSearchResult(
                text=r["text"],
                source=r.get("source", ""),
                category=r.get("category", ""),
                score=r.get("score", 0.0),
            )
            for r in results
        ],
        count=len(results),
    )
```

Update API lifespan with same pattern as MCP (init fulltext, pass to hybrid, close on shutdown).

Update `StatusResponse` to include `fulltext: str` field. Update `/status` endpoint accordingly.

**Step 5: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/interfaces/mcp_server.py src/interfaces/api.py tests/test_mcp_server.py tests/test_api.py
git commit -m "FEAT: Add fulltext search MCP tool and API endpoint"
```

---

### Task 5: Add CLI command for fulltext indexing

**Files:**
- Modify: `src/interfaces/cli.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
class TestFulltextCLI:
    """Test fabrik fulltext commands."""

    def test_fulltext_status_shows_unavailable(self, runner):
        result = runner.invoke(app, ["fulltext", "status"])
        assert result.exit_code == 0
        # When Meilisearch not running, should indicate unavailable
        assert "unavailable" in result.stdout.lower() or "status" in result.stdout.lower()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py::TestFulltextCLI -v`
Expected: FAIL — no "fulltext" command

**Step 3: Add CLI commands**

Add to `src/interfaces/cli.py`:

```python
# Fulltext subcommand group
fulltext_app = typer.Typer(help="Full-text search (Meilisearch)")
app.add_typer(fulltext_app, name="fulltext")


@fulltext_app.command("status")
def fulltext_status():
    """Check Meilisearch connection and index stats."""
    import asyncio
    from src.knowledge.fulltext_engine import FullTextEngine

    async def _check():
        async with FullTextEngine() as ft:
            healthy = await ft.health_check()
            if not healthy:
                console.print("[yellow]Meilisearch:[/yellow] unavailable")
                console.print(f"  URL: {ft._url}")
                console.print("  Run: meilisearch --master-key=fabrik-dev-key")
                return
            stats = await ft.get_stats()
            console.print("[green]Meilisearch:[/green] connected")
            console.print(f"  Documents: {stats['document_count']}")
            console.print(f"  Indexing: {stats['is_indexing']}")

    asyncio.run(_check())


@fulltext_app.command("index")
def fulltext_index():
    """Index datalake documents into Meilisearch."""
    import asyncio
    from src.knowledge.fulltext_engine import FullTextEngine

    async def _index():
        async with FullTextEngine() as ft:
            if not await ft.health_check():
                console.print("[red]Error:[/red] Meilisearch not available")
                raise typer.Exit(1)

            await ft.ensure_index()
            console.print("[blue]Indexing datalake into Meilisearch...[/blue]")

            # Index training pairs
            import json as json_mod
            from src.config import settings

            tp_dir = settings.datalake_path / "02-processed" / "training-pairs"
            total = 0
            if tp_dir.exists():
                for f in sorted(tp_dir.glob("*.jsonl")):
                    docs = []
                    for line in f.read_text().splitlines():
                        if not line.strip():
                            continue
                        try:
                            record = json_mod.loads(line)
                            text = record.get("output", record.get("text", ""))
                            instruction = record.get("instruction", "")
                            if text and len(text) >= 50:
                                docs.append({
                                    "id": FullTextEngine.make_doc_id(text, str(f.name)),
                                    "text": f"{instruction}\n{text}" if instruction else text,
                                    "source": f.name,
                                    "category": "training",
                                    "project": record.get("project", ""),
                                })
                        except json_mod.JSONDecodeError:
                            continue
                    if docs:
                        count = await ft.index_documents(docs)
                        total += count

            console.print(f"[green]Indexed {total} documents[/green]")

    asyncio.run(_index())


@fulltext_app.command("search")
def fulltext_search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-n", help="Max results"),
):
    """Search the full-text index."""
    import asyncio
    from src.knowledge.fulltext_engine import FullTextEngine

    async def _search():
        async with FullTextEngine() as ft:
            if not await ft.health_check():
                console.print("[red]Error:[/red] Meilisearch not available")
                raise typer.Exit(1)

            results = await ft.search(query, limit=limit)
            if not results:
                console.print("[yellow]No results[/yellow]")
                return

            for i, r in enumerate(results, 1):
                console.print(f"\n[bold]{i}.[/bold] [{r['category']}] {r['source']}")
                console.print(f"   {r['text'][:200]}")

    asyncio.run(_search())
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_cli.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/interfaces/cli.py tests/test_cli.py
git commit -m "FEAT: Add fulltext CLI commands (status, index, search)"
```

---

### Task 6: Run full test suite and verify backward compatibility

**Files:** None — verification only.

**Step 1: Run entire test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ALL 472+ tests PASS (existing tests unchanged + ~15 new tests)

**Step 2: Verify no regressions**

Check that:
- `HybridRAGEngine` with no fulltext_engine works exactly as before (fulltext_weight=0.0)
- MCP server starts without Meilisearch (graceful degradation)
- API server starts without Meilisearch
- Existing `_rrf_fusion` behavior unchanged when fulltext_results is empty

**Step 3: Final commit**

If any fixes needed, commit them. Then:

```bash
git log --oneline feat/mcp-server..HEAD
```

Verify all commits are clean and follow convention.

---

## Summary

| Task | What | LOC ~estimate |
|------|------|---------------|
| 1 | Config settings | 5 new + 15 test |
| 2 | FullTextEngine core | 150 new + 100 test |
| 3 | Three-tier RRF fusion | 40 modified + 80 test |
| 4 | MCP tool + API endpoint | 80 new + 40 test |
| 5 | CLI commands | 80 new + 10 test |
| 6 | Verification | 0 |
| **Total** | | **~600 LOC** |

**Zero new Python dependencies.** httpx already in project.

**Backward compatible.** fulltext_weight defaults to 0.0 — everything works exactly as before until user explicitly enables it.

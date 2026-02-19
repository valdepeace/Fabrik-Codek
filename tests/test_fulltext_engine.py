"""Tests for FullTextEngine (Meilisearch integration)."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.config import settings
from src.knowledge.fulltext_engine import FullTextEngine


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Engine init
# ---------------------------------------------------------------------------


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

    def test_auth_header_set_when_key_provided(self):
        e = FullTextEngine(key="my-key")
        assert e._client.headers.get("Authorization") == "Bearer my-key"

    def test_no_auth_header_without_key(self, engine):
        assert "Authorization" not in engine._client.headers


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestFullTextHealthCheck:
    """Verify health_check calls Meilisearch /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_ok(self, engine):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "available"}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "get", AsyncMock(return_value=mock_resp))
            result = await engine.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_unavailable(self, engine):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "get", AsyncMock(side_effect=httpx.ConnectError("refused")))
            result = await engine.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_timeout(self, engine):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "get", AsyncMock(side_effect=httpx.TimeoutException("timeout")))
            result = await engine.health_check()
            assert result is False


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


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

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "post", AsyncMock(return_value=mock_resp))
            results = await engine.search("FastAPI", limit=5)

            assert len(results) == 2
            assert results[0]["text"] == "FastAPI uses Pydantic"
            assert results[0]["source"] == "training.jsonl"
            assert results[0]["origin"] == "fulltext"
            assert results[0]["score"] == 1.0  # rank 0 → 1/(1+0)
            assert results[1]["score"] == 0.5  # rank 1 → 1/(1+1)

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self, engine):
        meili_response = {
            "hits": [{"id": "d1", "text": "match", "source": "s", "category": "training"}],
            "estimatedTotalHits": 1,
            "processingTimeMs": 1,
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = meili_response

        mock_post = AsyncMock(return_value=mock_resp)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "post", mock_post)
            await engine.search("query", limit=5, category="training")

            # Verify filter was passed in the request body
            call_kwargs = mock_post.call_args[1]
            body = call_kwargs.get("json", {})
            assert "filter" in body
            assert "training" in body["filter"]

    @pytest.mark.asyncio
    async def test_search_connection_error_returns_empty(self, engine):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "post", AsyncMock(side_effect=httpx.ConnectError("refused")))
            results = await engine.search("query")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_non_200_returns_empty(self, engine):
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "post", AsyncMock(return_value=mock_resp))
            results = await engine.search("query")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_empty_hits(self, engine):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"hits": [], "estimatedTotalHits": 0, "processingTimeMs": 0}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "post", AsyncMock(return_value=mock_resp))
            results = await engine.search("nonexistent")
            assert results == []


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class TestFullTextIndexing:
    """Verify document indexing calls Meilisearch documents endpoint."""

    @pytest.mark.asyncio
    async def test_index_documents(self, engine):
        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_resp.json.return_value = {"taskUid": 1, "status": "enqueued"}

        docs = [
            {"id": "1", "text": "doc one", "source": "a.jsonl", "category": "training"},
            {"id": "2", "text": "doc two", "source": "b.jsonl", "category": "learning"},
        ]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "post", AsyncMock(return_value=mock_resp))
            count = await engine.index_documents(docs)
            assert count == 2

    @pytest.mark.asyncio
    async def test_index_documents_empty_list(self, engine):
        count = await engine.index_documents([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_index_documents_connection_error(self, engine):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "post", AsyncMock(side_effect=httpx.ConnectError("refused")))
            count = await engine.index_documents([{"id": "1", "text": "t", "source": "s", "category": "c"}])
            assert count == 0

    @pytest.mark.asyncio
    async def test_index_documents_server_error(self, engine):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "post", AsyncMock(return_value=mock_resp))
            count = await engine.index_documents([{"id": "1", "text": "t", "source": "s", "category": "c"}])
            assert count == 0


# ---------------------------------------------------------------------------
# Ensure index
# ---------------------------------------------------------------------------


class TestFullTextEnsureIndex:
    """Verify index creation and configuration."""

    @pytest.mark.asyncio
    async def test_ensure_index_creates_new(self, engine):
        mock_post = MagicMock()
        mock_post.status_code = 202
        mock_put = MagicMock()
        mock_put.status_code = 202

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "post", AsyncMock(return_value=mock_post))
            mp.setattr(engine._client, "put", AsyncMock(return_value=mock_put))
            result = await engine.ensure_index()
            assert result is True

    @pytest.mark.asyncio
    async def test_ensure_index_already_exists(self, engine):
        mock_post = MagicMock()
        mock_post.status_code = 409  # Already exists
        mock_put = MagicMock()
        mock_put.status_code = 202

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "post", AsyncMock(return_value=mock_post))
            mp.setattr(engine._client, "put", AsyncMock(return_value=mock_put))
            result = await engine.ensure_index()
            assert result is True

    @pytest.mark.asyncio
    async def test_ensure_index_connection_error(self, engine):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "post", AsyncMock(side_effect=httpx.ConnectError("refused")))
            result = await engine.ensure_index()
            assert result is False


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


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

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "get", AsyncMock(return_value=mock_resp))
            stats = await engine.get_stats()

            assert stats["document_count"] == 150
            assert stats["is_indexing"] is False
            assert stats["available"] is True

    @pytest.mark.asyncio
    async def test_get_stats_unavailable(self, engine):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(engine._client, "get", AsyncMock(side_effect=httpx.ConnectError("refused")))
            stats = await engine.get_stats()

            assert stats["document_count"] == 0
            assert stats["available"] is False


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


class TestMakeDocId:
    """Verify deterministic document ID generation."""

    def test_same_input_same_id(self):
        id1 = FullTextEngine.make_doc_id("some text", "source.jsonl")
        id2 = FullTextEngine.make_doc_id("some text", "source.jsonl")
        assert id1 == id2

    def test_different_input_different_id(self):
        id1 = FullTextEngine.make_doc_id("text a", "source.jsonl")
        id2 = FullTextEngine.make_doc_id("text b", "source.jsonl")
        assert id1 != id2

    def test_id_length(self):
        doc_id = FullTextEngine.make_doc_id("text", "source")
        assert len(doc_id) == 12

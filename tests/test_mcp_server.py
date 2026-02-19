"""Tests for the Fabrik-Codek MCP Server."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src import __version__
from src.core.llm_client import LLMResponse
from src.knowledge.graph_schema import Entity, EntityType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_response(content="test answer", model="test-model", tokens=42, latency=100.0):
    return LLMResponse(content=content, model=model, tokens_used=tokens, latency_ms=latency)


def _make_entity(id="python_tech", name="Python", entity_type=EntityType.TECHNOLOGY,
                 mention_count=5, aliases=None):
    return Entity(
        id=id,
        name=name,
        entity_type=entity_type,
        mention_count=mention_count,
        aliases=aliases or ["py"],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    client = AsyncMock()
    client.health_check = AsyncMock(return_value=True)
    client.generate = AsyncMock(return_value=_make_llm_response())
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mock_rag():
    rag = AsyncMock()
    rag.retrieve = AsyncMock(return_value=[
        {"text": "some doc", "source": "file.jsonl", "category": "testing", "score": 0.9},
    ])
    rag.close = AsyncMock()
    return rag


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.load.return_value = True
    graph.search_entities.return_value = []
    graph.get_neighbors.return_value = []
    graph.get_stats.return_value = {
        "entity_count": 10,
        "edge_count": 25,
        "connected_components": 3,
        "entity_types": {"technology": 5, "concept": 3, "tool": 2},
        "relation_types": {"uses": 10, "depends_on": 8, "part_of": 7},
    }
    return graph


@pytest.fixture
def mock_hybrid():
    hybrid = AsyncMock()
    hybrid.retrieve = AsyncMock(return_value=[
        {"text": "hybrid doc", "source": "h.jsonl", "category": "agents", "score": 0.95,
         "origin": "vector"},
    ])
    hybrid._owns_rag = False
    return hybrid


@pytest.fixture(autouse=True)
def inject_state(mock_llm, mock_rag, mock_graph, mock_hybrid):
    """Inject mock state into the MCP server module for every test."""
    from src.interfaces import mcp_server

    mcp_server._state.clear()
    mcp_server._state["llm"] = mock_llm
    mcp_server._state["ollama_ok"] = True
    mcp_server._state["rag"] = mock_rag
    mcp_server._state["graph"] = mock_graph
    mcp_server._state["hybrid"] = mock_hybrid
    mcp_server._state["fulltext"] = None  # Default: not available
    yield
    mcp_server._state.clear()


# ---------------------------------------------------------------------------
# TestFabrikStatus
# ---------------------------------------------------------------------------


class TestFabrikStatus:
    @pytest.mark.asyncio
    async def test_all_healthy(self, mock_llm):
        from src.interfaces.mcp_server import fabrik_status

        with patch("src.interfaces.mcp_server.settings") as ms:
            ms.datalake_path = MagicMock()
            ms.datalake_path.exists.return_value = True
            ms.default_model = "test-model"

            result = json.loads(await fabrik_status())

        assert result["ollama"] == "ok"
        assert result["rag"] == "ok"
        assert result["graph"] == "ok"
        assert result["datalake"] == "ok"
        assert result["model"] == "test-model"
        assert result["version"] == __version__
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_ollama_down(self, mock_llm):
        from src.interfaces.mcp_server import fabrik_status

        mock_llm.health_check = AsyncMock(return_value=False)

        with patch("src.interfaces.mcp_server.settings") as ms:
            ms.datalake_path = MagicMock()
            ms.datalake_path.exists.return_value = True
            ms.default_model = "test-model"

            result = json.loads(await fabrik_status())

        assert result["ollama"] == "unavailable"

    @pytest.mark.asyncio
    async def test_no_rag(self, mock_llm):
        from src.interfaces import mcp_server
        from src.interfaces.mcp_server import fabrik_status

        mcp_server._state["rag"] = None

        with patch("src.interfaces.mcp_server.settings") as ms:
            ms.datalake_path = MagicMock()
            ms.datalake_path.exists.return_value = True
            ms.default_model = "test-model"

            result = json.loads(await fabrik_status())

        assert result["rag"] == "unavailable"

    @pytest.mark.asyncio
    async def test_no_graph(self, mock_llm):
        from src.interfaces import mcp_server
        from src.interfaces.mcp_server import fabrik_status

        mcp_server._state["graph"] = None

        with patch("src.interfaces.mcp_server.settings") as ms:
            ms.datalake_path = MagicMock()
            ms.datalake_path.exists.return_value = True
            ms.default_model = "test-model"

            result = json.loads(await fabrik_status())

        assert result["graph"] == "unavailable"

    @pytest.mark.asyncio
    async def test_datalake_missing(self, mock_llm):
        from src.interfaces.mcp_server import fabrik_status

        with patch("src.interfaces.mcp_server.settings") as ms:
            ms.datalake_path = MagicMock()
            ms.datalake_path.exists.return_value = False
            ms.default_model = "test-model"

            result = json.loads(await fabrik_status())

        assert result["datalake"] == "unavailable"


# ---------------------------------------------------------------------------
# TestFabrikSearch
# ---------------------------------------------------------------------------


class TestFabrikSearch:
    @pytest.mark.asyncio
    async def test_basic_search(self, mock_rag):
        from src.interfaces.mcp_server import fabrik_search

        result = json.loads(await fabrik_search(query="testing"))

        assert result["count"] == 1
        assert result["results"][0]["category"] == "testing"
        assert result["results"][0]["score"] == 0.9
        mock_rag.retrieve.assert_called_once_with("testing", limit=5)

    @pytest.mark.asyncio
    async def test_with_category_filter(self, mock_rag):
        from src.interfaces.mcp_server import fabrik_search

        result = json.loads(await fabrik_search(query="test", category="nonexistent"))

        assert result["count"] == 0
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_custom_limit(self, mock_rag):
        from src.interfaces.mcp_server import fabrik_search

        await fabrik_search(query="test", limit=20)

        mock_rag.retrieve.assert_called_once_with("test", limit=20)

    @pytest.mark.asyncio
    async def test_limit_clamped(self, mock_rag):
        from src.interfaces.mcp_server import fabrik_search

        await fabrik_search(query="test", limit=100)

        mock_rag.retrieve.assert_called_once_with("test", limit=50)

    @pytest.mark.asyncio
    async def test_rag_unavailable(self):
        from src.interfaces import mcp_server
        from src.interfaces.mcp_server import fabrik_search

        mcp_server._state["rag"] = None

        result = json.loads(await fabrik_search(query="anything"))

        assert result["count"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_rag_exception(self, mock_rag):
        from src.interfaces.mcp_server import fabrik_search

        mock_rag.retrieve = AsyncMock(side_effect=RuntimeError("DB connection lost"))

        result = json.loads(await fabrik_search(query="test"))

        assert result["count"] == 0
        assert "error" in result


# ---------------------------------------------------------------------------
# TestFabrikGraphSearch
# ---------------------------------------------------------------------------


class TestFabrikGraphSearch:
    @pytest.mark.asyncio
    async def test_finds_entities(self, mock_graph):
        from src.interfaces.mcp_server import fabrik_graph_search

        entity = _make_entity()
        mock_graph.search_entities.return_value = [entity]

        result = json.loads(await fabrik_graph_search(query="python"))

        assert result["count"] == 1
        assert result["entities"][0]["name"] == "Python"
        assert result["entities"][0]["entity_type"] == "technology"
        assert result["entities"][0]["aliases"] == ["py"]

    @pytest.mark.asyncio
    async def test_no_results(self, mock_graph):
        from src.interfaces.mcp_server import fabrik_graph_search

        result = json.loads(await fabrik_graph_search(query="nonexistent"))

        assert result["count"] == 0
        assert result["entities"] == []

    @pytest.mark.asyncio
    async def test_graph_unavailable(self):
        from src.interfaces import mcp_server
        from src.interfaces.mcp_server import fabrik_graph_search

        mcp_server._state["graph"] = None

        result = json.loads(await fabrik_graph_search(query="anything"))

        assert result["count"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_with_neighbors(self, mock_graph):
        from src.interfaces.mcp_server import fabrik_graph_search

        entity = _make_entity()
        neighbor = _make_entity(id="fastapi_tech", name="FastAPI")
        mock_graph.search_entities.return_value = [entity]
        mock_graph.get_neighbors.return_value = [(neighbor, 0.8)]

        result = json.loads(await fabrik_graph_search(query="python"))

        assert result["entities"][0]["neighbors"] == ["FastAPI"]

    @pytest.mark.asyncio
    async def test_depth_and_limit_clamped(self, mock_graph):
        from src.interfaces.mcp_server import fabrik_graph_search

        mock_graph.search_entities.return_value = []

        await fabrik_graph_search(query="test", limit=100, depth=10)

        mock_graph.search_entities.assert_called_once_with("test", limit=50)


# ---------------------------------------------------------------------------
# TestFabrikAsk
# ---------------------------------------------------------------------------


class TestFabrikAsk:
    @pytest.mark.asyncio
    async def test_basic_ask(self, mock_llm):
        from src.interfaces.mcp_server import fabrik_ask

        result = json.loads(await fabrik_ask(prompt="hello"))

        assert result["answer"] == "test answer"
        assert result["model"] == "test-model"
        assert result["tokens_used"] == 42
        assert result["sources"] == []
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_rag(self, mock_llm, mock_rag):
        from src.interfaces.mcp_server import fabrik_ask

        result = json.loads(await fabrik_ask(prompt="how to test", use_rag=True))

        assert result["answer"] == "test answer"
        mock_rag.retrieve.assert_called_once()
        # The prompt sent to LLM should include context
        prompt_sent = mock_llm.generate.call_args.args[0]
        assert "context" in prompt_sent.lower()
        assert len(result["sources"]) > 0

    @pytest.mark.asyncio
    async def test_with_graph(self, mock_llm, mock_hybrid):
        from src.interfaces.mcp_server import fabrik_ask

        result = json.loads(await fabrik_ask(prompt="explain DDD", use_graph=True))

        assert result["answer"] == "test answer"
        mock_hybrid.retrieve.assert_called_once()
        assert len(result["sources"]) > 0
        assert result["sources"][0]["origin"] == "vector"

    @pytest.mark.asyncio
    async def test_ollama_down(self, mock_llm):
        from src.interfaces.mcp_server import fabrik_ask

        mock_llm.health_check = AsyncMock(return_value=False)

        result = json.loads(await fabrik_ask(prompt="hello"))

        assert "error" in result
        assert "Ollama" in result["error"]

    @pytest.mark.asyncio
    async def test_no_llm(self):
        from src.interfaces import mcp_server
        from src.interfaces.mcp_server import fabrik_ask

        mcp_server._state["llm"] = None

        result = json.loads(await fabrik_ask(prompt="hello"))

        assert "error" in result

    @pytest.mark.asyncio
    async def test_model_override(self, mock_llm):
        from src.interfaces.mcp_server import fabrik_ask

        await fabrik_ask(prompt="hi", model="custom-model")

        assert mock_llm.generate.call_args.kwargs.get("model") == "custom-model"

    @pytest.mark.asyncio
    async def test_graph_depth_clamped(self, mock_llm, mock_hybrid):
        from src.interfaces.mcp_server import fabrik_ask

        await fabrik_ask(prompt="test", use_graph=True, graph_depth=10)

        call_kwargs = mock_hybrid.retrieve.call_args.kwargs
        assert call_kwargs["graph_depth"] == 5

    @pytest.mark.asyncio
    async def test_rag_fallback_without_engine(self, mock_llm):
        """When use_rag=True but RAG is None, should still answer without context."""
        from src.interfaces import mcp_server
        from src.interfaces.mcp_server import fabrik_ask

        mcp_server._state["rag"] = None

        result = json.loads(await fabrik_ask(prompt="hello", use_rag=True))

        assert result["answer"] == "test answer"
        assert result["sources"] == []


# ---------------------------------------------------------------------------
# TestFabrikGraphStats
# ---------------------------------------------------------------------------


class TestFabrikGraphStats:
    @pytest.mark.asyncio
    async def test_with_data(self, mock_graph):
        from src.interfaces.mcp_server import fabrik_graph_stats

        result = json.loads(await fabrik_graph_stats())

        assert result["entity_count"] == 10
        assert result["edge_count"] == 25
        assert result["connected_components"] == 3
        assert "technology" in result["entity_types"]
        assert "uses" in result["relation_types"]

    @pytest.mark.asyncio
    async def test_no_graph(self):
        from src.interfaces import mcp_server
        from src.interfaces.mcp_server import fabrik_graph_stats

        mcp_server._state["graph"] = None

        result = json.loads(await fabrik_graph_stats())

        assert result["entity_count"] == 0
        assert result["edge_count"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_graph_exception(self, mock_graph):
        from src.interfaces.mcp_server import fabrik_graph_stats

        mock_graph.get_stats.side_effect = RuntimeError("graph corrupted")

        result = json.loads(await fabrik_graph_stats())

        assert result["entity_count"] == 0
        assert "error" in result


# ---------------------------------------------------------------------------
# TestResources
# ---------------------------------------------------------------------------


class TestResources:
    @pytest.mark.asyncio
    async def test_status_resource(self, mock_llm):
        from src.interfaces.mcp_server import status_resource

        with patch("src.interfaces.mcp_server.settings") as ms:
            ms.datalake_path = MagicMock()
            ms.datalake_path.exists.return_value = True
            ms.default_model = "test-model"

            result = json.loads(await status_resource())

        assert result["ollama"] == "ok"
        assert result["version"] == __version__

    @pytest.mark.asyncio
    async def test_graph_stats_resource(self, mock_graph):
        from src.interfaces.mcp_server import graph_stats_resource

        result = json.loads(await graph_stats_resource())

        assert result["entity_count"] == 10

    @pytest.mark.asyncio
    async def test_config_resource(self):
        from src.interfaces.mcp_server import config_resource

        with patch("src.interfaces.mcp_server.settings") as ms:
            ms.default_model = "test-model"
            ms.fallback_model = "fallback-model"
            ms.embedding_model = "embed-model"
            ms.ollama_host = "http://localhost:11434"
            ms.vector_db = "lancedb"
            ms.flywheel_enabled = True
            ms.datalake_path = "/tmp/datalake"
            ms.api_port = 8420
            ms.mcp_port = 8421

            result = json.loads(await config_resource())

        assert result["default_model"] == "test-model"
        assert result["vector_db"] == "lancedb"
        assert result["version"] == __version__
        # Ensure no secrets leak
        assert "api_key" not in result


# ---------------------------------------------------------------------------
# TestOutputFormat
# ---------------------------------------------------------------------------


class TestOutputFormat:
    """All tools must return valid JSON strings."""

    @pytest.mark.asyncio
    async def test_status_returns_json_string(self):
        from src.interfaces.mcp_server import fabrik_status

        with patch("src.interfaces.mcp_server.settings") as ms:
            ms.datalake_path = MagicMock()
            ms.datalake_path.exists.return_value = True
            ms.default_model = "m"

            result = await fabrik_status()

        assert isinstance(result, str)
        json.loads(result)  # Should not raise

    @pytest.mark.asyncio
    async def test_search_returns_json_string(self):
        from src.interfaces.mcp_server import fabrik_search

        result = await fabrik_search(query="test")

        assert isinstance(result, str)
        json.loads(result)

    @pytest.mark.asyncio
    async def test_graph_search_returns_json_string(self):
        from src.interfaces.mcp_server import fabrik_graph_search

        result = await fabrik_graph_search(query="test")

        assert isinstance(result, str)
        json.loads(result)

    @pytest.mark.asyncio
    async def test_ask_returns_json_string(self):
        from src.interfaces.mcp_server import fabrik_ask

        result = await fabrik_ask(prompt="hello")

        assert isinstance(result, str)
        json.loads(result)

    @pytest.mark.asyncio
    async def test_graph_stats_returns_json_string(self):
        from src.interfaces.mcp_server import fabrik_graph_stats

        result = await fabrik_graph_stats()

        assert isinstance(result, str)
        json.loads(result)

    @pytest.mark.asyncio
    async def test_fulltext_search_returns_json_string(self):
        from src.interfaces.mcp_server import _state, fabrik_fulltext_search

        mock_ft = AsyncMock()
        mock_ft.search = AsyncMock(return_value=[
            {"text": "match", "source": "s", "category": "c", "score": 1.0, "origin": "fulltext"},
        ])
        _state["fulltext"] = mock_ft
        result = await fabrik_fulltext_search(query="test")

        assert isinstance(result, str)
        json.loads(result)


# ---------------------------------------------------------------------------
# TestFabrikFulltextSearch
# ---------------------------------------------------------------------------


class TestFabrikFulltextSearch:
    """Test fabrik_fulltext_search MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_results(self):
        from src.interfaces.mcp_server import _state, fabrik_fulltext_search

        mock_ft = AsyncMock()
        mock_ft.search = AsyncMock(return_value=[
            {"text": "exact match result", "source": "training.jsonl",
             "category": "training", "score": 1.0, "origin": "fulltext"},
        ])
        _state["fulltext"] = mock_ft

        result = await fabrik_fulltext_search("exact error message", limit=5)
        data = json.loads(result)
        assert data["count"] == 1
        assert data["results"][0]["text"] == "exact match result"
        assert data["results"][0]["origin"] == "fulltext"

    @pytest.mark.asyncio
    async def test_unavailable_returns_error(self):
        from src.interfaces.mcp_server import _state, fabrik_fulltext_search

        _state["fulltext"] = None

        result = await fabrik_fulltext_search("query")
        data = json.loads(result)
        assert data["count"] == 0
        assert "error" in data

    @pytest.mark.asyncio
    async def test_clamps_limit(self):
        from src.interfaces.mcp_server import _state, fabrik_fulltext_search

        mock_ft = AsyncMock()
        mock_ft.search = AsyncMock(return_value=[])
        _state["fulltext"] = mock_ft

        await fabrik_fulltext_search("query", limit=100)
        mock_ft.search.assert_called_once_with("query", limit=50, category=None)

    @pytest.mark.asyncio
    async def test_with_category(self):
        from src.interfaces.mcp_server import _state, fabrik_fulltext_search

        mock_ft = AsyncMock()
        mock_ft.search = AsyncMock(return_value=[])
        _state["fulltext"] = mock_ft

        await fabrik_fulltext_search("query", category="training")
        mock_ft.search.assert_called_once_with("query", limit=5, category="training")

    @pytest.mark.asyncio
    async def test_handles_exception(self):
        from src.interfaces.mcp_server import _state, fabrik_fulltext_search

        mock_ft = AsyncMock()
        mock_ft.search = AsyncMock(side_effect=Exception("connection lost"))
        _state["fulltext"] = mock_ft

        result = await fabrik_fulltext_search("query")
        data = json.loads(result)
        assert data["count"] == 0
        assert "error" in data


# ---------------------------------------------------------------------------
# TestFabrikStatusIncludesFulltext
# ---------------------------------------------------------------------------


class TestFabrikStatusIncludesFulltext:
    """Verify status tool reports fulltext availability."""

    @pytest.mark.asyncio
    async def test_status_shows_fulltext_unavailable(self):
        from src.interfaces.mcp_server import _state, fabrik_status

        _state["fulltext"] = None
        result = json.loads(await fabrik_status())
        assert result["fulltext"] == "unavailable"

    @pytest.mark.asyncio
    async def test_status_shows_fulltext_ok(self):
        from src.interfaces.mcp_server import _state, fabrik_status

        _state["fulltext"] = AsyncMock()  # Not None = available
        result = json.loads(await fabrik_status())
        assert result["fulltext"] == "ok"

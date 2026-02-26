"""Tests for Competence Model dataclasses and scoring functions."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.competence_model import (
    COMPETENT_THRESHOLD,
    ENTRY_CEILING,
    EXPERT_THRESHOLD,
    MAX_EDGES_CEILING,
    NOVICE_THRESHOLD,
    WEIGHTS_ALL,
    WEIGHTS_ALL_NO_OUTCOME,
    WEIGHTS_ENTRY_ONLY,
    WEIGHTS_ENTRY_ONLY_NO_OUTCOME,
    WEIGHTS_NO_GRAPH,
    WEIGHTS_NO_GRAPH_NO_OUTCOME,
    WEIGHTS_NO_RECENCY,
    WEIGHTS_NO_RECENCY_NO_OUTCOME,
    CompetenceBuilder,
    CompetenceEntry,
    CompetenceMap,
    _classify_level,
    _competence_cache,
    compute_competence_score,
    compute_entity_density,
    compute_entry_score,
    compute_recency_weight,
    get_active_competence_map,
    load_competence_map,
    save_competence_map,
)


class TestClassifyLevel:
    """Test threshold-based level classification."""

    def test_expert_above_threshold(self):
        assert _classify_level(0.95) == "Expert"

    def test_expert_at_exact_threshold(self):
        assert _classify_level(EXPERT_THRESHOLD) == "Expert"

    def test_competent_below_expert(self):
        assert _classify_level(0.79) == "Competent"

    def test_competent_at_exact_threshold(self):
        assert _classify_level(COMPETENT_THRESHOLD) == "Competent"

    def test_novice_below_competent(self):
        assert _classify_level(0.39) == "Novice"

    def test_novice_at_exact_threshold(self):
        assert _classify_level(NOVICE_THRESHOLD) == "Novice"

    def test_unknown_below_novice(self):
        assert _classify_level(0.09) == "Unknown"

    def test_unknown_at_zero(self):
        assert _classify_level(0.0) == "Unknown"

    def test_expert_at_max(self):
        assert _classify_level(1.0) == "Expert"


class TestCompetenceEntry:
    """Test CompetenceEntry dataclass."""

    def test_defaults(self):
        entry = CompetenceEntry(topic="python")
        assert entry.topic == "python"
        assert entry.score == 0.0
        assert entry.level == "Unknown"
        assert entry.entries == 0
        assert entry.entity_density == 0.0
        assert entry.recency_weight == 0.0
        assert entry.last_activity == ""

    def test_to_dict_roundtrip(self):
        entry = CompetenceEntry(
            topic="postgresql",
            score=0.85,
            level="Expert",
            entries=42,
            entity_density=0.73,
            recency_weight=0.91,
            last_activity="2026-02-20T10:00:00",
        )
        d = entry.to_dict()
        restored = CompetenceEntry.from_dict(d)
        assert restored.topic == entry.topic
        assert restored.score == entry.score
        assert restored.level == entry.level
        assert restored.entries == entry.entries
        assert restored.entity_density == entry.entity_density
        assert restored.recency_weight == entry.recency_weight
        assert restored.last_activity == entry.last_activity

    def test_score_rounding_in_to_dict(self):
        entry = CompetenceEntry(
            topic="docker",
            score=0.123456789,
            entity_density=0.987654321,
            recency_weight=0.111111111,
        )
        d = entry.to_dict()
        assert d["score"] == 0.1235
        assert d["entity_density"] == 0.9877
        assert d["recency_weight"] == 0.1111

    def test_from_dict_with_missing_optional_fields(self):
        data = {"topic": "terraform"}
        entry = CompetenceEntry.from_dict(data)
        assert entry.topic == "terraform"
        assert entry.score == 0.0
        assert entry.level == "Unknown"
        assert entry.entries == 0

    def test_from_dict_preserves_all_fields(self):
        data = {
            "topic": "fastapi",
            "score": 0.65,
            "level": "Competent",
            "entries": 15,
            "entity_density": 0.55,
            "recency_weight": 0.8,
            "last_activity": "2026-01-15T12:00:00",
        }
        entry = CompetenceEntry.from_dict(data)
        assert entry.topic == "fastapi"
        assert entry.score == 0.65
        assert entry.level == "Competent"
        assert entry.entries == 15
        assert entry.entity_density == 0.55
        assert entry.recency_weight == 0.8
        assert entry.last_activity == "2026-01-15T12:00:00"


class TestCompetenceMap:
    """Test CompetenceMap dataclass."""

    def test_empty_map_defaults(self):
        cmap = CompetenceMap()
        assert cmap.topics == []
        assert cmap.total_topics == 0
        assert cmap.built_at != ""

    def test_empty_map_get_level_returns_unknown(self):
        cmap = CompetenceMap()
        assert cmap.get_level("anything") == "Unknown"

    def test_empty_map_get_score_returns_zero(self):
        cmap = CompetenceMap()
        assert cmap.get_score("anything") == 0.0

    def test_empty_map_get_entry_returns_none(self):
        cmap = CompetenceMap()
        assert cmap.get_entry("anything") is None

    def test_get_level_lookup(self):
        cmap = CompetenceMap(
            topics=[
                CompetenceEntry(topic="python", score=0.9, level="Expert"),
                CompetenceEntry(topic="rust", score=0.3, level="Novice"),
            ]
        )
        assert cmap.get_level("python") == "Expert"
        assert cmap.get_level("rust") == "Novice"
        assert cmap.get_level("go") == "Unknown"

    def test_get_score_lookup(self):
        cmap = CompetenceMap(
            topics=[
                CompetenceEntry(topic="docker", score=0.55, level="Competent"),
            ]
        )
        assert cmap.get_score("docker") == 0.55
        assert cmap.get_score("kubernetes") == 0.0

    def test_get_entry_returns_correct_entry(self):
        entry = CompetenceEntry(topic="terraform", score=0.42, level="Competent")
        cmap = CompetenceMap(topics=[entry])
        result = cmap.get_entry("terraform")
        assert result is not None
        assert result.topic == "terraform"
        assert result.score == 0.42

    def test_experts_filter(self):
        cmap = CompetenceMap(
            topics=[
                CompetenceEntry(topic="python", score=0.95, level="Expert"),
                CompetenceEntry(topic="docker", score=0.5, level="Competent"),
                CompetenceEntry(topic="postgresql", score=0.85, level="Expert"),
                CompetenceEntry(topic="rust", score=0.2, level="Novice"),
            ]
        )
        experts = cmap.experts()
        assert len(experts) == 2
        expert_topics = {e.topic for e in experts}
        assert expert_topics == {"python", "postgresql"}

    def test_experts_empty_when_no_experts(self):
        cmap = CompetenceMap(
            topics=[
                CompetenceEntry(topic="go", score=0.5, level="Competent"),
                CompetenceEntry(topic="rust", score=0.2, level="Novice"),
            ]
        )
        assert cmap.experts() == []

    def test_serialization_roundtrip(self):
        cmap = CompetenceMap(
            topics=[
                CompetenceEntry(
                    topic="python",
                    score=0.92,
                    level="Expert",
                    entries=100,
                    entity_density=0.75,
                    recency_weight=0.95,
                    last_activity="2026-02-20",
                ),
                CompetenceEntry(
                    topic="docker",
                    score=0.45,
                    level="Competent",
                    entries=20,
                ),
            ],
            built_at="2026-02-20T10:00:00",
            total_topics=2,
        )
        d = cmap.to_dict()
        restored = CompetenceMap.from_dict(d)
        assert restored.built_at == cmap.built_at
        assert restored.total_topics == cmap.total_topics
        assert len(restored.topics) == 2
        assert restored.topics[0].topic == "python"
        assert restored.topics[0].score == 0.92
        assert restored.topics[1].topic == "docker"
        assert restored.topics[1].level == "Competent"

    def test_from_dict_with_empty_data(self):
        cmap = CompetenceMap.from_dict({})
        assert cmap.topics == []
        assert cmap.total_topics == 0

    def test_system_prompt_fragment_with_experts_and_competent(self):
        cmap = CompetenceMap(
            topics=[
                CompetenceEntry(topic="python", level="Expert"),
                CompetenceEntry(topic="postgresql", level="Expert"),
                CompetenceEntry(topic="docker", level="Competent"),
                CompetenceEntry(topic="terraform", level="Competent"),
                CompetenceEntry(topic="rust", level="Novice"),
            ]
        )
        fragment = cmap.to_system_prompt_fragment()
        assert "Expert in: python, postgresql" in fragment
        assert "Competent in: docker, terraform" in fragment
        assert "rust" not in fragment
        assert fragment.endswith(".")

    def test_system_prompt_fragment_experts_only(self):
        cmap = CompetenceMap(
            topics=[
                CompetenceEntry(topic="python", level="Expert"),
                CompetenceEntry(topic="rust", level="Novice"),
            ]
        )
        fragment = cmap.to_system_prompt_fragment()
        assert "Expert in: python" in fragment
        assert "Competent" not in fragment
        assert fragment.endswith(".")

    def test_system_prompt_fragment_competent_only(self):
        cmap = CompetenceMap(
            topics=[
                CompetenceEntry(topic="docker", level="Competent"),
                CompetenceEntry(topic="go", level="Novice"),
            ]
        )
        fragment = cmap.to_system_prompt_fragment()
        assert "Competent in: docker" in fragment
        assert "Expert" not in fragment
        assert fragment.endswith(".")

    def test_system_prompt_fragment_empty_when_no_expert_or_competent(self):
        cmap = CompetenceMap(
            topics=[
                CompetenceEntry(topic="rust", level="Novice"),
                CompetenceEntry(topic="go", level="Unknown"),
            ]
        )
        assert cmap.to_system_prompt_fragment() == ""

    def test_system_prompt_fragment_empty_map(self):
        cmap = CompetenceMap()
        assert cmap.to_system_prompt_fragment() == ""

    def test_system_prompt_fragment_max_five_each(self):
        experts = [CompetenceEntry(topic=f"expert_{i}", level="Expert") for i in range(8)]
        competents = [CompetenceEntry(topic=f"comp_{i}", level="Competent") for i in range(8)]
        cmap = CompetenceMap(topics=experts + competents)
        fragment = cmap.to_system_prompt_fragment()
        # Count topics listed after "Expert in:"
        expert_part = fragment.split("Expert in: ")[1].split(".")[0]
        expert_listed = [t.strip() for t in expert_part.split(",")]
        assert len(expert_listed) == 5
        # Count topics listed after "Competent in:"
        competent_part = fragment.split("Competent in: ")[1].split(".")[0]
        competent_listed = [t.strip() for t in competent_part.split(",")]
        assert len(competent_listed) == 5


# ---------------------------------------------------------------------------
# Scoring function tests
# ---------------------------------------------------------------------------


class TestEntryScore:
    """Test compute_entry_score logarithmic scaling."""

    def test_zero_entries_returns_zero(self):
        assert compute_entry_score(0) == 0.0

    def test_negative_entries_returns_zero(self):
        assert compute_entry_score(-5) == 0.0

    def test_one_entry(self):
        score = compute_entry_score(1)
        assert 0.10 <= score <= 0.20  # ~0.15

    def test_ten_entries(self):
        score = compute_entry_score(10)
        assert 0.48 <= score <= 0.56  # ~0.52

    def test_ceiling_entries_returns_one(self):
        assert compute_entry_score(ENTRY_CEILING) == 1.0

    def test_above_ceiling_capped(self):
        assert compute_entry_score(500) == 1.0

    def test_monotonically_increasing(self):
        scores = [compute_entry_score(n) for n in [1, 5, 10, 25, 50, 100]]
        for i in range(1, len(scores)):
            assert scores[i] > scores[i - 1], (
                f"score({[1,5,10,25,50,100][i]})={scores[i]} "
                f"should be > score({[1,5,10,25,50,100][i-1]})={scores[i-1]}"
            )


class TestEntityDensity:
    """Test compute_entity_density linear scaling."""

    def test_zero_edges_returns_zero(self):
        assert compute_entity_density(0) == 0.0

    def test_negative_edges_returns_zero(self):
        assert compute_entity_density(-10) == 0.0

    def test_half_ceiling(self):
        assert compute_entity_density(50) == 0.5

    def test_at_ceiling(self):
        assert compute_entity_density(MAX_EDGES_CEILING) == 1.0

    def test_above_ceiling_capped(self):
        assert compute_entity_density(200) == 1.0

    def test_custom_ceiling(self):
        assert compute_entity_density(25, max_ceiling=50) == 0.5


class TestRecencyWeight:
    """Test compute_recency_weight exponential decay."""

    def test_empty_string_returns_zero(self):
        assert compute_recency_weight("") == 0.0

    def test_whitespace_only_returns_zero(self):
        assert compute_recency_weight("   ") == 0.0

    def test_invalid_timestamp_returns_zero(self):
        assert compute_recency_weight("not-a-date") == 0.0

    def test_now_returns_approximately_one(self):
        now = datetime.now(tz=UTC)
        score = compute_recency_weight(now.isoformat(), reference_time=now)
        assert score >= 0.99

    def test_one_half_life_returns_half(self):
        now = datetime.now(tz=UTC)
        past = now - timedelta(days=30)
        score = compute_recency_weight(past.isoformat(), reference_time=now)
        assert 0.49 <= score <= 0.51  # ~0.5

    def test_two_half_lives_returns_quarter(self):
        now = datetime.now(tz=UTC)
        past = now - timedelta(days=60)
        score = compute_recency_weight(past.isoformat(), reference_time=now)
        assert 0.24 <= score <= 0.26  # ~0.25

    def test_future_timestamp_returns_one(self):
        now = datetime.now(tz=UTC)
        future = now + timedelta(days=10)
        score = compute_recency_weight(future.isoformat(), reference_time=now)
        assert score == 1.0

    def test_naive_vs_aware_compatible(self):
        """Naive last_activity and aware reference_time should work."""
        ref = datetime(2026, 2, 20, 12, 0, 0, tzinfo=UTC)
        naive_iso = "2026-02-20T12:00:00"
        score = compute_recency_weight(naive_iso, reference_time=ref)
        assert score >= 0.99

    def test_aware_vs_naive_compatible(self):
        """Aware last_activity and naive reference_time should work."""
        ref = datetime(2026, 2, 20, 12, 0, 0)
        aware_iso = "2026-02-20T12:00:00+00:00"
        score = compute_recency_weight(aware_iso, reference_time=ref)
        assert score >= 0.99


class TestComputeCompetenceScore:
    """Test the composite compute_competence_score."""

    def _ref(self) -> datetime:
        return datetime(2026, 2, 20, 12, 0, 0, tzinfo=UTC)

    def test_all_signals(self):
        """With all signals and no outcome, uses WEIGHTS_ALL_NO_OUTCOME."""
        ref = self._ref()
        last = ref.isoformat()
        final, entry_s, density_s, recency_s = compute_competence_score(
            entries=100,
            edge_count=100,
            last_activity_iso=last,
            reference_time=ref,
        )
        assert entry_s == 1.0
        assert density_s == 1.0
        assert recency_s >= 0.99
        expected = (
            WEIGHTS_ALL_NO_OUTCOME["entry"] * entry_s
            + WEIGHTS_ALL_NO_OUTCOME["density"] * density_s
            + WEIGHTS_ALL_NO_OUTCOME["recency"] * recency_s
        )
        assert abs(final - expected) < 0.01

    def test_no_graph(self):
        """edge_count=None -> WEIGHTS_NO_GRAPH_NO_OUTCOME, density_s=0.0."""
        ref = self._ref()
        last = ref.isoformat()
        final, entry_s, density_s, recency_s = compute_competence_score(
            entries=50,
            edge_count=None,
            last_activity_iso=last,
            reference_time=ref,
        )
        assert density_s == 0.0
        assert recency_s >= 0.99
        expected = (
            WEIGHTS_NO_GRAPH_NO_OUTCOME["entry"] * entry_s
            + WEIGHTS_NO_GRAPH_NO_OUTCOME["recency"] * recency_s
        )
        assert abs(final - expected) < 0.01

    def test_no_recency(self):
        """Empty last_activity -> WEIGHTS_NO_RECENCY_NO_OUTCOME, recency_s=0.0."""
        ref = self._ref()
        final, entry_s, density_s, recency_s = compute_competence_score(
            entries=50,
            edge_count=50,
            last_activity_iso="",
            reference_time=ref,
        )
        assert recency_s == 0.0
        assert density_s == 0.5
        expected = (
            WEIGHTS_NO_RECENCY_NO_OUTCOME["entry"] * entry_s
            + WEIGHTS_NO_RECENCY_NO_OUTCOME["density"] * density_s
        )
        assert abs(final - expected) < 0.01

    def test_entry_only_capped_at_weight(self):
        """No graph + no recency -> WEIGHTS_ENTRY_ONLY_NO_OUTCOME, max possible = 0.8."""
        ref = self._ref()
        final, entry_s, density_s, recency_s = compute_competence_score(
            entries=500,
            edge_count=None,
            last_activity_iso="",
            reference_time=ref,
        )
        assert entry_s == 1.0
        assert density_s == 0.0
        assert recency_s == 0.0
        assert abs(final - WEIGHTS_ENTRY_ONLY_NO_OUTCOME["entry"]) < 0.001  # 0.8

    def test_all_zeros_returns_zero(self):
        """Zero entries, no graph, no recency -> 0.0."""
        ref = self._ref()
        final, entry_s, density_s, recency_s = compute_competence_score(
            entries=0,
            edge_count=None,
            last_activity_iso="",
            reference_time=ref,
        )
        assert final == 0.0
        assert entry_s == 0.0
        assert density_s == 0.0
        assert recency_s == 0.0

    def test_partial_signals_reasonable_range(self):
        """Mid-range inputs should yield a score between 0 and 1."""
        ref = self._ref()
        past = (ref - timedelta(days=15)).isoformat()
        final, _, _, _ = compute_competence_score(
            entries=25,
            edge_count=30,
            last_activity_iso=past,
            reference_time=ref,
        )
        assert 0.0 < final < 1.0


# ---------------------------------------------------------------------------
# CompetenceBuilder gathering tests
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Helper: write a list of dicts as standard JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


@pytest.fixture()
def tmp_datalake(tmp_path: Path) -> Path:
    """Create a temporary datalake with training pairs and auto-captures."""
    dl = tmp_path / "datalake"

    # -- training pairs --
    tp_dir = dl / "02-processed" / "training-pairs"

    _write_jsonl(
        tp_dir / "postgresql_basics.jsonl",
        [{"input": f"q{i}", "output": f"a{i}", "category": "postgresql"} for i in range(20)],
    )
    _write_jsonl(
        tp_dir / "debugging_tips.jsonl",
        [{"input": f"q{i}", "output": f"a{i}", "category": "debugging"} for i in range(10)],
    )
    _write_jsonl(
        tp_dir / "angular_patterns.jsonl",
        [{"input": f"q{i}", "output": f"a{i}", "category": "angular"} for i in range(5)],
    )
    # META_CATEGORIES entry — should be filtered out by _get_entry_counts
    _write_jsonl(
        tp_dir / "2026-01-10_general.jsonl",
        [{"input": f"q{i}", "output": f"a{i}", "category": "general"} for i in range(8)],
    )

    # -- auto-captures --
    ac_dir = dl / "01-raw" / "code-changes"

    _write_jsonl(
        ac_dir / "2026-02-15_auto-captures.jsonl",
        [
            {
                "file_modified": "src/db/queries.sql",
                "project": "myproject",
                "timestamp": "2026-02-15T10:30:00",
                "tags": ["postgresql"],
            },
            {
                "file_modified": "src/app/component.tsx",
                "project": "frontend",
                "timestamp": "2026-02-14T08:00:00",
                "tags": [],
            },
            {
                "file_modified": "infra/main.tf",
                "project": "infra",
                "timestamp": "2026-02-13T09:00:00",
                "tags": ["terraform"],
            },
            {
                "file_modified": "src/utils/helper.py",
                "project": "myproject",
                "timestamp": "2026-02-15T12:00:00",
                "tags": ["python"],
            },
        ],
    )

    return dl


class TestCompetenceBuilderGathering:
    """Test CompetenceBuilder private data-gathering methods."""

    def test_get_entry_counts_returns_correct_counts(self, tmp_datalake: Path):
        builder = CompetenceBuilder(tmp_datalake)
        counts = builder._get_entry_counts()

        assert counts["postgresql"] == 20
        assert counts["debugging"] == 10
        assert counts["angular"] == 5

    def test_get_entry_counts_filters_meta_categories(self, tmp_datalake: Path):
        builder = CompetenceBuilder(tmp_datalake)
        counts = builder._get_entry_counts()

        # "general" is in META_CATEGORIES, must be absent
        assert "general" not in counts

    def test_get_topic_edge_counts_without_graph(self, tmp_datalake: Path):
        builder = CompetenceBuilder(tmp_datalake, graph_engine=None)
        result = builder._get_topic_edge_counts(["postgresql", "debugging"])
        assert result == {}

    def test_get_topic_edge_counts_with_mock_graph(self, tmp_datalake: Path):
        """Mock GraphEngine with 3 entities, 2 real edges + 1 self-loop."""
        graph = MagicMock()

        # Simulated entities
        pg_entity = MagicMock()
        pg_entity.id = "ent_pg"
        debug_entity = MagicMock()
        debug_entity.id = "ent_debug"

        def find_entity(name: str):
            if name == "postgresql":
                return pg_entity
            if name == "debugging":
                return debug_entity
            return None

        graph.find_entity_by_name.side_effect = find_entity

        # Relations for postgresql: 2 real edges + 1 self-loop
        rel_real_1 = MagicMock()
        rel_real_1.source_id = "ent_pg"
        rel_real_1.target_id = "ent_other"

        rel_real_2 = MagicMock()
        rel_real_2.source_id = "ent_other2"
        rel_real_2.target_id = "ent_pg"

        rel_self_loop = MagicMock()
        rel_self_loop.source_id = "ent_pg"
        rel_self_loop.target_id = "ent_pg"

        # Relations for debugging: 1 real edge
        rel_debug = MagicMock()
        rel_debug.source_id = "ent_debug"
        rel_debug.target_id = "ent_something"

        def get_relations(entity_id: str, direction: str = "both"):
            if entity_id == "ent_pg":
                return [rel_real_1, rel_real_2, rel_self_loop]
            if entity_id == "ent_debug":
                return [rel_debug]
            return []

        graph.get_relations.side_effect = get_relations

        builder = CompetenceBuilder(tmp_datalake, graph_engine=graph)
        result = builder._get_topic_edge_counts(["postgresql", "debugging", "angular"])

        # postgresql: 2 real edges (self-loop excluded)
        assert result["postgresql"] == 2
        # debugging: 1 edge
        assert result["debugging"] == 1
        # angular: not in graph, should not appear
        assert "angular" not in result

    def test_get_topic_recency_from_auto_captures(self, tmp_datalake: Path):
        builder = CompetenceBuilder(tmp_datalake)
        topics = ["postgresql", "angular", "terraform", "python"]
        recency = builder._get_topic_recency(topics)

        # postgresql matched by tag in auto-captures
        assert recency["postgresql"] == "2026-02-15T10:30:00"
        # angular matched by .tsx extension
        assert recency["angular"] == "2026-02-14T08:00:00"
        # terraform matched by tag
        assert recency["terraform"] == "2026-02-13T09:00:00"
        # python matched by tag
        assert recency["python"] == "2026-02-15T12:00:00"

    def test_get_topic_recency_from_filename_date(self, tmp_datalake: Path):
        """Training-pair filename '2026-01-10_general.jsonl' has date prefix."""
        builder = CompetenceBuilder(tmp_datalake)
        # "general" appears in the filename stem "2026-01-10_general"
        recency = builder._get_topic_recency(["general"])
        assert recency["general"] == "2026-01-10T00:00:00"

    def test_map_capture_to_topics_by_tag(self):
        record = {
            "file_modified": "src/main.py",
            "project": "myproject",
            "tags": ["postgresql", "debugging"],
        }
        topics = ["postgresql", "debugging", "angular"]
        matched = CompetenceBuilder._map_capture_to_topics(record, topics)
        assert "postgresql" in matched
        assert "debugging" in matched
        assert "angular" not in matched

    def test_map_capture_to_topics_by_project(self):
        record = {
            "file_modified": "src/main.py",
            "project": "angular-app",
            "tags": [],
        }
        topics = ["angular", "postgresql"]
        matched = CompetenceBuilder._map_capture_to_topics(record, topics)
        assert "angular" in matched
        assert "postgresql" not in matched

    def test_map_capture_to_topics_by_extension(self):
        record = {
            "file_modified": "src/db/queries.sql",
            "project": "myproject",
            "tags": [],
        }
        topics = ["postgresql", "angular"]
        matched = CompetenceBuilder._map_capture_to_topics(record, topics)
        assert "postgresql" in matched
        assert "angular" not in matched

    def test_map_capture_to_topics_tsx_maps_to_angular(self):
        record = {
            "file_modified": "src/App.tsx",
            "project": "frontend",
            "tags": [],
        }
        topics = ["angular", "typescript"]
        matched = CompetenceBuilder._map_capture_to_topics(record, topics)
        assert "angular" in matched

    def test_extract_date_from_filename_with_prefix(self):
        assert (
            CompetenceBuilder._extract_date_from_filename("2026-01-15_postgresql_batch")
            == "2026-01-15"
        )

    def test_extract_date_from_filename_without_date(self):
        assert CompetenceBuilder._extract_date_from_filename("postgresql_basics") is None

    def test_extract_date_from_filename_embedded_date(self):
        assert (
            CompetenceBuilder._extract_date_from_filename("data_2025-12-01_export") == "2025-12-01"
        )


# ---------------------------------------------------------------------------
# CompetenceBuilder.build() tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def build_datalake(tmp_path: Path) -> Path:
    """Create a datalake with varied topic counts for build() testing.

    postgresql(50), docker(20), angular(5), testing(2) + recent auto-capture.
    """
    dl = tmp_path / "datalake"

    tp_dir = dl / "02-processed" / "training-pairs"

    _write_jsonl(
        tp_dir / "postgresql_advanced.jsonl",
        [{"input": f"q{i}", "output": f"a{i}", "category": "postgresql"} for i in range(50)],
    )
    _write_jsonl(
        tp_dir / "docker_containers.jsonl",
        [{"input": f"q{i}", "output": f"a{i}", "category": "docker"} for i in range(20)],
    )
    _write_jsonl(
        tp_dir / "angular_components.jsonl",
        [{"input": f"q{i}", "output": f"a{i}", "category": "angular"} for i in range(5)],
    )
    _write_jsonl(
        tp_dir / "testing_basics.jsonl",
        [{"input": f"q{i}", "output": f"a{i}", "category": "testing"} for i in range(2)],
    )

    # Recent auto-capture tagged "postgresql"
    ac_dir = dl / "01-raw" / "code-changes"
    now = datetime.now(tz=UTC)
    _write_jsonl(
        ac_dir / "2026-02-20_auto-captures.jsonl",
        [
            {
                "file_modified": "src/db/schema.sql",
                "project": "backend",
                "timestamp": now.isoformat(),
                "tags": ["postgresql"],
            },
        ],
    )

    return dl


class TestCompetenceBuilderBuild:
    """Test CompetenceBuilder.build() end-to-end."""

    def test_build_produces_sorted_results(self, build_datalake: Path):
        builder = CompetenceBuilder(build_datalake)
        cmap = builder.build()

        assert len(cmap.topics) >= 4
        scores = [e.score for e in cmap.topics]
        assert scores == sorted(scores, reverse=True), "Topics must be sorted by score descending"

    def test_postgresql_scores_highest(self, build_datalake: Path):
        builder = CompetenceBuilder(build_datalake)
        cmap = builder.build()

        assert cmap.topics[0].topic == "postgresql"
        assert cmap.topics[0].score > 0.0

    def test_testing_scores_lowest(self, build_datalake: Path):
        builder = CompetenceBuilder(build_datalake)
        cmap = builder.build()

        topic_names = [e.topic for e in cmap.topics]
        assert "testing" in topic_names
        testing_entry = cmap.get_entry("testing")
        assert testing_entry is not None
        # testing has fewest entries (2) and no recency, should score lowest
        assert testing_entry.score == cmap.topics[-1].score

    def test_build_with_explicit_topics(self, build_datalake: Path):
        builder = CompetenceBuilder(build_datalake)
        cmap = builder.build(topics=["docker", "angular"])

        assert len(cmap.topics) == 2
        topic_names = {e.topic for e in cmap.topics}
        assert topic_names == {"docker", "angular"}

    def test_build_with_graph(self, build_datalake: Path):
        """Build with a mock graph engine, assert entity_density > 0."""
        graph = MagicMock()

        pg_entity = MagicMock()
        pg_entity.id = "ent_pg"

        docker_entity = MagicMock()
        docker_entity.id = "ent_docker"

        def find_entity(name: str):
            if name == "postgresql":
                return pg_entity
            if name == "docker":
                return docker_entity
            return None

        graph.find_entity_by_name.side_effect = find_entity

        # postgresql: 15 edges (non-self-loop)
        pg_relations = []
        for i in range(15):
            rel = MagicMock()
            rel.source_id = "ent_pg"
            rel.target_id = f"ent_other_{i}"
            pg_relations.append(rel)

        # docker: 5 edges
        docker_relations = []
        for i in range(5):
            rel = MagicMock()
            rel.source_id = "ent_docker"
            rel.target_id = f"ent_other_{i}"
            docker_relations.append(rel)

        def get_relations(entity_id: str, direction: str = "both"):
            if entity_id == "ent_pg":
                return pg_relations
            if entity_id == "ent_docker":
                return docker_relations
            return []

        graph.get_relations.side_effect = get_relations

        builder = CompetenceBuilder(build_datalake, graph_engine=graph)
        cmap = builder.build()

        pg_entry = cmap.get_entry("postgresql")
        assert pg_entry is not None
        assert pg_entry.entity_density > 0.0, "Graph entity_density must be > 0 when edges exist"

        docker_entry = cmap.get_entry("docker")
        assert docker_entry is not None
        assert docker_entry.entity_density > 0.0

    def test_build_saves_to_file(self, build_datalake: Path, tmp_path: Path):
        output = tmp_path / "output" / "competence_map.json"
        builder = CompetenceBuilder(build_datalake)
        cmap = builder.build(output_path=output)

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["total_topics"] == cmap.total_topics
        assert len(data["topics"]) == len(cmap.topics)

    def test_build_empty_datalake(self, tmp_path: Path):
        empty_dl = tmp_path / "empty_datalake"
        empty_dl.mkdir(parents=True)
        builder = CompetenceBuilder(empty_dl)
        cmap = builder.build()

        assert cmap.topics == []
        assert cmap.total_topics == 0

    def test_build_total_topics_set(self, build_datalake: Path):
        builder = CompetenceBuilder(build_datalake)
        cmap = builder.build()

        assert cmap.total_topics == len(cmap.topics)
        assert cmap.total_topics >= 4

    def test_build_entries_have_valid_levels(self, build_datalake: Path):
        builder = CompetenceBuilder(build_datalake)
        cmap = builder.build()

        valid_levels = {"Expert", "Competent", "Novice", "Unknown"}
        for entry in cmap.topics:
            assert entry.level in valid_levels, f"Invalid level: {entry.level}"


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    """Test save/load/cache for CompetenceMap."""

    def test_save_load_roundtrip(self, tmp_path: Path):
        cmap = CompetenceMap(
            topics=[
                CompetenceEntry(
                    topic="python",
                    score=0.92,
                    level="Expert",
                    entries=100,
                    entity_density=0.75,
                    recency_weight=0.95,
                    last_activity="2026-02-20T10:00:00",
                ),
                CompetenceEntry(
                    topic="docker",
                    score=0.45,
                    level="Competent",
                    entries=20,
                    entity_density=0.3,
                    recency_weight=0.6,
                    last_activity="2026-02-10T10:00:00",
                ),
            ],
            built_at="2026-02-20T12:00:00",
            total_topics=2,
        )

        path = tmp_path / "profile" / "competence_map.json"
        save_competence_map(cmap, path)
        assert path.exists()

        loaded = load_competence_map(path)
        assert loaded.total_topics == 2
        assert loaded.built_at == "2026-02-20T12:00:00"
        assert len(loaded.topics) == 2
        assert loaded.topics[0].topic == "python"
        assert loaded.topics[0].score == 0.92
        assert loaded.topics[1].topic == "docker"
        assert loaded.topics[1].level == "Competent"

    def test_load_nonexistent_returns_empty(self, tmp_path: Path):
        path = tmp_path / "nonexistent" / "map.json"
        cmap = load_competence_map(path)
        assert cmap.topics == []
        assert cmap.total_topics == 0

    def test_load_corrupt_json_returns_empty(self, tmp_path: Path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{{")
        cmap = load_competence_map(path)
        assert cmap.topics == []
        assert cmap.total_topics == 0

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "deep" / "nested" / "dir" / "map.json"
        cmap = CompetenceMap(
            topics=[CompetenceEntry(topic="go", score=0.5, level="Competent")],
            total_topics=1,
        )
        save_competence_map(cmap, path)
        assert path.exists()

    def test_get_active_competence_map_caches(self, tmp_path: Path):
        _competence_cache.clear()

        path = tmp_path / "profile" / "competence_map.json"
        cmap = CompetenceMap(
            topics=[CompetenceEntry(topic="rust", score=0.3, level="Novice")],
            total_topics=1,
        )
        save_competence_map(cmap, path)

        result1 = get_active_competence_map(map_path=path)
        result2 = get_active_competence_map(map_path=path)

        assert result1 is result2, "Cached results must be the same object"
        assert result1.total_topics == 1
        assert result1.topics[0].topic == "rust"

        _competence_cache.clear()

    def test_get_active_competence_map_nonexistent_returns_empty(self, tmp_path: Path):
        _competence_cache.clear()

        path = tmp_path / "missing" / "map.json"
        result = get_active_competence_map(map_path=path)
        assert result.topics == []
        assert result.total_topics == 0

        _competence_cache.clear()


# ---------------------------------------------------------------------------
# System prompt integration tests
# ---------------------------------------------------------------------------


class TestSystemPromptIntegration:
    """Test competence fragment injection into system prompts."""

    def test_profile_prompt_plus_competence_fragment(self):
        """Combined profile prompt and competence fragment."""
        from src.core.personal_profile import PersonalProfile

        profile = PersonalProfile(domain="software_development", domain_confidence=0.9)
        profile_system = profile.to_system_prompt()

        cmap = CompetenceMap(
            topics=[
                CompetenceEntry(topic="python", score=0.9, level="Expert"),
                CompetenceEntry(topic="docker", score=0.5, level="Competent"),
            ]
        )
        fragment = cmap.to_system_prompt_fragment()

        combined = f"{profile_system} {fragment}"
        assert "software development" in combined
        assert "Expert in: python" in combined
        assert "Competent in: docker" in combined

    def test_empty_competence_adds_nothing(self):
        """Empty competence map should not alter the prompt."""
        from src.core.personal_profile import PersonalProfile

        profile = PersonalProfile(domain="software_development", domain_confidence=0.9)
        profile_system = profile.to_system_prompt()

        cmap = CompetenceMap()
        fragment = cmap.to_system_prompt_fragment()

        if fragment:
            combined = f"{profile_system} {fragment}"
        else:
            combined = profile_system

        assert combined == profile_system


# ---------------------------------------------------------------------------
# Outcome signal tests
# ---------------------------------------------------------------------------


class TestOutcomeSignal:
    """Test outcome_rate as 4th signal in compute_competence_score."""

    def _ref(self) -> datetime:
        return datetime(2026, 2, 20, 12, 0, 0, tzinfo=UTC)

    def test_compute_with_outcome_rate(self):
        """outcome_rate=0.8 produces a valid score using WEIGHTS_ALL."""
        ref = self._ref()
        last = ref.isoformat()
        final, entry_s, density_s, recency_s = compute_competence_score(
            entries=100,
            edge_count=100,
            last_activity_iso=last,
            reference_time=ref,
            outcome_rate=0.8,
        )
        assert entry_s == 1.0
        assert density_s == 1.0
        assert recency_s >= 0.99
        expected = (
            WEIGHTS_ALL["entry"] * entry_s
            + WEIGHTS_ALL["density"] * density_s
            + WEIGHTS_ALL["recency"] * recency_s
            + WEIGHTS_ALL["outcome"] * 0.8
        )
        assert abs(final - expected) < 0.01

    def test_outcome_rate_none_degrades(self):
        """outcome_rate=None uses NO_OUTCOME weights (original behavior)."""
        ref = self._ref()
        last = ref.isoformat()
        final, entry_s, density_s, recency_s = compute_competence_score(
            entries=100,
            edge_count=100,
            last_activity_iso=last,
            reference_time=ref,
            outcome_rate=None,
        )
        # Should use WEIGHTS_ALL_NO_OUTCOME which is the original WEIGHTS_ALL
        expected = (
            WEIGHTS_ALL_NO_OUTCOME["entry"] * entry_s
            + WEIGHTS_ALL_NO_OUTCOME["density"] * density_s
            + WEIGHTS_ALL_NO_OUTCOME["recency"] * recency_s
        )
        assert abs(final - expected) < 0.01

    def test_high_outcome_boosts_score(self):
        """0.9 rate > 0.2 rate for same entries/edges/recency."""
        ref = self._ref()
        last = ref.isoformat()
        final_high, _, _, _ = compute_competence_score(
            entries=50,
            edge_count=50,
            last_activity_iso=last,
            reference_time=ref,
            outcome_rate=0.9,
        )
        final_low, _, _, _ = compute_competence_score(
            entries=50,
            edge_count=50,
            last_activity_iso=last,
            reference_time=ref,
            outcome_rate=0.2,
        )
        assert final_high > final_low

    def test_weights_sum_to_one(self):
        """Weight sets with multiple signals sum to approximately 1.0.

        ENTRY_ONLY variants intentionally sum to < 1.0 to cap
        the maximum score when only entry data is available.
        """
        full_weight_sets = [
            WEIGHTS_ALL,
            WEIGHTS_NO_GRAPH,
            WEIGHTS_NO_RECENCY,
            WEIGHTS_ALL_NO_OUTCOME,
            WEIGHTS_NO_GRAPH_NO_OUTCOME,
            WEIGHTS_NO_RECENCY_NO_OUTCOME,
        ]
        for ws in full_weight_sets:
            total = sum(ws.values())
            assert abs(total - 1.0) < 0.01, f"Weight set {ws} sums to {total}, not ~1.0"

        # ENTRY_ONLY variants are intentionally < 1.0 (capped)
        for ws in [WEIGHTS_ENTRY_ONLY, WEIGHTS_ENTRY_ONLY_NO_OUTCOME]:
            total = sum(ws.values())
            assert total <= 1.0, f"Weight set {ws} sums to {total}, should be <= 1.0"
            assert total > 0.0, f"Weight set {ws} sums to {total}, should be > 0.0"

    def test_no_graph_with_outcome(self):
        """edge_count=None + outcome_rate uses WEIGHTS_NO_GRAPH."""
        ref = self._ref()
        last = ref.isoformat()
        final, entry_s, _, recency_s = compute_competence_score(
            entries=50,
            edge_count=None,
            last_activity_iso=last,
            reference_time=ref,
            outcome_rate=0.7,
        )
        expected = (
            WEIGHTS_NO_GRAPH["entry"] * entry_s
            + WEIGHTS_NO_GRAPH["recency"] * recency_s
            + WEIGHTS_NO_GRAPH["outcome"] * 0.7
        )
        assert abs(final - expected) < 0.01

    def test_no_recency_with_outcome(self):
        """Empty recency + outcome_rate uses WEIGHTS_NO_RECENCY."""
        ref = self._ref()
        final, entry_s, density_s, _ = compute_competence_score(
            entries=50,
            edge_count=50,
            last_activity_iso="",
            reference_time=ref,
            outcome_rate=0.6,
        )
        expected = (
            WEIGHTS_NO_RECENCY["entry"] * entry_s
            + WEIGHTS_NO_RECENCY["density"] * density_s
            + WEIGHTS_NO_RECENCY["outcome"] * 0.6
        )
        assert abs(final - expected) < 0.01

    def test_entry_only_with_outcome(self):
        """No graph + no recency + outcome_rate uses WEIGHTS_ENTRY_ONLY."""
        ref = self._ref()
        final, entry_s, _, _ = compute_competence_score(
            entries=100,
            edge_count=None,
            last_activity_iso="",
            reference_time=ref,
            outcome_rate=0.5,
        )
        expected = WEIGHTS_ENTRY_ONLY["entry"] * entry_s + WEIGHTS_ENTRY_ONLY["outcome"] * 0.5
        assert abs(final - expected) < 0.01

    def test_outcome_zero_lowers_score(self):
        """outcome_rate=0.0 should produce lower score than no outcome at all."""
        ref = self._ref()
        last = ref.isoformat()
        # With outcome=0.0, outcome signal contributes 0 but weights are redistributed
        final_with_zero, _, _, _ = compute_competence_score(
            entries=100,
            edge_count=100,
            last_activity_iso=last,
            reference_time=ref,
            outcome_rate=0.0,
        )
        # Without outcome, uses original weights
        final_without, _, _, _ = compute_competence_score(
            entries=100,
            edge_count=100,
            last_activity_iso=last,
            reference_time=ref,
            outcome_rate=None,
        )
        # 0.0 outcome should lower the score compared to no outcome
        # because no-outcome uses original weights which give more to entry/density/recency
        assert final_with_zero < final_without


class TestCompetenceBuilderOutcomes:
    """Test CompetenceBuilder integration with outcome data."""

    def test_build_reads_outcome_rates(self, tmp_datalake: Path):
        """With outcome data in datalake, build uses outcome_rate."""
        # Create outcomes directory with data
        outcomes_dir = tmp_datalake / "01-raw" / "outcomes"
        outcomes_dir.mkdir(parents=True, exist_ok=True)

        # Write outcome data for postgresql (>= 5 non-neutral outcomes)
        _write_jsonl(
            outcomes_dir / "2026-02-20_outcomes.jsonl",
            [
                {"topic": "postgresql", "outcome": "accepted"},
                {"topic": "postgresql", "outcome": "accepted"},
                {"topic": "postgresql", "outcome": "accepted"},
                {"topic": "postgresql", "outcome": "accepted"},
                {"topic": "postgresql", "outcome": "rejected"},
                {"topic": "postgresql", "outcome": "accepted"},
                {"topic": "postgresql", "outcome": "accepted"},
                # debugging has only 3 outcomes (below threshold of 5)
                {"topic": "debugging", "outcome": "accepted"},
                {"topic": "debugging", "outcome": "rejected"},
                {"topic": "debugging", "outcome": "accepted"},
            ],
        )

        builder = CompetenceBuilder(tmp_datalake)
        cmap = builder.build()

        # postgresql should have outcome data factored in
        pg_entry = cmap.get_entry("postgresql")
        assert pg_entry is not None
        assert pg_entry.score > 0.0

        # debugging had < 5 outcomes, so outcome_rate should be None (degraded)
        debug_entry = cmap.get_entry("debugging")
        assert debug_entry is not None
        assert debug_entry.score > 0.0

    def test_build_without_outcomes_still_works(self, tmp_datalake: Path):
        """No outcome directory -> works fine with original weights."""
        # Ensure no outcomes directory exists
        outcomes_dir = tmp_datalake / "01-raw" / "outcomes"
        assert not outcomes_dir.exists()

        builder = CompetenceBuilder(tmp_datalake)
        cmap = builder.build()

        assert len(cmap.topics) >= 3
        for entry in cmap.topics:
            assert entry.score > 0.0

    def test_get_outcome_rates_empty_dir(self, tmp_datalake: Path):
        """Empty outcomes directory returns empty dict."""
        outcomes_dir = tmp_datalake / "01-raw" / "outcomes"
        outcomes_dir.mkdir(parents=True, exist_ok=True)

        builder = CompetenceBuilder(tmp_datalake)
        rates = builder._get_outcome_rates(["postgresql", "debugging"])
        assert rates == {}

    def test_get_outcome_rates_below_threshold(self, tmp_datalake: Path):
        """Topics with < 5 outcomes are excluded."""
        outcomes_dir = tmp_datalake / "01-raw" / "outcomes"
        outcomes_dir.mkdir(parents=True, exist_ok=True)

        _write_jsonl(
            outcomes_dir / "2026-02-20_outcomes.jsonl",
            [
                {"topic": "postgresql", "outcome": "accepted"},
                {"topic": "postgresql", "outcome": "rejected"},
                {"topic": "postgresql", "outcome": "accepted"},
                {"topic": "postgresql", "outcome": "accepted"},
                # Only 4 outcomes - below threshold
            ],
        )

        builder = CompetenceBuilder(tmp_datalake)
        rates = builder._get_outcome_rates(["postgresql"])
        assert "postgresql" not in rates

    def test_get_outcome_rates_correct_rate(self, tmp_datalake: Path):
        """Rate is accepted/total for topics with >= 5 outcomes."""
        outcomes_dir = tmp_datalake / "01-raw" / "outcomes"
        outcomes_dir.mkdir(parents=True, exist_ok=True)

        _write_jsonl(
            outcomes_dir / "2026-02-20_outcomes.jsonl",
            [
                {"topic": "postgresql", "outcome": "accepted"},
                {"topic": "postgresql", "outcome": "accepted"},
                {"topic": "postgresql", "outcome": "accepted"},
                {"topic": "postgresql", "outcome": "rejected"},
                {"topic": "postgresql", "outcome": "rejected"},
            ],
        )

        builder = CompetenceBuilder(tmp_datalake)
        rates = builder._get_outcome_rates(["postgresql"])
        assert "postgresql" in rates
        # 3 accepted / 5 total = 0.6
        assert abs(rates["postgresql"] - 0.6) < 0.001

    def test_get_outcome_rates_ignores_neutral(self, tmp_datalake: Path):
        """Neutral outcomes are not counted."""
        outcomes_dir = tmp_datalake / "01-raw" / "outcomes"
        outcomes_dir.mkdir(parents=True, exist_ok=True)

        _write_jsonl(
            outcomes_dir / "2026-02-20_outcomes.jsonl",
            [
                {"topic": "postgresql", "outcome": "accepted"},
                {"topic": "postgresql", "outcome": "accepted"},
                {"topic": "postgresql", "outcome": "accepted"},
                {"topic": "postgresql", "outcome": "rejected"},
                {"topic": "postgresql", "outcome": "neutral"},
                {"topic": "postgresql", "outcome": "neutral"},
                {"topic": "postgresql", "outcome": "accepted"},
            ],
        )

        builder = CompetenceBuilder(tmp_datalake)
        rates = builder._get_outcome_rates(["postgresql"])
        assert "postgresql" in rates
        # 4 accepted / 5 non-neutral (4 accepted + 1 rejected) = 0.8
        assert abs(rates["postgresql"] - 0.8) < 0.001

    def test_get_outcome_rates_multiple_files(self, tmp_datalake: Path):
        """Outcomes are aggregated across multiple JSONL files."""
        outcomes_dir = tmp_datalake / "01-raw" / "outcomes"
        outcomes_dir.mkdir(parents=True, exist_ok=True)

        _write_jsonl(
            outcomes_dir / "2026-02-18_outcomes.jsonl",
            [
                {"topic": "postgresql", "outcome": "accepted"},
                {"topic": "postgresql", "outcome": "accepted"},
                {"topic": "postgresql", "outcome": "accepted"},
            ],
        )
        _write_jsonl(
            outcomes_dir / "2026-02-19_outcomes.jsonl",
            [
                {"topic": "postgresql", "outcome": "rejected"},
                {"topic": "postgresql", "outcome": "rejected"},
            ],
        )

        builder = CompetenceBuilder(tmp_datalake)
        rates = builder._get_outcome_rates(["postgresql"])
        assert "postgresql" in rates
        # 3 accepted / 5 total = 0.6
        assert abs(rates["postgresql"] - 0.6) < 0.001

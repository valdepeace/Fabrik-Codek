"""Tests for the Strategy Optimizer."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.core.strategy_optimizer import (
    MINIMUM_SAMPLE_SIZE,
    StrategyOptimizer,
)
from src.core.task_router import TASK_STRATEGIES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_outcome(
    task_type: str = "debugging",
    topic: str = "postgresql",
    outcome: str = "accepted",
) -> dict:
    return {
        "task_type": task_type,
        "topic": topic,
        "outcome": outcome,
        "timestamp": datetime.now().isoformat(),
    }


def _write_outcomes(path: Path, outcomes: list[dict], date_str: str | None = None) -> None:
    outdir = path / "01-raw" / "outcomes"
    outdir.mkdir(parents=True, exist_ok=True)
    today = date_str or datetime.now().strftime("%Y-%m-%d")
    filepath = outdir / f"{today}_outcomes.jsonl"
    with open(filepath, "w") as f:
        for o in outcomes:
            f.write(json.dumps(o) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoOutcomes:
    def test_no_outcomes_no_overrides(self, tmp_path: Path) -> None:
        """With no outcome data at all, compute_overrides returns empty dict."""
        optimizer = StrategyOptimizer(tmp_path)
        result = optimizer.compute_overrides()
        assert result == {}


class TestBelowMinimumSample:
    def test_below_minimum_sample_no_override(self, tmp_path: Path) -> None:
        """Fewer than MINIMUM_SAMPLE_SIZE non-neutral outcomes -> no override."""
        # Write 9 accepted outcomes (below threshold of 10)
        outcomes = [_make_outcome() for _ in range(MINIMUM_SAMPLE_SIZE - 1)]
        _write_outcomes(tmp_path, outcomes)

        optimizer = StrategyOptimizer(tmp_path)
        result = optimizer.compute_overrides()
        assert result == {}


class TestHighAcceptance:
    def test_high_acceptance_no_override(self, tmp_path: Path) -> None:
        """Acceptance rate >= 0.7 produces no override (defaults are fine)."""
        # 8 accepted + 2 rejected = 80% acceptance rate
        outcomes = [_make_outcome(outcome="accepted") for _ in range(8)] + [
            _make_outcome(outcome="rejected") for _ in range(2)
        ]
        _write_outcomes(tmp_path, outcomes)

        optimizer = StrategyOptimizer(tmp_path)
        result = optimizer.compute_overrides()
        assert result == {}


class TestMediumAcceptance:
    def test_medium_acceptance_mild_boost(self, tmp_path: Path) -> None:
        """Acceptance rate 0.5-0.7 boosts graph_depth +1 and graph_weight +0.1."""
        # 6 accepted + 4 rejected = 60% acceptance (between 0.5 and 0.7)
        outcomes = [_make_outcome(outcome="accepted") for _ in range(6)] + [
            _make_outcome(outcome="rejected") for _ in range(4)
        ]
        _write_outcomes(tmp_path, outcomes)

        optimizer = StrategyOptimizer(tmp_path)
        result = optimizer.compute_overrides()

        key = "debugging_postgresql"
        assert key in result

        base = TASK_STRATEGIES["debugging"]
        override = result[key]
        assert override["graph_depth"] == base["graph_depth"] + 1
        assert override["graph_weight"] == pytest.approx(base["graph_weight"] + 0.1)
        assert override["vector_weight"] == base["vector_weight"]
        assert "fulltext_weight" not in override or override["fulltext_weight"] == 0.0
        assert override["acceptance_rate"] == pytest.approx(0.6)
        assert override["sample_size"] == 10
        assert "updated_at" in override


class TestLowAcceptance:
    def test_low_acceptance_strong_boost(self, tmp_path: Path) -> None:
        """Acceptance rate < 0.5 gets depth +2, weight +0.2, fulltext 0.1."""
        # 4 accepted + 6 rejected = 40% acceptance (below 0.5)
        outcomes = [_make_outcome(outcome="accepted") for _ in range(4)] + [
            _make_outcome(outcome="rejected") for _ in range(6)
        ]
        _write_outcomes(tmp_path, outcomes)

        optimizer = StrategyOptimizer(tmp_path)
        result = optimizer.compute_overrides()

        key = "debugging_postgresql"
        assert key in result

        base = TASK_STRATEGIES["debugging"]
        override = result[key]
        assert override["graph_depth"] == base["graph_depth"] + 2
        assert override["graph_weight"] == pytest.approx(base["graph_weight"] + 0.2)
        assert override["fulltext_weight"] == pytest.approx(0.1)
        assert override["acceptance_rate"] == pytest.approx(0.4)
        assert override["sample_size"] == 10


class TestNeutralExcluded:
    def test_neutral_outcomes_excluded_from_rate(self, tmp_path: Path) -> None:
        """Neutral outcomes are excluded from acceptance rate calculation."""
        # 7 accepted + 3 rejected + 5 neutral = 10 non-neutral, 70% acceptance
        # 70% >= HIGH_ACCEPTANCE_THRESHOLD so no override
        outcomes = (
            [_make_outcome(outcome="accepted") for _ in range(7)]
            + [_make_outcome(outcome="rejected") for _ in range(3)]
            + [_make_outcome(outcome="neutral") for _ in range(5)]
        )
        _write_outcomes(tmp_path, outcomes)

        optimizer = StrategyOptimizer(tmp_path)
        result = optimizer.compute_overrides()
        assert result == {}


class TestSaveOverrides:
    def test_save_overrides(self, tmp_path: Path) -> None:
        """save_overrides writes JSON file and returns count of overrides."""
        # Low acceptance to produce an override
        outcomes = [_make_outcome(outcome="accepted") for _ in range(4)] + [
            _make_outcome(outcome="rejected") for _ in range(6)
        ]
        _write_outcomes(tmp_path, outcomes)

        optimizer = StrategyOptimizer(tmp_path)
        output_path = tmp_path / "overrides.json"
        count = optimizer.save_overrides(output_path)

        assert count == 1
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)
        assert "debugging_postgresql" in data


class TestNoneTopicKey:
    def test_none_topic_uses_task_type_only(self, tmp_path: Path) -> None:
        """When topic is None, the key is just the task_type string."""
        outcomes = [_make_outcome(topic=None, outcome="accepted") for _ in range(4)] + [
            _make_outcome(topic=None, outcome="rejected") for _ in range(6)
        ]
        _write_outcomes(tmp_path, outcomes)

        optimizer = StrategyOptimizer(tmp_path)
        result = optimizer.compute_overrides()

        # Key should be just "debugging" (no topic suffix)
        assert "debugging" in result
        assert "debugging_None" not in result
        assert "debugging_" not in result

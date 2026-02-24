"""Strategy Optimizer - Generates retrieval overrides from outcome data.

Reads outcome records from the datalake, computes acceptance rates per
(task_type, topic) combination, and generates retrieval strategy overrides
for underperforming combinations.  Overrides are layered on top of the
default ``TASK_STRATEGIES`` defined in ``task_router.py``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import structlog

from src.core.task_router import TASK_STRATEGIES

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINIMUM_SAMPLE_SIZE: int = 10
HIGH_ACCEPTANCE_THRESHOLD: float = 0.7
MEDIUM_ACCEPTANCE_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# StrategyOptimizer
# ---------------------------------------------------------------------------


class StrategyOptimizer:
    """Analyze outcome data and produce retrieval strategy overrides.

    Parameters
    ----------
    datalake_path:
        Root of the datalake directory.
        Outcome JSONL files are expected under
        ``datalake_path / "01-raw" / "outcomes" / "*_outcomes.jsonl"``.
    """

    def __init__(self, datalake_path: Path) -> None:
        self.datalake_path = datalake_path

    # -- reading outcomes ---------------------------------------------------

    def _read_outcomes(self, days: int = 30) -> list[dict]:
        """Read outcome records from the last *days* days.

        Parses the date from the filename (``YYYY-MM-DD_outcomes.jsonl``)
        and skips files older than the window.
        """
        outcomes_dir = self.datalake_path / "01-raw" / "outcomes"
        if not outcomes_dir.exists():
            return []

        cutoff = datetime.now() - timedelta(days=days)
        results: list[dict] = []

        for filepath in sorted(outcomes_dir.glob("*_outcomes.jsonl")):
            # Extract date from filename: "2026-02-24_outcomes.jsonl"
            date_str = filepath.name.split("_outcomes")[0]
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                logger.warning("skipping_unparseable_outcome_file", path=str(filepath))
                continue

            if file_date < cutoff:
                continue

            with filepath.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("skipping_invalid_json_line", path=str(filepath))
                        continue

        return results

    # -- computing overrides ------------------------------------------------

    def compute_overrides(self, days: int = 30) -> dict:
        """Compute retrieval strategy overrides for underperforming combos.

        Returns a dict keyed by ``"{task_type}_{topic}"`` (or just
        ``"{task_type}"`` when topic is ``None``), where each value
        contains the adjusted retrieval parameters plus metadata.
        """
        outcomes = self._read_outcomes(days)
        if not outcomes:
            return {}

        # Aggregate by (task_type, topic) ----------------------------------
        buckets: dict[str, list[dict]] = {}
        for record in outcomes:
            task_type = record.get("task_type", "general")
            topic = record.get("topic")
            key = f"{task_type}_{topic}" if topic else task_type
            buckets.setdefault(key, []).append(record)

        # Build overrides --------------------------------------------------
        overrides: dict[str, dict] = {}
        for key, records in buckets.items():
            # Exclude neutral outcomes
            non_neutral = [r for r in records if r.get("outcome") != "neutral"]
            total = len(non_neutral)

            if total < MINIMUM_SAMPLE_SIZE:
                continue

            accepted = sum(1 for r in non_neutral if r.get("outcome") == "accepted")
            rate = accepted / total

            if rate >= HIGH_ACCEPTANCE_THRESHOLD:
                # Defaults are fine -- no override needed
                continue

            # Resolve base strategy from the task_type part of the key
            task_type = key.split("_")[0]
            base = TASK_STRATEGIES.get(task_type, TASK_STRATEGIES["general"])

            override: dict = {
                "vector_weight": base["vector_weight"],
                "acceptance_rate": round(rate, 4),
                "sample_size": total,
                "updated_at": datetime.now().isoformat(),
            }

            if rate >= MEDIUM_ACCEPTANCE_THRESHOLD:
                # Mild boost: depth +1, graph_weight +0.1
                override["graph_depth"] = base["graph_depth"] + 1
                override["graph_weight"] = round(base["graph_weight"] + 0.1, 4)
                override["fulltext_weight"] = 0.0
            else:
                # Strong boost: depth +2, graph_weight +0.2, fulltext 0.1
                override["graph_depth"] = base["graph_depth"] + 2
                override["graph_weight"] = round(base["graph_weight"] + 0.2, 4)
                override["fulltext_weight"] = 0.1

            overrides[key] = override

        return overrides

    # -- persistence --------------------------------------------------------

    def save_overrides(self, output_path: Path) -> int:
        """Compute overrides and write them to *output_path* as JSON.

        Returns the number of overrides written.
        """
        overrides = self.compute_overrides()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(overrides, fh, indent=2, ensure_ascii=False)

        logger.info(
            "strategy_overrides_saved",
            count=len(overrides),
            path=str(output_path),
        )
        return len(overrides)

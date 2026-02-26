"""Tests for flywheel collector."""

import json
import tempfile
from pathlib import Path

import aiofiles
import pytest

from src.flywheel.collector import FlywheelCollector, InteractionRecord


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def collector(temp_data_dir):
    """Create collector with temp directory."""
    return FlywheelCollector(data_dir=temp_data_dir, batch_size=5, enabled=True)


class TestInteractionRecord:
    """Tests for InteractionRecord."""

    def test_create_record(self):
        """Test creating a record."""
        record = InteractionRecord(
            interaction_type="code_generation",
            input_text="Write a hello world",
            output_text="print('Hello, World!')",
            model="qwen2.5-coder:7b",
        )

        assert record.interaction_type == "code_generation"
        assert record.input_text == "Write a hello world"
        assert record.id is not None
        assert record.timestamp is not None

    def test_to_training_pair(self):
        """Test conversion to training pair."""
        record = InteractionRecord(
            interaction_type="code_generation",
            input_text="Write a hello world",
            output_text="print('Hello, World!')",
            context="Python file",
            user_feedback="positive",
        )

        pair = record.to_training_pair()

        assert pair["instruction"] == "Write a hello world"
        assert pair["input"] == "Python file"
        assert pair["output"] == "print('Hello, World!')"
        assert pair["metadata"]["feedback"] == "positive"


class TestFlywheelCollector:
    """Tests for FlywheelCollector."""

    @pytest.mark.asyncio
    async def test_capture_record(self, collector):
        """Test capturing a record."""
        record = InteractionRecord(
            interaction_type="prompt_response",
            input_text="test",
            output_text="response",
        )

        await collector.capture(record)
        stats = await collector.get_session_stats()

        assert stats["buffered_records"] == 1

    @pytest.mark.asyncio
    async def test_capture_prompt_response(self, collector):
        """Test convenience capture method."""
        record = await collector.capture_prompt_response(
            prompt="Hello",
            response="Hi there!",
            model="test-model",
            tokens=10,
        )

        assert record.input_text == "Hello"
        assert record.output_text == "Hi there!"
        assert record.model == "test-model"

    @pytest.mark.asyncio
    async def test_flush_on_batch(self, collector, temp_data_dir):
        """Test auto-flush when batch size reached."""
        # Capture more than batch size
        for i in range(6):
            await collector.capture_prompt_response(
                prompt=f"prompt {i}",
                response=f"response {i}",
            )

        # Should have flushed 5, keeping 1 in buffer
        stats = await collector.get_session_stats()
        assert stats["buffered_records"] == 1

        # Check file was created
        interactions_dir = temp_data_dir / "interactions"
        files = list(interactions_dir.glob("*.jsonl"))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_disabled_collector(self, temp_data_dir):
        """Test that disabled collector doesn't capture."""
        collector = FlywheelCollector(
            data_dir=temp_data_dir,
            enabled=False,
        )

        await collector.capture_prompt_response(
            prompt="test",
            response="response",
        )

        stats = await collector.get_session_stats()
        assert stats["buffered_records"] == 0

    @pytest.mark.asyncio
    async def test_mark_feedback(self, collector):
        """Test marking feedback on record."""
        record = await collector.capture_prompt_response(
            prompt="test",
            response="response",
        )

        await collector.mark_feedback(record.id, "positive", was_edited=True)

        # Verify feedback was set
        for r in collector._buffer:
            if r.id == record.id:
                assert r.user_feedback == "positive"
                assert r.was_edited is True


class TestFeedbackPersistence:
    """Tests for mark_feedback sidecar persistence."""

    @pytest.mark.asyncio
    async def test_mark_feedback_persists_to_sidecar(self, collector, temp_data_dir):
        """Feedback on flushed record writes to sidecar file."""
        # Capture enough to trigger flush (batch_size=5)
        records = []
        for i in range(5):
            r = await collector.capture_prompt_response(
                prompt=f"prompt {i}",
                response=f"response {i}",
            )
            records.append(r)

        # Buffer is flushed, record is on disk only
        stats = await collector.get_session_stats()
        assert stats["buffered_records"] == 0

        # Mark feedback on flushed record
        await collector.mark_feedback(records[0].id, "positive")

        # Verify sidecar file exists with correct content
        feedback_dir = temp_data_dir / "feedback"
        files = list(feedback_dir.glob("*.jsonl"))
        assert len(files) == 1

        content = files[0].read_text().strip()
        entry = json.loads(content)
        assert entry["record_id"] == records[0].id
        assert entry["feedback"] == "positive"
        assert entry["was_edited"] is False
        assert entry["source"] == "manual"

    @pytest.mark.asyncio
    async def test_mark_feedback_buffer_priority(self, collector, temp_data_dir):
        """Feedback on buffered record updates buffer, no sidecar write."""
        record = await collector.capture_prompt_response(
            prompt="test",
            response="response",
        )

        await collector.mark_feedback(record.id, "negative", was_edited=True)

        # Buffer updated
        for r in collector._buffer:
            if r.id == record.id:
                assert r.user_feedback == "negative"
                assert r.was_edited is True

        # No sidecar written
        feedback_dir = temp_data_dir / "feedback"
        files = list(feedback_dir.glob("*.jsonl"))
        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_mark_feedback_source_field(self, collector, temp_data_dir):
        """Source field is persisted correctly."""
        for i in range(5):
            await collector.capture_prompt_response(
                prompt=f"p{i}",
                response=f"r{i}",
            )

        await collector.mark_feedback(
            "nonexistent-id",
            "positive",
            source="outcome_tracker",
        )

        feedback_dir = temp_data_dir / "feedback"
        files = list(feedback_dir.glob("*.jsonl"))
        assert len(files) == 1

        entry = json.loads(files[0].read_text().strip())
        assert entry["source"] == "outcome_tracker"


class TestFeedbackMerge:
    """Tests for feedback merge in export_training_pairs."""

    @pytest.mark.asyncio
    async def test_export_merges_feedback(self, collector, temp_data_dir):
        """Exported training pairs include sidecar feedback."""
        # Capture and flush
        records = []
        for i in range(5):
            r = await collector.capture_prompt_response(
                prompt=f"prompt {i}",
                response=f"response {i}",
            )
            records.append(r)

        # Mark feedback on flushed record
        await collector.mark_feedback(records[2].id, "positive")

        # Export
        output = await collector.export_training_pairs()

        # Read exported pairs
        pairs = []
        async with aiofiles.open(output, encoding="utf-8") as f:
            content = await f.read()
            for line in content.strip().split("\n"):
                if line:
                    pairs.append(json.loads(line))

        # Find the pair for records[2]
        feedbacked = [p for p in pairs if p["instruction"] == "prompt 2"]
        assert len(feedbacked) == 1
        assert feedbacked[0]["metadata"]["feedback"] == "positive"

    @pytest.mark.asyncio
    async def test_feedback_last_wins(self, collector, temp_data_dir):
        """Multiple feedbacks for same record — most recent wins."""
        for i in range(5):
            await collector.capture_prompt_response(
                prompt=f"p{i}",
                response=f"r{i}",
            )

        target_id = collector._buffer[0].id if collector._buffer else "dummy"

        # Force flush to ensure buffer empty, feedback goes to sidecar
        await collector.flush()

        await collector.mark_feedback(target_id, "negative")
        await collector.mark_feedback(target_id, "positive")

        index = collector._load_feedback_index()
        assert index[target_id].feedback == "positive"

    @pytest.mark.asyncio
    async def test_load_feedback_index_empty(self, collector, temp_data_dir):
        """No feedback files returns empty dict without errors."""
        index = collector._load_feedback_index()
        assert index == {}


class TestOutcomeFeedbackBridge:
    """Tests for OutcomeTracker -> mark_feedback bridge."""

    @pytest.mark.asyncio
    async def test_outcome_accepted_triggers_positive_feedback(self):
        """When OutcomeTracker infers 'accepted', mark_feedback is called
        with 'positive'."""
        from unittest.mock import AsyncMock, MagicMock

        from src.flywheel.collector import FlywheelCollector, bridge_outcome_to_feedback
        from src.flywheel.outcome_tracker import OutcomeRecord

        collector = MagicMock(spec=FlywheelCollector)
        collector.mark_feedback = AsyncMock()

        outcome = OutcomeRecord(
            query="how do I do X",
            response_summary="do Y",
            task_type="explanation",
            model="test",
            outcome="accepted",
            inference_reason="topic change",
        )

        await bridge_outcome_to_feedback(collector, outcome, "record-123")

        collector.mark_feedback.assert_called_once_with(
            "record-123",
            "positive",
            source="outcome_tracker",
        )

    @pytest.mark.asyncio
    async def test_outcome_rejected_triggers_negative_feedback(self):
        """When OutcomeTracker infers 'rejected', mark_feedback is called
        with 'negative'."""
        from unittest.mock import AsyncMock, MagicMock

        from src.flywheel.collector import FlywheelCollector, bridge_outcome_to_feedback
        from src.flywheel.outcome_tracker import OutcomeRecord

        collector = MagicMock(spec=FlywheelCollector)
        collector.mark_feedback = AsyncMock()

        outcome = OutcomeRecord(
            query="how do I do X",
            response_summary="do Y",
            task_type="explanation",
            model="test",
            outcome="rejected",
            inference_reason="reformulation",
        )

        await bridge_outcome_to_feedback(collector, outcome, "record-456")

        collector.mark_feedback.assert_called_once_with(
            "record-456",
            "negative",
            source="outcome_tracker",
        )

    @pytest.mark.asyncio
    async def test_outcome_neutral_no_feedback(self):
        """Neutral outcome does not trigger mark_feedback."""
        from unittest.mock import AsyncMock, MagicMock

        from src.flywheel.collector import FlywheelCollector, bridge_outcome_to_feedback
        from src.flywheel.outcome_tracker import OutcomeRecord

        collector = MagicMock(spec=FlywheelCollector)
        collector.mark_feedback = AsyncMock()

        outcome = OutcomeRecord(
            query="how do I do X",
            response_summary="do Y",
            task_type="explanation",
            model="test",
            outcome="neutral",
            inference_reason="session_close",
        )

        await bridge_outcome_to_feedback(collector, outcome, "record-789")

        collector.mark_feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_outcome_none_no_feedback(self):
        """None outcome (first turn) does not trigger mark_feedback."""
        from unittest.mock import AsyncMock, MagicMock

        from src.flywheel.collector import FlywheelCollector, bridge_outcome_to_feedback

        collector = MagicMock(spec=FlywheelCollector)
        collector.mark_feedback = AsyncMock()

        await bridge_outcome_to_feedback(collector, None, "record-000")

        collector.mark_feedback.assert_not_called()

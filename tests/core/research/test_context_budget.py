"""Tests for context budget management utilities.

Tests cover:
1. Priority scoring (compute_priority, compute_recency_score)
2. Allocation strategies (PRIORITY_FIRST, EQUAL_SHARE, PROPORTIONAL)
3. Protected content handling (protected flag prevents dropping)
4. Fidelity metadata accuracy (allocation_ratio, tokens_used)
5. ContentItem dataclass functionality
"""

import pytest

from foundry_mcp.core.research.context_budget import (
    AllocationStrategy,
    AllocatedItem,
    AllocationResult,
    ContentItem,
    ContentItemProtocol,
    ContextBudgetManager,
    CONFIDENCE_SCORES,
    PRIORITY_WEIGHT_CONFIDENCE,
    PRIORITY_WEIGHT_SOURCE_QUALITY,
    SOURCE_QUALITY_SCORES,
    compute_priority,
    compute_recency_score,
)
from foundry_mcp.core.research.models import ConfidenceLevel, SourceQuality


# =============================================================================
# Test: Priority Scoring (compute_priority)
# =============================================================================


class TestComputePriority:
    """Tests for compute_priority function."""

    def test_maximum_priority_score(self):
        """Test that best values produce score of 1.0."""
        score = compute_priority(
            source_quality=SourceQuality.HIGH,
            confidence=ConfidenceLevel.CONFIRMED,
            recency_score=1.0,
            relevance_score=1.0,
        )
        assert score == 1.0

    def test_minimum_priority_score(self):
        """Test that worst values produce low score."""
        score = compute_priority(
            source_quality=SourceQuality.LOW,
            confidence=ConfidenceLevel.SPECULATION,
            recency_score=0.0,
            relevance_score=0.0,
        )
        # 0.4 * 0.4 + 0.3 * 0.2 + 0.15 * 0 + 0.15 * 0 = 0.16 + 0.06 = 0.22
        assert 0.2 <= score <= 0.25

    def test_default_values(self):
        """Test priority with default parameters."""
        score = compute_priority()
        # UNKNOWN quality (0.5) and MEDIUM confidence (0.7), 0.5 recency/relevance
        # 0.4 * 0.5 + 0.3 * 0.7 + 0.15 * 0.5 + 0.15 * 0.5 = 0.2 + 0.21 + 0.075 + 0.075 = 0.56
        assert 0.5 <= score <= 0.6

    def test_source_quality_weight(self):
        """Test source quality impacts score correctly."""
        high = compute_priority(source_quality=SourceQuality.HIGH)
        low = compute_priority(source_quality=SourceQuality.LOW)
        # Difference should be proportional to weight
        assert high > low
        expected_diff = PRIORITY_WEIGHT_SOURCE_QUALITY * (1.0 - 0.4)
        assert abs((high - low) - expected_diff) < 0.01

    def test_confidence_weight(self):
        """Test confidence level impacts score correctly."""
        confirmed = compute_priority(confidence=ConfidenceLevel.CONFIRMED)
        speculation = compute_priority(confidence=ConfidenceLevel.SPECULATION)
        assert confirmed > speculation
        expected_diff = PRIORITY_WEIGHT_CONFIDENCE * (1.0 - 0.2)
        assert abs((confirmed - speculation) - expected_diff) < 0.01

    def test_invalid_recency_score_raises(self):
        """Test that invalid recency score raises ValueError."""
        with pytest.raises(ValueError, match="recency_score"):
            compute_priority(recency_score=1.5)
        with pytest.raises(ValueError, match="recency_score"):
            compute_priority(recency_score=-0.1)

    def test_invalid_relevance_score_raises(self):
        """Test that invalid relevance score raises ValueError."""
        with pytest.raises(ValueError, match="relevance_score"):
            compute_priority(relevance_score=1.1)
        with pytest.raises(ValueError, match="relevance_score"):
            compute_priority(relevance_score=-0.5)

    def test_all_source_qualities_have_scores(self):
        """Test that all SourceQuality values have defined scores."""
        for quality in SourceQuality:
            assert quality in SOURCE_QUALITY_SCORES

    def test_all_confidence_levels_have_scores(self):
        """Test that all ConfidenceLevel values have defined scores."""
        for confidence in ConfidenceLevel:
            assert confidence in CONFIDENCE_SCORES


class TestComputeRecencyScore:
    """Tests for compute_recency_score function."""

    def test_brand_new_content(self):
        """Test that age 0 gives score of 1.0."""
        score = compute_recency_score(0.0)
        assert score == 1.0

    def test_max_age_content(self):
        """Test that age at max gives score of 0.0."""
        score = compute_recency_score(720.0)  # Default max is 720
        assert score == 0.0

    def test_beyond_max_age(self):
        """Test that age beyond max gives score of 0.0."""
        score = compute_recency_score(1000.0)
        assert score == 0.0

    def test_half_age_gives_half_score(self):
        """Test linear decay: half age = half score."""
        score = compute_recency_score(360.0)  # Half of 720
        assert score == 0.5

    def test_custom_max_age(self):
        """Test custom max_age_hours parameter."""
        score = compute_recency_score(12.0, max_age_hours=24.0)
        assert score == 0.5

    def test_negative_age_raises(self):
        """Test that negative age raises ValueError."""
        with pytest.raises(ValueError, match="age_hours"):
            compute_recency_score(-1.0)

    def test_zero_max_age_raises(self):
        """Test that zero max_age raises ValueError."""
        with pytest.raises(ValueError, match="max_age_hours"):
            compute_recency_score(10.0, max_age_hours=0.0)


# =============================================================================
# Test: ContentItem Dataclass
# =============================================================================


class TestContentItem:
    """Tests for ContentItem dataclass."""

    def test_create_basic_item(self):
        """Test creating a basic content item."""
        item = ContentItem(id="test-1", content="Hello world", priority=1)
        assert item.id == "test-1"
        assert item.content == "Hello world"
        assert item.priority == 1
        assert item.protected is False
        assert item.source_id is None
        assert item.token_count is None

    def test_create_protected_item(self):
        """Test creating a protected content item."""
        item = ContentItem(
            id="citation-1",
            content="Important citation",
            priority=1,
            protected=True,
        )
        assert item.protected is True

    def test_token_count_alias(self):
        """Test that tokens property returns token_count."""
        item = ContentItem(id="test", content="x", token_count=500)
        assert item.tokens == 500

    def test_tokens_none_when_not_set(self):
        """Test that tokens returns None when token_count not set."""
        item = ContentItem(id="test", content="x")
        assert item.tokens is None

    def test_implements_protocol(self):
        """Test that ContentItem implements ContentItemProtocol."""
        item = ContentItem(id="test", content="x", priority=1)
        assert isinstance(item, ContentItemProtocol)


# =============================================================================
# Test: Allocation Under Tight Budget
# =============================================================================


class TestAllocationTightBudget:
    """Tests for allocation behavior under tight budget constraints."""

    @pytest.fixture
    def manager(self):
        """Create a ContextBudgetManager with fixed token estimation."""
        # Use a simple estimator for predictable tests
        return ContextBudgetManager(
            token_estimator=lambda content: len(content) // 4
        )

    @pytest.fixture
    def items(self):
        """Create test items with known token counts."""
        return [
            ContentItem(id="high-1", content="A" * 400, priority=1),   # 100 tokens
            ContentItem(id="med-1", content="B" * 600, priority=2),    # 150 tokens
            ContentItem(id="low-1", content="C" * 800, priority=3),    # 200 tokens
        ]

    def test_all_items_fit(self, manager, items):
        """Test allocation when all items fit in budget."""
        result = manager.allocate_budget(items, budget=500)
        assert len(result.items) == 3
        assert len(result.dropped_ids) == 0
        assert result.fidelity == 1.0

    def test_partial_allocation(self, manager, items):
        """Test allocation when only some items fit."""
        result = manager.allocate_budget(items, budget=200)
        # High priority (100) fits, med priority (150) partially fits
        assert len(result.items) == 2
        assert len(result.dropped_ids) == 1
        assert "low-1" in result.dropped_ids

    def test_high_priority_preserved(self, manager, items):
        """Test that high-priority items get full allocation first."""
        result = manager.allocate_budget(items, budget=150)
        high_item = next(i for i in result.items if i.id == "high-1")
        assert not high_item.needs_summarization
        assert high_item.allocation_ratio == 1.0

    def test_low_priority_summarized(self, manager, items):
        """Test that low-priority items are marked for summarization."""
        result = manager.allocate_budget(items, budget=200)
        # Medium priority should need summarization (only 100 tokens left)
        med_item = next(i for i in result.items if i.id == "med-1")
        assert med_item.needs_summarization
        assert med_item.allocation_ratio < 1.0

    def test_zero_budget_drops_all_non_protected(self, manager):
        """Test that zero remaining budget drops unprotected items."""
        items = [
            ContentItem(id="a", content="x" * 400, priority=1),  # 100 tokens
            ContentItem(id="b", content="y" * 400, priority=2),  # 100 tokens
        ]
        # With only 50 tokens, first item takes what it can
        result = manager.allocate_budget(items, budget=50)
        assert "b" in result.dropped_ids


# =============================================================================
# Test: Protected Content Handling
# =============================================================================


class TestProtectedContentHandling:
    """Tests for protected content preservation."""

    @pytest.fixture
    def manager(self):
        """Create a ContextBudgetManager with fixed token estimation."""
        return ContextBudgetManager(
            token_estimator=lambda content: len(content) // 4
        )

    def test_protected_item_never_dropped(self, manager):
        """Test that protected items are allocated even when budget exhausted."""
        items = [
            ContentItem(id="regular-1", content="A" * 400, priority=1),  # 100 tokens
            ContentItem(id="regular-2", content="B" * 800, priority=2),  # 200 tokens
            ContentItem(id="protected-1", content="C" * 400, priority=3, protected=True),  # 100 tokens
            ContentItem(id="regular-3", content="D" * 400, priority=4),  # 100 tokens
        ]
        # Budget only fits first 2-3 items
        result = manager.allocate_budget(items, budget=250)

        # Protected item should be allocated, not dropped
        protected_allocated = any(i.id == "protected-1" for i in result.items)
        assert protected_allocated, "Protected item should never be dropped"
        assert "protected-1" not in result.dropped_ids

    def test_protected_item_gets_minimal_allocation(self, manager):
        """Test protected item gets at least minimal allocation when budget exhausted."""
        items = [
            ContentItem(id="big", content="A" * 2000, priority=1),  # 500 tokens
            ContentItem(id="protected", content="B" * 400, priority=2, protected=True),
        ]
        # Budget exhausted by first item
        result = manager.allocate_budget(items, budget=500)

        protected_item = next(i for i in result.items if i.id == "protected")
        assert protected_item.allocated_tokens >= 1
        assert protected_item.needs_summarization

    def test_multiple_protected_items(self, manager):
        """Test handling of multiple protected items."""
        items = [
            ContentItem(id="p1", content="A" * 400, priority=1, protected=True),
            ContentItem(id="regular", content="B" * 800, priority=2),
            ContentItem(id="p2", content="C" * 400, priority=3, protected=True),
        ]
        result = manager.allocate_budget(items, budget=100)

        # Both protected items should be present
        allocated_ids = {i.id for i in result.items}
        assert "p1" in allocated_ids
        assert "p2" in allocated_ids


# =============================================================================
# Test: Fidelity Metadata Accuracy
# =============================================================================


class TestFidelityMetadata:
    """Tests for fidelity metadata accuracy."""

    @pytest.fixture
    def manager(self):
        """Create a ContextBudgetManager with fixed token estimation."""
        return ContextBudgetManager(
            token_estimator=lambda content: len(content) // 4
        )

    def test_full_fidelity_ratio(self, manager):
        """Test that fully allocated items have ratio 1.0."""
        items = [ContentItem(id="a", content="x" * 400, priority=1)]
        result = manager.allocate_budget(items, budget=1000)

        assert result.items[0].allocation_ratio == 1.0
        assert not result.items[0].needs_summarization

    def test_partial_fidelity_ratio(self, manager):
        """Test that partially allocated items have correct ratio."""
        items = [ContentItem(id="a", content="x" * 400, priority=1)]  # 100 tokens
        result = manager.allocate_budget(items, budget=50)

        item = result.items[0]
        assert item.original_tokens == 100
        assert item.allocated_tokens == 50
        assert item.allocation_ratio == 0.5
        assert item.needs_summarization

    def test_overall_fidelity_calculation(self, manager):
        """Test that overall fidelity reflects allocation quality."""
        items = [
            ContentItem(id="a", content="x" * 400, priority=1),  # 100 tokens
            ContentItem(id="b", content="y" * 400, priority=2),  # 100 tokens
        ]

        # Full allocation
        result_full = manager.allocate_budget(items, budget=200)
        assert result_full.fidelity == 1.0

        # Half allocation (only first item fits)
        result_half = manager.allocate_budget(items, budget=100)
        assert result_half.fidelity == 0.5

    def test_tokens_used_accuracy(self, manager):
        """Test that tokens_used reflects actual allocation."""
        items = [
            ContentItem(id="a", content="x" * 400, priority=1),  # 100 tokens
            ContentItem(id="b", content="y" * 600, priority=2),  # 150 tokens
        ]
        result = manager.allocate_budget(items, budget=250)

        assert result.tokens_used == 250
        assert result.tokens_available == 250

    def test_utilization_calculation(self, manager):
        """Test that utilization is calculated correctly."""
        items = [ContentItem(id="a", content="x" * 400, priority=1)]  # 100 tokens
        result = manager.allocate_budget(items, budget=200)

        assert result.utilization == 0.5  # 100 / 200

    def test_to_dict_includes_metadata(self, manager):
        """Test that to_dict includes all fidelity metadata."""
        items = [ContentItem(id="a", content="x" * 400, priority=1)]
        result = manager.allocate_budget(items, budget=50)

        d = result.to_dict()
        assert "fidelity" in d
        assert "tokens_used" in d
        assert "tokens_available" in d
        assert "utilization" in d
        assert "items" in d
        assert "allocation_ratio" in d["items"][0]


# =============================================================================
# Test: Allocation Strategies
# =============================================================================


class TestAllocationStrategies:
    """Tests for different allocation strategies."""

    @pytest.fixture
    def manager(self):
        """Create a ContextBudgetManager with fixed token estimation."""
        return ContextBudgetManager(
            token_estimator=lambda content: len(content) // 4
        )

    @pytest.fixture
    def items(self):
        """Create test items with known token counts."""
        return [
            ContentItem(id="a", content="A" * 400, priority=1),   # 100 tokens
            ContentItem(id="b", content="B" * 800, priority=2),   # 200 tokens
            ContentItem(id="c", content="C" * 1200, priority=3),  # 300 tokens
        ]

    def test_priority_first_allocates_by_priority(self, manager, items):
        """Test PRIORITY_FIRST allocates highest priority first."""
        result = manager.allocate_budget(
            items, budget=250, strategy=AllocationStrategy.PRIORITY_FIRST
        )

        # Item 'a' (priority 1) should get full allocation
        item_a = next(i for i in result.items if i.id == "a")
        assert not item_a.needs_summarization

    def test_equal_share_distributes_evenly(self, manager, items):
        """Test EQUAL_SHARE distributes budget equally."""
        result = manager.allocate_budget(
            items, budget=300, strategy=AllocationStrategy.EQUAL_SHARE
        )

        # Each item gets 100 tokens base share
        # Item 'a' (100 tokens) should fit fully
        item_a = next(i for i in result.items if i.id == "a")
        assert not item_a.needs_summarization

        # All items should be allocated (no drops in EQUAL_SHARE)
        assert len(result.items) == 3
        assert len(result.dropped_ids) == 0

    def test_proportional_maintains_ratios(self, manager, items):
        """Test PROPORTIONAL maintains size ratios."""
        result = manager.allocate_budget(
            items, budget=300, strategy=AllocationStrategy.PROPORTIONAL
        )

        # Total is 600 tokens, budget is 300 = 50% compression
        for item in result.items:
            # All items should be approximately 50% compressed
            assert 0.45 <= item.allocation_ratio <= 0.55


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def manager(self):
        """Create a ContextBudgetManager."""
        return ContextBudgetManager()

    def test_empty_items_list(self, manager):
        """Test allocation with empty items list."""
        result = manager.allocate_budget([], budget=1000)
        assert result.items == []
        assert result.tokens_used == 0
        assert result.fidelity == 1.0
        assert result.dropped_ids == []

    def test_invalid_budget_raises(self, manager):
        """Test that zero/negative budget raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            manager.allocate_budget([], budget=0)
        with pytest.raises(ValueError, match="positive"):
            manager.allocate_budget([], budget=-100)

    def test_allocation_result_validation(self):
        """Test AllocationResult validation."""
        with pytest.raises(ValueError, match="tokens_used"):
            AllocationResult(tokens_used=-1)
        with pytest.raises(ValueError, match="fidelity"):
            AllocationResult(fidelity=1.5)

    def test_allocated_item_ratio_calculation(self):
        """Test AllocatedItem calculates ratio correctly."""
        item = AllocatedItem(
            id="test",
            content="x",
            priority=1,
            original_tokens=100,
            allocated_tokens=50,
        )
        assert item.allocation_ratio == 0.5

    def test_allocated_item_zero_original(self):
        """Test AllocatedItem handles zero original tokens."""
        item = AllocatedItem(
            id="test",
            content="",
            priority=1,
            original_tokens=0,
            allocated_tokens=0,
        )
        assert item.allocation_ratio == 1.0

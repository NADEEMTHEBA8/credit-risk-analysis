"""
Unit tests for src/utils.py.

These are deliberately small and focused. The goal isn't 100% coverage
of the whole pipeline — it's to prove the pure-function helpers behave
correctly on edge cases. The pipeline itself is integration-tested
indirectly through the SHA-256 checksum approach (see README).

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src/ importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import missing_profile, reduce_memory, safe_divide  # noqa: E402


# ─── safe_divide ────────────────────────────────────────────────────────────

class TestSafeDivide:
    """safe_divide should never raise on divide-by-zero."""

    def test_basic_division(self):
        assert safe_divide(10, 2) == 5.0

    def test_zero_denominator_returns_fill(self):
        # The whole point of this function
        assert safe_divide(10, 0) == 0

    def test_zero_denominator_custom_fill(self):
        assert safe_divide(10, 0, fill=-1) == -1

    def test_array_division(self):
        a = np.array([10, 20, 30])
        b = np.array([2, 4, 5])
        np.testing.assert_array_almost_equal(
            safe_divide(a, b), np.array([5.0, 5.0, 6.0])
        )

    def test_array_with_zeros(self):
        a = np.array([10, 20, 30])
        b = np.array([2, 0, 5])
        result = safe_divide(a, b, fill=999)
        np.testing.assert_array_almost_equal(result, np.array([5.0, 999.0, 6.0]))

    def test_pandas_series(self):
        a = pd.Series([10, 20, 0])
        b = pd.Series([2, 0, 5])
        result = safe_divide(a, b)
        np.testing.assert_array_almost_equal(result, np.array([5.0, 0.0, 0.0]))


# ─── missing_profile ────────────────────────────────────────────────────────

class TestMissingProfile:

    def test_no_missing_returns_empty(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = missing_profile(df)
        assert len(result) == 0

    def test_returns_correct_counts(self):
        df = pd.DataFrame({
            "a": [1, 2, None],
            "b": [None, None, 3],
            "c": [1, 2, 3],
        })
        result = missing_profile(df)
        # 'c' has no missing values so it should not appear
        assert "a" in result.index
        assert "b" in result.index
        assert "c" not in result.index
        assert result.loc["a", "missing_count"] == 1
        assert result.loc["b", "missing_count"] == 2

    def test_sorted_descending_by_pct(self):
        df = pd.DataFrame({
            "a": [None, 2, 3],          # 33% missing
            "b": [None, None, None],    # 100% missing
            "c": [None, None, 3],       # 67% missing
        })
        result = missing_profile(df)
        # Should be ordered b (100), c (67), a (33)
        assert list(result.index) == ["b", "c", "a"]


# ─── reduce_memory ──────────────────────────────────────────────────────────

class TestReduceMemory:
    """Tests that dtype downcasting doesn't change values, just types."""

    def test_int64_gets_downcasted(self):
        df = pd.DataFrame({"a": [1, 2, 3]}, dtype="int64")
        result = reduce_memory(df.copy())
        # Small ints should fit in int8 or int16, not int64
        assert result["a"].dtype != np.int64

    def test_float64_gets_downcasted(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, dtype="float64")
        result = reduce_memory(df.copy())
        assert result["a"].dtype != np.float64

    def test_values_preserved(self):
        # The critical safety property: downcasting must not change values
        df = pd.DataFrame({
            "small_int": [1, 2, 3],
            "small_float": [1.5, 2.5, 3.5],
        })
        original = df.copy()
        result = reduce_memory(df)
        np.testing.assert_array_equal(
            result["small_int"].values, original["small_int"].values
        )
        np.testing.assert_array_almost_equal(
            result["small_float"].values, original["small_float"].values, decimal=5
        )

    def test_object_columns_untouched(self):
        df = pd.DataFrame({"text": ["a", "b", "c"], "num": [1, 2, 3]})
        result = reduce_memory(df.copy())
        assert result["text"].dtype == object


# ─── Configuration sanity ───────────────────────────────────────────────────

class TestConfiguration:
    """Catch typos and dead-letter constants."""

    def test_decision_beta_is_above_one(self):
        # beta > 1 weights recall over precision, which is what credit needs
        from src.utils import DECISION_BETA
        assert DECISION_BETA > 1.0
        assert DECISION_BETA == 2.5

    def test_missing_drop_threshold_in_valid_range(self):
        from src.utils import MISSING_DROP_PCT
        assert 0 < MISSING_DROP_PCT < 100

    def test_random_state_is_fixed_int(self):
        # Pipeline must be deterministic for the SHA checksum approach to work
        from src.utils import RANDOM_STATE
        assert isinstance(RANDOM_STATE, int)

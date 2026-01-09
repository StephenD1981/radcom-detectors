"""
Unit tests for error handling utilities.

Tests decorators and utilities for graceful error handling.
"""
import pytest
import pandas as pd
import numpy as np
from ran_optimizer.utils.error_handling import (
    require_columns,
    handle_empty_dataframe,
    safe_numeric_operation,
    validate_dataframe_not_empty,
    validate_columns_exist,
    safe_division,
)
from ran_optimizer.utils.exceptions import DataValidationError


class TestRequireColumnsDecorator:
    """Test require_columns decorator."""

    def test_passes_with_all_columns(self):
        """Should pass when all required columns exist."""
        @require_columns(['a', 'b', 'c'], df_param='df')
        def process_df(df):
            return len(df)

        df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        result = process_df(df)
        assert result == 1

    def test_raises_on_missing_columns(self):
        """Should raise DataValidationError when columns are missing."""
        @require_columns(['a', 'b', 'c'], df_param='df')
        def process_df(df):
            return len(df)

        df = pd.DataFrame({'a': [1], 'b': [2]})  # Missing 'c'

        with pytest.raises(DataValidationError) as exc_info:
            process_df(df)

        assert 'Missing required columns' in str(exc_info.value)
        assert 'c' in str(exc_info.value)

    def test_works_with_kwargs(self):
        """Should work with keyword arguments."""
        @require_columns(['x', 'y'], df_param='data')
        def process_data(data):
            return len(data)

        df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        result = process_data(data=df)
        assert result == 2


class TestHandleEmptyDataframe:
    """Test handle_empty_dataframe decorator."""

    def test_returns_default_for_empty_df(self):
        """Should return default value for empty DataFrame."""
        @handle_empty_dataframe(return_value=pd.DataFrame({'result': []}))
        def process_df(df):
            return df.copy()

        empty_df = pd.DataFrame()
        result = process_df(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert 'result' in result.columns

    def test_processes_non_empty_df(self):
        """Should process non-empty DataFrame normally."""
        @handle_empty_dataframe(return_value=None)
        def process_df(df):
            return len(df) * 2

        df = pd.DataFrame({'a': [1, 2, 3]})
        result = process_df(df)

        assert result == 6

    def test_returns_empty_df_by_default(self):
        """Should return empty DataFrame if no return_value specified."""
        @handle_empty_dataframe()
        def process_df(df):
            return df.copy()

        empty_df = pd.DataFrame()
        result = process_df(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestSafeNumericOperation:
    """Test safe_numeric_operation decorator."""

    def test_returns_result_for_valid_operation(self):
        """Should return result for valid numeric operation."""
        @safe_numeric_operation(default_value=0.0)
        def divide(a, b):
            return a / b

        result = divide(10, 2)
        assert result == 5.0

    def test_returns_default_on_division_by_zero(self):
        """Should return default value on division by zero."""
        @safe_numeric_operation(default_value=-1.0)
        def divide(a, b):
            return a / b

        result = divide(10, 0)
        assert result == -1.0

    def test_returns_default_on_nan(self):
        """Should return default value if result is NaN."""
        @safe_numeric_operation(default_value=0.0)
        def bad_calc():
            return float('nan')

        result = bad_calc()
        assert result == 0.0

    def test_returns_default_on_infinity(self):
        """Should return default value if result is infinity."""
        @safe_numeric_operation(default_value=0.0)
        def bad_calc():
            return float('inf')

        result = bad_calc()
        assert result == 0.0


class TestValidateDataframeNotEmpty:
    """Test validate_dataframe_not_empty function."""

    def test_passes_for_non_empty_df(self):
        """Should pass for non-empty DataFrame."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        validate_dataframe_not_empty(df, "test_df")  # Should not raise

    def test_raises_for_empty_df(self):
        """Should raise DataValidationError for empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(DataValidationError) as exc_info:
            validate_dataframe_not_empty(df, "test_df")

        assert 'test_df' in str(exc_info.value)
        assert 'empty' in str(exc_info.value).lower()


class TestValidateColumnsExist:
    """Test validate_columns_exist function."""

    def test_passes_when_all_columns_exist(self):
        """Should pass when all required columns exist."""
        df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        validate_columns_exist(df, {'a', 'b'}, "test_df")  # Should not raise

    def test_raises_when_columns_missing(self):
        """Should raise DataValidationError when columns are missing."""
        df = pd.DataFrame({'a': [1], 'b': [2]})

        with pytest.raises(DataValidationError) as exc_info:
            validate_columns_exist(df, {'a', 'b', 'c', 'd'}, "test_df")

        error_msg = str(exc_info.value)
        assert 'test_df' in error_msg
        assert 'missing required columns' in error_msg.lower()

    def test_error_message_shows_available_columns(self):
        """Should show available columns in error message."""
        df = pd.DataFrame({'a': [1], 'b': [2]})

        with pytest.raises(DataValidationError) as exc_info:
            validate_columns_exist(df, {'c', 'd'}, "test_df")

        error_msg = str(exc_info.value)
        assert 'Available columns' in error_msg
        assert 'a' in error_msg
        assert 'b' in error_msg


class TestSafeDivision:
    """Test safe_division function."""

    def test_valid_division(self):
        """Should perform valid division correctly."""
        result = safe_division(10.0, 2.0)
        assert result == 5.0

    def test_division_by_zero_returns_default(self):
        """Should return default on division by zero."""
        result = safe_division(10.0, 0.0, default=0.0)
        assert result == 0.0

    def test_custom_default_value(self):
        """Should use custom default value."""
        result = safe_division(10.0, 0.0, default=-999.0)
        assert result == -999.0

    def test_handles_nan_numerator(self):
        """Should return default for NaN numerator."""
        result = safe_division(float('nan'), 5.0, default=0.0)
        assert result == 0.0

    def test_handles_nan_denominator(self):
        """Should return default for NaN denominator."""
        result = safe_division(10.0, float('nan'), default=0.0)
        assert result == 0.0

    def test_handles_negative_numbers(self):
        """Should handle negative numbers correctly."""
        result = safe_division(-10.0, 2.0)
        assert result == -5.0

    def test_handles_very_small_denominator(self):
        """Should handle very small but non-zero denominator."""
        result = safe_division(1.0, 0.0001)
        assert result == 10000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

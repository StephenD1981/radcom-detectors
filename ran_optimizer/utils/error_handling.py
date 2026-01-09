"""
Error handling utilities for RAN optimization.

Provides decorators and utilities for graceful error handling and recovery.
"""
from functools import wraps
from typing import Set, List, Callable, Any
import pandas as pd
import numpy as np
from ran_optimizer.utils.logging_config import get_logger
from ran_optimizer.utils.exceptions import (
    DataValidationError,
    ProcessingError,
)

logger = get_logger(__name__)


def require_columns(required_cols: List[str], df_param: str = "df"):
    """
    Decorator to validate required columns exist in DataFrame.

    Parameters
    ----------
    required_cols : List[str]
        List of required column names
    df_param : str
        Name of the DataFrame parameter to check

    Raises
    ------
    DataValidationError
        If required columns are missing
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the DataFrame from kwargs or args
            df = kwargs.get(df_param)
            if df is None:
                # Try to find in positional args
                import inspect
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if df_param in param_names:
                    param_idx = param_names.index(df_param)
                    if param_idx < len(args):
                        df = args[param_idx]

            if df is None:
                raise DataValidationError(f"DataFrame parameter '{df_param}' not found")

            if not isinstance(df, pd.DataFrame):
                raise DataValidationError(
                    f"Parameter '{df_param}' must be a pandas DataFrame, got {type(df)}"
                )

            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise DataValidationError(
                    f"Missing required columns in {df_param}: {sorted(missing_cols)}. "
                    f"Available columns: {sorted(df.columns.tolist())}"
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def handle_empty_dataframe(return_value: Any = None):
    """
    Decorator to handle empty DataFrames gracefully.

    Parameters
    ----------
    return_value : Any
        Value to return if DataFrame is empty (default: empty DataFrame)

    Returns
    -------
    Callable
        Decorated function that returns return_value for empty DataFrames
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find the first DataFrame argument
            df = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df = arg
                    break

            if df is None:
                for arg in kwargs.values():
                    if isinstance(arg, pd.DataFrame):
                        df = arg
                        break

            if df is not None and len(df) == 0:
                logger.warning(
                    f"{func.__name__} called with empty DataFrame, returning early",
                    return_value=type(return_value).__name__ if return_value is not None else "None"
                )
                if return_value is None:
                    return pd.DataFrame()
                return return_value

            return func(*args, **kwargs)
        return wrapper
    return decorator


def safe_numeric_operation(default_value: float = 0.0):
    """
    Decorator to safely handle numeric operations that might fail.

    Parameters
    ----------
    default_value : float
        Value to return if operation fails

    Returns
    -------
    Callable
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                # Check for NaN or infinity
                if isinstance(result, (int, float)):
                    if pd.isna(result) or not np.isfinite(result):
                        logger.warning(
                            f"{func.__name__} returned invalid numeric value",
                            result=result,
                            returning=default_value
                        )
                        return default_value
                return result
            except (ZeroDivisionError, ValueError, TypeError) as e:
                logger.warning(
                    f"{func.__name__} numeric operation failed",
                    error=str(e),
                    returning=default_value
                )
                return default_value
        return wrapper
    return decorator


def validate_dataframe_not_empty(
    df: pd.DataFrame,
    name: str = "DataFrame"
) -> None:
    """
    Validate that DataFrame is not empty.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    name : str
        Name of DataFrame for error message

    Raises
    ------
    DataValidationError
        If DataFrame is empty
    """
    if len(df) == 0:
        raise DataValidationError(f"{name} is empty - no data to process")


def validate_columns_exist(
    df: pd.DataFrame,
    required_columns: Set[str],
    df_name: str = "DataFrame"
) -> None:
    """
    Validate that all required columns exist in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : Set[str]
        Set of required column names
    df_name : str
        Name of DataFrame for error message

    Raises
    ------
    DataValidationError
        If required columns are missing
    """
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise DataValidationError(
            f"{df_name} missing required columns: {sorted(missing_cols)}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


def log_dataframe_summary(
    df: pd.DataFrame,
    name: str,
    include_columns: bool = True
) -> None:
    """
    Log summary statistics for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to summarize
    name : str
        Name for logging
    include_columns : bool
        Whether to log column names
    """
    log_data = {
        "dataframe": name,
        "rows": len(df),
        "columns": len(df.columns),
    }

    if include_columns:
        log_data["column_names"] = df.columns.tolist()

    if len(df) > 0:
        log_data["memory_mb"] = df.memory_usage(deep=True).sum() / 1024 / 1024

    logger.debug("DataFrame summary", **log_data)


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default on error.

    Parameters
    ----------
    numerator : float
        Numerator
    denominator : float
        Denominator
    default : float
        Value to return on division by zero or error

    Returns
    -------
    float
        Result of division or default value
    """
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default

    try:
        result = numerator / denominator
        if not np.isfinite(result):
            return default
        return result
    except (ZeroDivisionError, ValueError, TypeError):
        return default

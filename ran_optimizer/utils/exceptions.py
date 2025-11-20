"""
Custom exception hierarchy for RAN Optimizer.

All custom exceptions inherit from RANOptimizerError for easy catching.
"""


class RANOptimizerError(Exception):
    """Base exception for all RAN Optimizer errors."""
    pass


class ConfigurationError(RANOptimizerError):
    """Configuration-related errors.

    Raised when configuration loading or validation fails.

    Example:
        >>> raise ConfigurationError("Invalid operator config: missing 'region' field")
    """
    pass


class DataValidationError(RANOptimizerError):
    """Data validation errors.

    Raised when input data fails validation checks.

    Attributes:
        invalid_rows: Number of rows that failed validation
        details: Dictionary with validation error details
    """

    def __init__(self, message: str, invalid_rows: int = 0, details: dict = None):
        super().__init__(message)
        self.invalid_rows = invalid_rows
        self.details = details or {}

    def __str__(self):
        base = super().__str__()
        if self.invalid_rows > 0:
            return f"{base} (invalid_rows={self.invalid_rows})"
        return base


class DataLoadError(RANOptimizerError):
    """Data loading errors.

    Raised when data files cannot be loaded or parsed.

    Example:
        >>> raise DataLoadError("Failed to load grid data: file not found")
    """
    pass


class ProcessingError(RANOptimizerError):
    """Processing pipeline errors.

    Raised when pipeline stages fail during execution.

    Attributes:
        stage: Name of the pipeline stage that failed
        details: Dictionary with error details
    """

    def __init__(self, message: str, stage: str = None, details: dict = None):
        super().__init__(message)
        self.stage = stage
        self.details = details or {}

    def __str__(self):
        base = super().__str__()
        if self.stage:
            return f"{base} (stage={self.stage})"
        return base


class AlgorithmError(RANOptimizerError):
    """Algorithm execution errors.

    Raised when recommendation algorithms fail.

    Example:
        >>> raise AlgorithmError("Overshooting detection failed: insufficient data")
    """
    pass


class GeometryError(RANOptimizerError):
    """Geometric calculation errors.

    Raised when geometry operations fail (e.g., invalid coordinates).

    Example:
        >>> raise GeometryError("Invalid coordinates: latitude out of range")
    """
    pass

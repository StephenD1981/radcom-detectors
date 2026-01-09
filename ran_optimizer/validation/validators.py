"""
Validation framework for RAN optimization recommendations.

Automatically flags unrealistic or suspicious results before deployment.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from ran_optimizer.utils.logging_config import get_logger
from ran_optimizer.utils.overshooting_config import (
    load_overshooting_config,
    get_environment_params,
    get_default_config_path,
)

logger = get_logger(__name__)


class IssueSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "CRITICAL"  # Invalid result, must be filtered
    WARNING = "WARNING"    # Suspicious result, review recommended
    INFO = "INFO"          # Informational flag


@dataclass
class ValidationIssue:
    """A single validation issue found in a recommendation."""
    cell_name: Any
    severity: IssueSeverity
    rule: str
    message: str
    actual_value: Any = None
    expected_range: str = None


@dataclass
class ValidationResult:
    """Result of validating a set of recommendations."""
    total_recommendations: int
    valid_recommendations: int
    filtered_recommendations: int
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        """Number of critical issues."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        """Number of warnings."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)

    @property
    def info_count(self) -> int:
        """Number of info flags."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.INFO)

    def log_summary(self):
        """Log a summary of validation results."""
        logger.info(
            "Validation complete",
            total=self.total_recommendations,
            valid=self.valid_recommendations,
            filtered=self.filtered_recommendations,
            critical=self.critical_count,
            warnings=self.warning_count,
            info=self.info_count
        )

        if self.critical_count > 0:
            logger.warning(
                f"Filtered {self.filtered_recommendations} invalid recommendations",
                critical_issues=self.critical_count
            )


class OvershootingValidator:
    """
    Validator for overshooting cell recommendations.

    Checks for unrealistic downtilt recommendations and flags suspicious results.
    """

    def __init__(
        self,
        max_downtilt_deg: float = 2.0,
        max_tilt_total_deg: float = 15.0,
        max_coverage_reduction_pct: float = 0.60,
        min_rsrp_reduction_db: float = 0.0,
        max_rsrp_reduction_db: float = 25.0,
    ):
        """
        Initialize validator with thresholds.

        Parameters
        ----------
        max_downtilt_deg : float
            Maximum allowed downtilt recommendation (degrees)
        max_tilt_total_deg : float
            Maximum allowed total tilt after downtilt (degrees)
        max_coverage_reduction_pct : float
            Maximum allowed coverage reduction (0-1 range)
        min_rsrp_reduction_db : float
            Minimum expected RSRP reduction (dB)
        max_rsrp_reduction_db : float
            Maximum realistic RSRP reduction (dB)
        """
        self.max_downtilt_deg = max_downtilt_deg
        self.max_tilt_total_deg = max_tilt_total_deg
        self.max_coverage_reduction_pct = max_coverage_reduction_pct
        self.min_rsrp_reduction_db = min_rsrp_reduction_db
        self.max_rsrp_reduction_db = max_rsrp_reduction_db

        logger.info(
            "Initialized OvershootingValidator",
            max_downtilt=max_downtilt_deg,
            max_total_tilt=max_tilt_total_deg,
            max_coverage_reduction=max_coverage_reduction_pct
        )

    @classmethod
    def from_config(
        cls,
        config_path: Optional[str] = None
    ) -> 'OvershootingValidator':
        """
        Create OvershootingValidator from JSON configuration file.

        Parameters
        ----------
        config_path : str, optional
            Path to overshooting_params.json. If None, uses default config.

        Returns
        -------
        OvershootingValidator
            Validator with thresholds loaded from config

        Examples
        --------
        >>> # Load with default config
        >>> validator = OvershootingValidator.from_config()

        >>> # Load with custom config
        >>> validator = OvershootingValidator.from_config('config/custom.json')
        """
        if config_path is None:
            config_path = get_default_config_path()

        logger.info("Loading validation config", config_file=str(config_path))
        config = load_overshooting_config(str(config_path))

        # Get default params (validation params don't have environment overrides)
        params_dict = get_environment_params(config, environment=None)

        # Extract validation parameters
        validation = params_dict.get('validation', {})

        return cls(
            max_downtilt_deg=validation.get('max_downtilt_deg', 2.0),
            max_tilt_total_deg=validation.get('max_tilt_total_deg', 15.0),
            max_coverage_reduction_pct=validation.get('max_coverage_reduction_pct', 0.60),
            min_rsrp_reduction_db=validation.get('min_rsrp_reduction_db', 0.0),
            max_rsrp_reduction_db=validation.get('max_rsrp_reduction_db', 25.0),
        )

    def validate(self, recommendations: pd.DataFrame) -> ValidationResult:
        """
        Validate overshooting recommendations.

        Parameters
        ----------
        recommendations : pd.DataFrame
            Overshooting recommendations to validate

        Returns
        -------
        ValidationResult
            Validation results with flagged issues
        """
        if len(recommendations) == 0:
            return ValidationResult(
                total_recommendations=0,
                valid_recommendations=0,
                filtered_recommendations=0
            )

        issues = []
        critical_cell_names = set()

        # Check each recommendation
        for idx, row in recommendations.iterrows():
            cell_name = row['cell_name']

            # Rule 1: Downtilt within realistic range
            downtilt = row.get('recommended_downtilt_deg', 0)
            if downtilt > self.max_downtilt_deg:
                issues.append(ValidationIssue(
                    cell_name=cell_name,
                    severity=IssueSeverity.CRITICAL,
                    rule="MAX_DOWNTILT",
                    message=f"Downtilt {downtilt:.1f}° exceeds maximum {self.max_downtilt_deg}°",
                    actual_value=downtilt,
                    expected_range=f"0-{self.max_downtilt_deg}°"
                ))
                critical_cell_names.add(cell_name)
            elif downtilt > 1.5:
                issues.append(ValidationIssue(
                    cell_name=cell_name,
                    severity=IssueSeverity.WARNING,
                    rule="HIGH_DOWNTILT",
                    message=f"Large downtilt {downtilt:.1f}° - review recommended",
                    actual_value=downtilt
                ))

            # Rule 2: Total tilt after downtilt within limits
            current_tilt = row.get('current_total_tilt_deg', 0)
            new_tilt = current_tilt + downtilt
            if new_tilt > self.max_tilt_total_deg:
                issues.append(ValidationIssue(
                    cell_name=cell_name,
                    severity=IssueSeverity.CRITICAL,
                    rule="MAX_TOTAL_TILT",
                    message=f"Total tilt {new_tilt:.1f}° exceeds maximum {self.max_tilt_total_deg}°",
                    actual_value=new_tilt,
                    expected_range=f"0-{self.max_tilt_total_deg}°"
                ))
                critical_cell_names.add(cell_name)

            # Rule 3: Coverage reduction within acceptable range
            coverage_reduction = row.get('coverage_reduction_percentage', 0)
            if coverage_reduction < 0:
                issues.append(ValidationIssue(
                    cell_name=cell_name,
                    severity=IssueSeverity.CRITICAL,
                    rule="NEGATIVE_COVERAGE_REDUCTION",
                    message=f"Negative coverage reduction {coverage_reduction:.1%}",
                    actual_value=coverage_reduction
                ))
                critical_cell_names.add(cell_name)
            elif coverage_reduction > self.max_coverage_reduction_pct:
                issues.append(ValidationIssue(
                    cell_name=cell_name,
                    severity=IssueSeverity.WARNING,
                    rule="HIGH_COVERAGE_REDUCTION",
                    message=f"High coverage reduction {coverage_reduction:.1%} - review impact",
                    actual_value=coverage_reduction
                ))

            # Rule 4: RSRP reduction within realistic range
            if 'avg_rsrp_reduction_db' in row:
                rsrp_reduction = row['avg_rsrp_reduction_db']
                if rsrp_reduction < self.min_rsrp_reduction_db:
                    issues.append(ValidationIssue(
                        cell_name=cell_name,
                        severity=IssueSeverity.WARNING,
                        rule="LOW_RSRP_REDUCTION",
                        message=f"Very low RSRP reduction {rsrp_reduction:.1f} dB",
                        actual_value=rsrp_reduction
                    ))
                elif rsrp_reduction > self.max_rsrp_reduction_db:
                    issues.append(ValidationIssue(
                        cell_name=cell_name,
                        severity=IssueSeverity.CRITICAL,
                        rule="EXCESSIVE_RSRP_REDUCTION",
                        message=f"Unrealistic RSRP reduction {rsrp_reduction:.1f} dB",
                        actual_value=rsrp_reduction,
                        expected_range=f"{self.min_rsrp_reduction_db}-{self.max_rsrp_reduction_db} dB"
                    ))
                    critical_cell_names.add(cell_name)

            # Rule 5: Severity score validity
            if 'severity_score' in row:
                severity_score = row['severity_score']
                if not (0.0 <= severity_score <= 1.0):
                    issues.append(ValidationIssue(
                        cell_name=cell_name,
                        severity=IssueSeverity.CRITICAL,
                        rule="INVALID_SEVERITY_SCORE",
                        message=f"Severity score {severity_score} outside valid range",
                        actual_value=severity_score,
                        expected_range="0.0-1.0"
                    ))
                    critical_cell_names.add(cell_name)

        # Filter critical recommendations
        valid_recommendations = recommendations[
            ~recommendations['cell_name'].isin(critical_cell_names)
        ]

        result = ValidationResult(
            total_recommendations=len(recommendations),
            valid_recommendations=len(valid_recommendations),
            filtered_recommendations=len(critical_cell_names),
            issues=issues
        )

        result.log_summary()
        return result


class UndershootingValidator:
    """
    Validator for undershooting cell recommendations.

    Checks for unrealistic uptilt recommendations and flags suspicious results.
    """

    def __init__(
        self,
        max_uptilt_deg: float = 2.0,
        min_tilt_total_deg: float = 0.0,
        max_coverage_increase_pct: float = 1.0,
        max_distance_increase_pct: float = 1.0,
        max_interference_pct: float = 0.40,
    ):
        """
        Initialize validator with thresholds.

        Parameters
        ----------
        max_uptilt_deg : float
            Maximum allowed uptilt recommendation (degrees)
        min_tilt_total_deg : float
            Minimum allowed total tilt (usually 0)
        max_coverage_increase_pct : float
            Maximum realistic coverage increase (0-1 range)
        max_distance_increase_pct : float
            Maximum realistic distance increase (0-1 range)
        max_interference_pct : float
            Maximum acceptable interference percentage
        """
        self.max_uptilt_deg = max_uptilt_deg
        self.min_tilt_total_deg = min_tilt_total_deg
        self.max_coverage_increase_pct = max_coverage_increase_pct
        self.max_distance_increase_pct = max_distance_increase_pct
        self.max_interference_pct = max_interference_pct

        logger.info(
            "Initialized UndershootingValidator",
            max_uptilt=max_uptilt_deg,
            max_coverage_increase=max_coverage_increase_pct,
            max_distance_increase=max_distance_increase_pct
        )

    @classmethod
    def from_config(
        cls,
        config_path: Optional[str] = None
    ) -> 'UndershootingValidator':
        """
        Create UndershootingValidator from JSON configuration file.

        Parameters
        ----------
        config_path : str, optional
            Path to undershooting_params.json. If None, uses default config.

        Returns
        -------
        UndershootingValidator
            Validator with thresholds loaded from config

        Examples
        --------
        >>> # Load with default config
        >>> validator = UndershootingValidator.from_config()

        >>> # Load with custom config
        >>> validator = UndershootingValidator.from_config('config/custom.json')
        """
        if config_path is None:
            config_path = str(Path(get_default_config_path()).parent / 'undershooting_params.json')

        logger.info("Loading validation config", config_file=str(config_path))
        config = load_overshooting_config(str(config_path))

        # Get default params (validation params don't have environment overrides)
        params_dict = get_environment_params(config, environment=None)

        # Extract validation parameters
        validation = params_dict.get('validation', {})

        return cls(
            max_uptilt_deg=validation.get('max_uptilt_deg', 3.0),
            min_tilt_total_deg=validation.get('min_tilt_total_deg', 0.0),
            max_coverage_increase_pct=validation.get('max_coverage_increase_pct', 1.0),
            max_distance_increase_pct=validation.get('max_distance_increase_pct', 1.0),
            max_interference_pct=validation.get('max_interference_pct', 0.40),
        )

    def validate(self, recommendations: pd.DataFrame) -> ValidationResult:
        """
        Validate undershooting recommendations.

        Parameters
        ----------
        recommendations : pd.DataFrame
            Undershooting recommendations to validate

        Returns
        -------
        ValidationResult
            Validation results with flagged issues
        """
        if len(recommendations) == 0:
            return ValidationResult(
                total_recommendations=0,
                valid_recommendations=0,
                filtered_recommendations=0
            )

        issues = []
        critical_cell_names = set()

        # Check each recommendation
        for idx, row in recommendations.iterrows():
            cell_name = row['cell_name']

            # Rule 1: Uptilt within realistic range
            uptilt = row.get('recommended_uptilt_deg', 0)
            if uptilt > self.max_uptilt_deg:
                issues.append(ValidationIssue(
                    cell_name=cell_name,
                    severity=IssueSeverity.CRITICAL,
                    rule="MAX_UPTILT",
                    message=f"Uptilt {uptilt:.1f}° exceeds maximum {self.max_uptilt_deg}°",
                    actual_value=uptilt,
                    expected_range=f"0-{self.max_uptilt_deg}°"
                ))
                critical_cell_names.add(cell_name)
            elif uptilt > 1.5:
                issues.append(ValidationIssue(
                    cell_name=cell_name,
                    severity=IssueSeverity.WARNING,
                    rule="HIGH_UPTILT",
                    message=f"Large uptilt {uptilt:.1f}° - review recommended",
                    actual_value=uptilt
                ))

            # Rule 2: Total tilt after uptilt within limits
            current_tilt = row.get('current_total_tilt', 0)
            new_tilt = current_tilt - uptilt  # Uptilt reduces tilt
            if new_tilt < self.min_tilt_total_deg:
                issues.append(ValidationIssue(
                    cell_name=cell_name,
                    severity=IssueSeverity.CRITICAL,
                    rule="NEGATIVE_TOTAL_TILT",
                    message=f"Total tilt {new_tilt:.1f}° below minimum {self.min_tilt_total_deg}°",
                    actual_value=new_tilt,
                    expected_range=f">={self.min_tilt_total_deg}°"
                ))
                critical_cell_names.add(cell_name)

            # Rule 3: Coverage increase within realistic range
            coverage_increase = row.get('coverage_increase_percentage', 0)
            if coverage_increase < 0:
                issues.append(ValidationIssue(
                    cell_name=cell_name,
                    severity=IssueSeverity.CRITICAL,
                    rule="NEGATIVE_COVERAGE_INCREASE",
                    message=f"Negative coverage increase {coverage_increase:.1%}",
                    actual_value=coverage_increase
                ))
                critical_cell_names.add(cell_name)
            elif coverage_increase > self.max_coverage_increase_pct:
                issues.append(ValidationIssue(
                    cell_name=cell_name,
                    severity=IssueSeverity.WARNING,
                    rule="EXCESSIVE_COVERAGE_INCREASE",
                    message=f"Unrealistic coverage increase {coverage_increase:.1%}",
                    actual_value=coverage_increase,
                    expected_range=f"0-{self.max_coverage_increase_pct*100:.0f}%"
                ))

            # Rule 4: Distance increase within realistic range
            if 'distance_increase_m' in row:
                distance_increase = row['distance_increase_m']
                current_distance = row.get('current_distance_m', 1)
                distance_increase_pct = distance_increase / current_distance if current_distance > 0 else 0

                if distance_increase < 0:
                    issues.append(ValidationIssue(
                        cell_name=cell_name,
                        severity=IssueSeverity.CRITICAL,
                        rule="NEGATIVE_DISTANCE_INCREASE",
                        message=f"Negative distance increase {distance_increase:.0f}m",
                        actual_value=distance_increase
                    ))
                    critical_cell_names.add(cell_name)
                elif distance_increase_pct > self.max_distance_increase_pct:
                    issues.append(ValidationIssue(
                        cell_name=cell_name,
                        severity=IssueSeverity.WARNING,
                        rule="EXCESSIVE_DISTANCE_INCREASE",
                        message=f"Very large distance increase {distance_increase_pct:.1%}",
                        actual_value=distance_increase_pct
                    ))

            # Rule 5: Interference within acceptable limits
            if 'interference_percentage' in row:
                interference = row['interference_percentage']
                if not (0.0 <= interference <= 1.0):
                    issues.append(ValidationIssue(
                        cell_name=cell_name,
                        severity=IssueSeverity.CRITICAL,
                        rule="INVALID_INTERFERENCE",
                        message=f"Interference {interference:.1%} outside valid range",
                        actual_value=interference,
                        expected_range="0-100%"
                    ))
                    critical_cell_names.add(cell_name)
                elif interference > self.max_interference_pct:
                    issues.append(ValidationIssue(
                        cell_name=cell_name,
                        severity=IssueSeverity.WARNING,
                        rule="HIGH_INTERFERENCE",
                        message=f"High interference {interference:.1%} - may cause issues",
                        actual_value=interference
                    ))

        # Filter critical recommendations
        valid_recommendations = recommendations[
            ~recommendations['cell_name'].isin(critical_cell_names)
        ]

        result = ValidationResult(
            total_recommendations=len(recommendations),
            valid_recommendations=len(valid_recommendations),
            filtered_recommendations=len(critical_cell_names),
            issues=issues
        )

        result.log_summary()
        return result

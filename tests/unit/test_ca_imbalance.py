"""
Unit tests for CA imbalance detection.
"""

import json
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, Point

from ran_optimizer.recommendations.ca_imbalance import (
    CAPairConfig,
    CAImbalanceParams,
    CAImbalanceDetector,
    detect_ca_imbalance,
    VALID_LTE_BANDS,
    VALID_NR_BANDS,
    VALID_BANDS,
)


class TestCAPairConfig:
    """Test CAPairConfig validation."""

    def test_valid_ca_pair(self):
        """Valid CA pair should be created successfully."""
        pair = CAPairConfig(
            name="L800-L1800",
            coverage_band="L800",
            capacity_band="L1800",
            coverage_threshold=0.70
        )
        assert pair.name == "L800-L1800"
        assert pair.coverage_band == "L800"
        assert pair.capacity_band == "L1800"
        assert pair.coverage_threshold == 0.70

    def test_same_band_rejected(self):
        """Same band for coverage and capacity should be rejected."""
        with pytest.raises(ValueError, match="must be different"):
            CAPairConfig(
                name="L800-L800",
                coverage_band="L800",
                capacity_band="L800"
            )

    def test_invalid_threshold_rejected(self):
        """Threshold outside (0, 1] should be rejected."""
        with pytest.raises(ValueError, match="coverage_threshold must be in"):
            CAPairConfig(
                name="L800-L1800",
                coverage_band="L800",
                capacity_band="L1800",
                coverage_threshold=1.5
            )

        with pytest.raises(ValueError, match="coverage_threshold must be in"):
            CAPairConfig(
                name="L800-L1800",
                coverage_band="L800",
                capacity_band="L1800",
                coverage_threshold=0.0
            )

    def test_default_threshold(self):
        """Default threshold should be 0.70."""
        pair = CAPairConfig(
            name="L800-L1800",
            coverage_band="L800",
            capacity_band="L1800"
        )
        assert pair.coverage_threshold == 0.70


class TestValidBands:
    """Test 3GPP band definitions."""

    def test_lte_bands_defined(self):
        """LTE bands should include common bands."""
        assert 'L800' in VALID_LTE_BANDS
        assert 'L1800' in VALID_LTE_BANDS
        assert 'L2100' in VALID_LTE_BANDS
        assert 'L2600' in VALID_LTE_BANDS

    def test_nr_bands_defined(self):
        """NR bands should include common bands."""
        assert 'N78' in VALID_NR_BANDS
        assert 'N77' in VALID_NR_BANDS
        assert 'N258' in VALID_NR_BANDS

    def test_valid_bands_is_union(self):
        """VALID_BANDS should be union of LTE and NR bands."""
        assert VALID_BANDS == VALID_LTE_BANDS | VALID_NR_BANDS


class TestCAImbalanceParams:
    """Test CAImbalanceParams validation and config loading."""

    @pytest.fixture
    def valid_config(self, tmp_path):
        """Create a valid config file."""
        config = {
            "ca_pairs": [
                {
                    "name": "L800-L1800",
                    "coverage_band": "L800",
                    "capacity_band": "L1800",
                    "coverage_threshold": 0.70
                }
            ],
            "cell_name_pattern": "(CK\\d+)[A-Z]+(\\d)"
        }
        config_path = tmp_path / "ca_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
        return config_path

    def test_from_config_valid(self, valid_config):
        """Valid config should load successfully."""
        params = CAImbalanceParams.from_config(str(valid_config))
        assert len(params.ca_pairs) == 1
        assert params.ca_pairs[0].name == "L800-L1800"
        assert params.cell_name_pattern == "(CK\\d+)[A-Z]+(\\d)"

    def test_from_config_missing_file(self):
        """Missing config file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CAImbalanceParams.from_config("/nonexistent/path.json")

    def test_from_config_invalid_json(self, tmp_path):
        """Invalid JSON should raise JSONDecodeError."""
        config_path = tmp_path / "invalid.json"
        with open(config_path, 'w') as f:
            f.write("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            CAImbalanceParams.from_config(str(config_path))

    def test_from_config_missing_ca_pairs(self, tmp_path):
        """Missing ca_pairs should raise ValueError."""
        config_path = tmp_path / "missing_pairs.json"
        with open(config_path, 'w') as f:
            json.dump({"cell_name_pattern": "pattern"}, f)

        with pytest.raises(ValueError, match="ca_pairs must be specified"):
            CAImbalanceParams.from_config(str(config_path))

    def test_from_config_missing_pattern(self, tmp_path):
        """Missing cell_name_pattern should raise ValueError."""
        config_path = tmp_path / "missing_pattern.json"
        with open(config_path, 'w') as f:
            json.dump({
                "ca_pairs": [{"name": "L800-L1800", "coverage_band": "L800", "capacity_band": "L1800"}]
            }, f)

        with pytest.raises(ValueError, match="cell_name_pattern must be specified"):
            CAImbalanceParams.from_config(str(config_path))

    def test_from_config_none_rejected(self):
        """None config path should raise ValueError."""
        with pytest.raises(ValueError, match="config_path is required"):
            CAImbalanceParams.from_config(None)

    def test_post_init_empty_ca_pairs(self):
        """Empty ca_pairs should be rejected."""
        with pytest.raises(ValueError, match="ca_pairs cannot be empty"):
            CAImbalanceParams(
                ca_pairs=[],
                cell_name_pattern="pattern"
            )

    def test_post_init_empty_pattern(self):
        """Empty cell_name_pattern should be rejected."""
        pair = CAPairConfig(name="L800-L1800", coverage_band="L800", capacity_band="L1800")
        with pytest.raises(ValueError, match="cell_name_pattern cannot be empty"):
            CAImbalanceParams(
                ca_pairs=[pair],
                cell_name_pattern=""
            )

    def test_post_init_invalid_regex(self):
        """Invalid regex pattern should be rejected."""
        pair = CAPairConfig(name="L800-L1800", coverage_band="L800", capacity_band="L1800")
        with pytest.raises(ValueError, match="not a valid regex"):
            CAImbalanceParams(
                ca_pairs=[pair],
                cell_name_pattern="[invalid("  # Unclosed bracket
            )

    def test_post_init_wrong_capture_groups(self):
        """Pattern with wrong number of capture groups should be rejected."""
        pair = CAPairConfig(name="L800-L1800", coverage_band="L800", capacity_band="L1800")
        # Pattern with only 1 capture group
        with pytest.raises(ValueError, match="exactly 2 capture groups"):
            CAImbalanceParams(
                ca_pairs=[pair],
                cell_name_pattern=r"(SITE\d+)"  # Only 1 group
            )
        # Pattern with 3 capture groups
        with pytest.raises(ValueError, match="exactly 2 capture groups"):
            CAImbalanceParams(
                ca_pairs=[pair],
                cell_name_pattern=r"(SITE)(\d+)([A-Z]+)"  # 3 groups
            )


class TestCAImbalanceDetector:
    """Test CAImbalanceDetector functionality."""

    @pytest.fixture
    def sample_params(self):
        """Create sample parameters for testing."""
        pair = CAPairConfig(
            name="L800-L1800",
            coverage_band="L800",
            capacity_band="L1800",
            coverage_threshold=0.70
        )
        # Pattern: (SITE\d+) captures site, S(\d) captures sector
        # E.g., SITE001L800S1 -> site_id=SITE001, sector=1
        return CAImbalanceParams(
            ca_pairs=[pair],
            cell_name_pattern=r"(SITE\d+)[A-Z0-9]+S(\d)"
        )

    @pytest.fixture
    def sample_hulls_gdf(self):
        """Create sample hulls GeoDataFrame for testing."""
        # Create two cells at same site/sector with different bands
        # L800 cell covers a larger area, L1800 covers a subset
        coverage_poly = Polygon([
            (0, 0), (0, 1000), (1000, 1000), (1000, 0), (0, 0)
        ])
        capacity_poly = Polygon([
            (0, 0), (0, 800), (800, 800), (800, 0), (0, 0)
        ])

        # Cell names: SITE001L800S1 and SITE001L1800S1
        # Both have site_id=SITE001 and sector=1
        data = {
            'cell_name': ['SITE001L800S1', 'SITE001L1800S1'],
            'band': ['L800', 'L1800'],
            'geometry': [coverage_poly, capacity_poly]
        }

        gdf = gpd.GeoDataFrame(data, crs='EPSG:32629')  # UTM zone 29N
        return gdf

    def test_initialization(self, sample_params):
        """Detector should initialize with valid params."""
        detector = CAImbalanceDetector(sample_params)
        assert detector.params == sample_params
        assert detector.target_crs is None

    def test_initialization_with_crs(self, sample_params):
        """Detector should accept custom CRS."""
        detector = CAImbalanceDetector(sample_params, target_crs='EPSG:32629')
        assert detector.target_crs == 'EPSG:32629'

    def test_detect_returns_list(self, sample_params, sample_hulls_gdf):
        """Detect should return a list of issues."""
        detector = CAImbalanceDetector(sample_params)
        issues = detector.detect(sample_hulls_gdf)
        assert isinstance(issues, list)

    def test_detect_missing_columns(self, sample_params):
        """Detect should raise ValueError for missing columns."""
        detector = CAImbalanceDetector(sample_params)
        incomplete_gdf = gpd.GeoDataFrame({'cell_name': ['A']})

        with pytest.raises(ValueError, match="Missing required columns"):
            detector.detect(incomplete_gdf)

    def test_detect_empty_after_filter(self, sample_params):
        """Detect should raise ValueError if empty after filtering."""
        detector = CAImbalanceDetector(sample_params)
        # GeoDataFrame with null geometries only
        gdf = gpd.GeoDataFrame({
            'cell_name': ['A'],
            'band': ['L800'],
            'geometry': [None]
        })

        with pytest.raises(ValueError, match="empty after filtering"):
            detector.detect(gdf)

    def test_detect_ca_imbalance_found(self, sample_params):
        """Should detect CA imbalance when capacity coverage is insufficient."""
        # Create cells where capacity band only covers 25% of coverage band
        # (500x500 / 1000x1000 = 0.25)
        coverage_poly = Polygon([
            (0, 0), (0, 1000), (1000, 1000), (1000, 0), (0, 0)
        ])
        capacity_poly = Polygon([
            (0, 0), (0, 500), (500, 500), (500, 0), (0, 0)
        ])

        gdf = gpd.GeoDataFrame({
            'cell_name': ['SITE001L800S1', 'SITE001L1800S1'],
            'band': ['L800', 'L1800'],
            'geometry': [coverage_poly, capacity_poly]
        }, crs='EPSG:32629')

        detector = CAImbalanceDetector(sample_params)
        issues = detector.detect(gdf)

        # Should find an issue (25% coverage < 70% threshold)
        assert len(issues) == 1
        assert issues[0]['severity'] in ['critical', 'high', 'medium', 'warning', 'low']
        assert issues[0]['coverage_ratio'] < 0.70

    def test_detect_no_imbalance_sufficient_coverage(self, sample_params):
        """Should not flag when capacity coverage is sufficient."""
        # Create cells where capacity band covers 90% of coverage band
        coverage_poly = Polygon([
            (0, 0), (0, 1000), (1000, 1000), (1000, 0), (0, 0)
        ])
        capacity_poly = Polygon([
            (0, 0), (0, 950), (950, 950), (950, 0), (0, 0)
        ])

        gdf = gpd.GeoDataFrame({
            'cell_name': ['SITE001L800S1', 'SITE001L1800S1'],
            'band': ['L800', 'L1800'],
            'geometry': [coverage_poly, capacity_poly]
        }, crs='EPSG:32629')

        detector = CAImbalanceDetector(sample_params)
        issues = detector.detect(gdf)

        # Should find no issues (90% coverage > 70% threshold)
        assert len(issues) == 0

    def test_detect_no_capacity_band(self, sample_params):
        """Should not flag sites without capacity band."""
        # Only coverage band present
        coverage_poly = Polygon([
            (0, 0), (0, 1000), (1000, 1000), (1000, 0), (0, 0)
        ])

        gdf = gpd.GeoDataFrame({
            'cell_name': ['SITE001L800S1'],
            'band': ['L800'],
            'geometry': [coverage_poly]
        }, crs='EPSG:32629')

        detector = CAImbalanceDetector(sample_params)
        issues = detector.detect(gdf)

        # No issues (no capacity band to compare)
        assert len(issues) == 0


class TestCellNameParsing:
    """Test cell name parsing validation."""

    def test_parse_failure_threshold(self):
        """Should fail if too many cells fail to parse."""
        pair = CAPairConfig(
            name="L800-L1800",
            coverage_band="L800",
            capacity_band="L1800"
        )
        params = CAImbalanceParams(
            ca_pairs=[pair],
            cell_name_pattern=r"(WRONG\d+)[A-Z0-9]+S(\d)",  # Won't match SITE format
            min_parse_success_ratio=0.5
        )

        # All cells use SITE format which won't match WRONG pattern
        coverage_poly = Polygon([(0, 0), (0, 1000), (1000, 1000), (1000, 0)])
        capacity_poly = Polygon([(0, 0), (0, 800), (800, 800), (800, 0)])

        gdf = gpd.GeoDataFrame({
            'cell_name': ['SITE001L800S1', 'SITE001L1800S1'],
            'band': ['L800', 'L1800'],
            'geometry': [coverage_poly, capacity_poly]
        }, crs='EPSG:32629')

        detector = CAImbalanceDetector(params)

        with pytest.raises(ValueError, match="only matched"):
            detector.detect(gdf)


class TestSeverityCalculation:
    """Test severity level calculation."""

    @pytest.fixture
    def detector(self):
        """Create detector with default severity thresholds."""
        pair = CAPairConfig(name="L800-L1800", coverage_band="L800", capacity_band="L1800")
        params = CAImbalanceParams(
            ca_pairs=[pair],
            cell_name_pattern=r"(SITE\d+)[A-Z0-9]+S(\d)"
        )
        return CAImbalanceDetector(params)

    def test_critical_severity(self, detector):
        """Coverage ratio < 0.30 should be critical."""
        assert detector._calculate_severity(0.20) == 'critical'

    def test_high_severity(self, detector):
        """Coverage ratio 0.30-0.50 should be high."""
        assert detector._calculate_severity(0.40) == 'high'

    def test_medium_severity(self, detector):
        """Coverage ratio 0.50-0.60 should be medium."""
        assert detector._calculate_severity(0.55) == 'medium'

    def test_warning_severity(self, detector):
        """Coverage ratio 0.60-0.70 should be warning."""
        assert detector._calculate_severity(0.65) == 'warning'

    def test_low_severity(self, detector):
        """Coverage ratio >= 0.70 should be low."""
        assert detector._calculate_severity(0.75) == 'low'


class TestConvenienceFunction:
    """Test detect_ca_imbalance convenience function."""

    def test_convenience_function_returns_dataframe(self):
        """Convenience function should return DataFrame."""
        pair = CAPairConfig(name="L800-L1800", coverage_band="L800", capacity_band="L1800")
        params = CAImbalanceParams(
            ca_pairs=[pair],
            cell_name_pattern=r"(SITE\d+)[A-Z0-9]+S(\d)"
        )

        coverage_poly = Polygon([(0, 0), (0, 1000), (1000, 1000), (1000, 0)])
        gdf = gpd.GeoDataFrame({
            'cell_name': ['SITE001L800S1'],
            'band': ['L800'],
            'geometry': [coverage_poly]
        }, crs='EPSG:32629')

        result = detect_ca_imbalance(gdf, params)
        assert isinstance(result, pd.DataFrame)

    def test_convenience_function_none_params_rejected(self):
        """Convenience function should reject None params."""
        coverage_poly = Polygon([(0, 0), (0, 1000), (1000, 1000), (1000, 0)])
        gdf = gpd.GeoDataFrame({
            'cell_name': ['SITE001L800S1'],
            'band': ['L800'],
            'geometry': [coverage_poly]
        }, crs='EPSG:32629')

        with pytest.raises(ValueError, match="params is required"):
            detect_ca_imbalance(gdf, None)


class TestInputValidation:
    """Test input validation edge cases."""

    @pytest.fixture
    def sample_params(self):
        """Create sample parameters."""
        pair = CAPairConfig(name="L800-L1800", coverage_band="L800", capacity_band="L1800")
        return CAImbalanceParams(
            ca_pairs=[pair],
            cell_name_pattern=r"(SITE\d+)[A-Z0-9]+S(\d)"
        )

    def test_invalid_geometry_types_filtered(self, sample_params):
        """Point and Line geometries should be filtered out."""
        detector = CAImbalanceDetector(sample_params)

        poly = Polygon([(0, 0), (0, 1000), (1000, 1000), (1000, 0)])
        point = Point(500, 500)

        gdf = gpd.GeoDataFrame({
            'cell_name': ['SITE001L800S1', 'SITE002L800S1'],
            'band': ['L800', 'L800'],
            'geometry': [poly, point]
        }, crs='EPSG:32629')

        # Should not raise - just filter out the point
        result = detector._validate_input(gdf)
        assert len(result) == 1  # Only polygon remains

    def test_zero_area_geometries_filtered(self, sample_params):
        """Zero-area (degenerate) polygons should be filtered out."""
        detector = CAImbalanceDetector(sample_params)

        valid_poly = Polygon([(0, 0), (0, 1000), (1000, 1000), (1000, 0)])
        # Degenerate polygon (line-like, zero area)
        zero_area_poly = Polygon([(0, 0), (100, 0), (200, 0), (0, 0)])

        gdf = gpd.GeoDataFrame({
            'cell_name': ['SITE001L800S1', 'SITE002L800S1'],
            'band': ['L800', 'L800'],
            'geometry': [valid_poly, zero_area_poly]
        }, crs='EPSG:32629')

        # Should not raise - just filter out the zero-area polygon
        result = detector._validate_input(gdf)
        assert len(result) == 1  # Only valid polygon remains

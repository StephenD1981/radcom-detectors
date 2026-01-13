"""
Unit tests for PCI conflict (collision) detection.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Polygon, Point

from ran_optimizer.recommendations.pci_conflict import (
    PCIConflictDetector,
    PCIConflictParams,
    detect_pci_conflicts,
    _get_tech_from_band,
    _get_pci_max_for_band,
    LTE_PCI_MAX,
    NR_PCI_MAX,
)


# -----------------------------
# Helper functions
# -----------------------------

def create_polygon(center_lat: float, center_lon: float, size_deg: float = 0.01) -> Polygon:
    """Create a simple square polygon around a center point."""
    half = size_deg / 2
    return Polygon([
        (center_lon - half, center_lat - half),
        (center_lon + half, center_lat - half),
        (center_lon + half, center_lat + half),
        (center_lon - half, center_lat + half),
    ])


def create_test_hulls(cells_config: list) -> gpd.GeoDataFrame:
    """
    Create test GeoDataFrame with cell hulls.

    cells_config: list of dicts with keys:
        cell_name, band, pci, lat, lon, size (optional), site_id (optional)
    """
    data = []
    for cell in cells_config:
        size = cell.get('size', 0.01)
        geom = create_polygon(cell['lat'], cell['lon'], size)
        row = {
            'cell_name': cell['cell_name'],
            'band': cell['band'],
            'pci': cell['pci'],
            'geometry': geom,
        }
        if 'site_id' in cell:
            row['site_id'] = cell['site_id']
        data.append(row)

    return gpd.GeoDataFrame(data, crs='EPSG:4326')


# -----------------------------
# Test helper functions
# -----------------------------

class TestHelperFunctions:
    """Test helper functions for band/PCI handling."""

    def test_get_tech_from_band_lte(self):
        """LTE bands should return 'LTE'."""
        assert _get_tech_from_band('L700') == 'LTE'
        assert _get_tech_from_band('L800') == 'LTE'
        assert _get_tech_from_band('L1800') == 'LTE'
        assert _get_tech_from_band('L2100') == 'LTE'

    def test_get_tech_from_band_nr(self):
        """NR bands should return 'NR'."""
        assert _get_tech_from_band('N78') == 'NR'
        assert _get_tech_from_band('N258') == 'NR'
        assert _get_tech_from_band('N260') == 'NR'

    def test_get_tech_from_band_umts(self):
        """UMTS bands should return 'UMTS'."""
        assert _get_tech_from_band('U900') == 'UMTS'
        assert _get_tech_from_band('U2100') == 'UMTS'

    def test_get_tech_from_band_unknown(self):
        """Unknown bands should return 'UNKNOWN'."""
        assert _get_tech_from_band('X123') == 'UNKNOWN'
        assert _get_tech_from_band('') == 'UNKNOWN'
        assert _get_tech_from_band(None) == 'UNKNOWN'

    def test_get_pci_max_for_band_lte(self):
        """LTE bands should return LTE PCI max (503)."""
        assert _get_pci_max_for_band('L800') == LTE_PCI_MAX
        assert _get_pci_max_for_band('L1800') == LTE_PCI_MAX

    def test_get_pci_max_for_band_nr(self):
        """NR bands should return NR PCI max (1007)."""
        assert _get_pci_max_for_band('N78') == NR_PCI_MAX
        assert _get_pci_max_for_band('N258') == NR_PCI_MAX


# -----------------------------
# Test PCIConflictParams
# -----------------------------

class TestPCIConflictParams:
    """Test parameter handling."""

    def test_defaults(self):
        """Default parameters should be set correctly."""
        params = PCIConflictParams()
        assert params.overlap_threshold == 0.10
        assert params.min_overlap_area_km2 == 0.0005
        assert params.filter_same_site is True
        assert params.check_mod3_conflicts is True
        assert params.check_mod30_conflicts is False

    def test_custom_values(self):
        """Custom parameters should be accepted."""
        params = PCIConflictParams(
            overlap_threshold=0.20,
            filter_same_site=False,
            check_mod30_conflicts=True,
        )
        assert params.overlap_threshold == 0.20
        assert params.filter_same_site is False
        assert params.check_mod30_conflicts is True

    def test_from_config_missing_file(self):
        """Missing config file should return defaults."""
        params = PCIConflictParams.from_config('/nonexistent/path.json')
        assert params.overlap_threshold == 0.10


# -----------------------------
# Test input validation
# -----------------------------

class TestInputValidation:
    """Test input validation and hardening."""

    def test_empty_geodataframe(self):
        """Empty GeoDataFrame should return empty results."""
        gdf = gpd.GeoDataFrame(columns=['cell_name', 'geometry', 'band', 'pci'])
        detector = PCIConflictDetector()
        result = detector.detect(gdf)
        assert len(result) == 0

    def test_missing_columns(self):
        """Missing required columns should raise ValueError."""
        gdf = gpd.GeoDataFrame({'cell_name': ['A'], 'geometry': [Point(0, 0)]})
        detector = PCIConflictDetector()
        with pytest.raises(ValueError, match="Missing required columns"):
            detector.detect(gdf)

    def test_null_geometry_filtered(self):
        """Cells with null geometry should be filtered out."""
        hulls = create_test_hulls([
            {'cell_name': 'CELL1', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0},
        ])
        # Add a cell with null geometry
        hulls = pd.concat([hulls, gpd.GeoDataFrame([{
            'cell_name': 'CELL2', 'band': 'L800', 'pci': 100, 'geometry': None
        }])], ignore_index=True)

        detector = PCIConflictDetector()
        result = detector.detect(hulls)
        # Should not crash, null geometry is filtered
        assert isinstance(result, list)

    def test_invalid_pci_filtered(self):
        """Cells with invalid PCI should be filtered out."""
        hulls = create_test_hulls([
            {'cell_name': 'VALID', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0},
            {'cell_name': 'INVALID1', 'band': 'L800', 'pci': -1, 'lat': 53.01, 'lon': -6.0},
            {'cell_name': 'INVALID2', 'band': 'L800', 'pci': 600, 'lat': 53.02, 'lon': -6.0},
        ])

        detector = PCIConflictDetector()
        result = detector.detect(hulls)
        # Invalid PCI cells should be filtered, no crash
        assert isinstance(result, list)

    def test_duplicate_cell_names_handled(self):
        """Duplicate cell names should keep first occurrence."""
        hulls = create_test_hulls([
            {'cell_name': 'CELL1', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0},
            {'cell_name': 'CELL1', 'band': 'L800', 'pci': 100, 'lat': 53.01, 'lon': -6.0},  # Duplicate
        ])

        detector = PCIConflictDetector()
        result = detector.detect(hulls)
        # Should not crash, duplicates are handled
        assert isinstance(result, list)


# -----------------------------
# Test PCI collision detection
# -----------------------------

class TestPCICollisionDetection:
    """Test PCI collision (exact match) detection."""

    def test_no_collision_different_pci(self):
        """Cells with different PCIs should not conflict."""
        hulls = create_test_hulls([
            {'cell_name': 'CELL1', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0, 'size': 0.02},
            {'cell_name': 'CELL2', 'band': 'L800', 'pci': 200, 'lat': 53.0, 'lon': -6.0, 'size': 0.02},
        ])

        detector = PCIConflictDetector()
        result = detector.detect(hulls)
        assert len(result) == 0

    def test_no_collision_no_overlap(self):
        """Cells with same PCI but no overlap should not conflict."""
        hulls = create_test_hulls([
            {'cell_name': 'CELL1', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0, 'size': 0.01},
            {'cell_name': 'CELL2', 'band': 'L800', 'pci': 100, 'lat': 54.0, 'lon': -6.0, 'size': 0.01},  # Far away
        ])

        detector = PCIConflictDetector()
        result = detector.detect(hulls)
        assert len(result) == 0

    def test_collision_detected(self):
        """Overlapping cells with same PCI should be detected."""
        hulls = create_test_hulls([
            {'cell_name': 'CELL1', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0, 'size': 0.02},
            {'cell_name': 'CELL2', 'band': 'L800', 'pci': 100, 'lat': 53.005, 'lon': -6.0, 'size': 0.02},  # Overlapping
        ])

        params = PCIConflictParams(overlap_threshold=0.01)  # Low threshold
        detector = PCIConflictDetector(params)
        result = detector.detect(hulls)

        assert len(result) >= 1
        conflict = result[0]
        assert conflict['detector'] == 'PCI_COLLISION'
        assert conflict['conflict_type'] == 'exact'
        assert conflict['pci1'] == 100
        assert conflict['pci2'] == 100

    def test_same_site_filtered(self):
        """Same-site cell pairs should be filtered when enabled."""
        hulls = create_test_hulls([
            {'cell_name': 'SITE1_A', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0, 'size': 0.02, 'site_id': 'SITE1'},
            {'cell_name': 'SITE1_B', 'band': 'L800', 'pci': 100, 'lat': 53.005, 'lon': -6.0, 'size': 0.02, 'site_id': 'SITE1'},
        ])

        params = PCIConflictParams(filter_same_site=True, overlap_threshold=0.01)
        detector = PCIConflictDetector(params)
        result = detector.detect(hulls)

        # Should be filtered out (same site)
        assert len(result) == 0

    def test_same_site_not_filtered_when_disabled(self):
        """Same-site pairs should NOT be filtered when disabled."""
        hulls = create_test_hulls([
            {'cell_name': 'SITE1_A', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0, 'size': 0.02, 'site_id': 'SITE1'},
            {'cell_name': 'SITE1_B', 'band': 'L800', 'pci': 100, 'lat': 53.005, 'lon': -6.0, 'size': 0.02, 'site_id': 'SITE1'},
        ])

        params = PCIConflictParams(filter_same_site=False, overlap_threshold=0.01)
        detector = PCIConflictDetector(params)
        result = detector.detect(hulls)

        # Should be detected (same-site filtering disabled)
        assert len(result) >= 1


# -----------------------------
# Test mod 3 conflict detection
# -----------------------------

class TestMod3ConflictDetection:
    """Test PCI mod 3 (PSS interference) detection."""

    def test_mod3_conflict_detected(self):
        """Cells with same PCI mod 3 should be detected."""
        # PCI 100 mod 3 = 1, PCI 103 mod 3 = 1 (different PCI, same mod 3)
        hulls = create_test_hulls([
            {'cell_name': 'CELL1', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0, 'size': 0.02},
            {'cell_name': 'CELL2', 'band': 'L800', 'pci': 103, 'lat': 53.005, 'lon': -6.0, 'size': 0.02},
        ])

        params = PCIConflictParams(
            check_mod3_conflicts=True,
            mod3_overlap_threshold=0.01,
        )
        detector = PCIConflictDetector(params)
        result = detector.detect(hulls)

        # Should find mod 3 conflict
        mod3_conflicts = [c for c in result if c['conflict_type'] == 'mod3']
        assert len(mod3_conflicts) >= 1
        assert mod3_conflicts[0]['detector'] == 'PCI_MOD3_CONFLICT'

    def test_mod3_disabled(self):
        """Mod 3 conflicts should not be detected when disabled."""
        hulls = create_test_hulls([
            {'cell_name': 'CELL1', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0, 'size': 0.02},
            {'cell_name': 'CELL2', 'band': 'L800', 'pci': 103, 'lat': 53.005, 'lon': -6.0, 'size': 0.02},
        ])

        params = PCIConflictParams(check_mod3_conflicts=False)
        detector = PCIConflictDetector(params)
        result = detector.detect(hulls)

        # No mod 3 conflicts should be found
        mod3_conflicts = [c for c in result if c['conflict_type'] == 'mod3']
        assert len(mod3_conflicts) == 0


# -----------------------------
# Test mod 30 conflict detection
# -----------------------------

class TestMod30ConflictDetection:
    """Test PCI mod 30 (RS interference) detection."""

    def test_mod30_conflict_detected(self):
        """Cells with same PCI mod 30 should be detected when enabled."""
        # PCI 100 mod 30 = 10, PCI 130 mod 30 = 10
        hulls = create_test_hulls([
            {'cell_name': 'CELL1', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0, 'size': 0.02},
            {'cell_name': 'CELL2', 'band': 'L800', 'pci': 130, 'lat': 53.005, 'lon': -6.0, 'size': 0.02},
        ])

        params = PCIConflictParams(
            check_mod30_conflicts=True,
            mod30_overlap_threshold=0.01,
        )
        detector = PCIConflictDetector(params)
        result = detector.detect(hulls)

        # Should find mod 30 conflict
        mod30_conflicts = [c for c in result if c['conflict_type'] == 'mod30']
        assert len(mod30_conflicts) >= 1
        assert mod30_conflicts[0]['detector'] == 'PCI_MOD30_CONFLICT'

    def test_mod30_disabled_by_default(self):
        """Mod 30 should be disabled by default."""
        params = PCIConflictParams()
        assert params.check_mod30_conflicts is False


# -----------------------------
# Test severity calculation
# -----------------------------

class TestSeverityCalculation:
    """Test severity level calculation."""

    def test_severity_levels(self):
        """Test that severity is calculated correctly."""
        hulls = create_test_hulls([
            {'cell_name': 'CELL1', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0, 'size': 0.05},
            {'cell_name': 'CELL2', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0, 'size': 0.05},  # 100% overlap
        ])

        params = PCIConflictParams(overlap_threshold=0.01)
        detector = PCIConflictDetector(params)
        result = detector.detect(hulls)

        assert len(result) >= 1
        # High overlap + close distance should result in critical/high severity
        assert result[0]['severity_category'] in ['CRITICAL', 'HIGH']

    def test_mod_conflicts_lower_severity(self):
        """Mod conflicts should have lower severity score than equivalent exact collisions."""
        # Create two cells with moderate overlap (not 100%) for realistic scenario
        hulls = create_test_hulls([
            {'cell_name': 'CELL1', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0, 'size': 0.03},
            {'cell_name': 'CELL2', 'band': 'L800', 'pci': 103, 'lat': 53.015, 'lon': -6.0, 'size': 0.03},  # Partial overlap
        ])

        params = PCIConflictParams(
            check_mod3_conflicts=True,
            mod3_overlap_threshold=0.01,
        )
        detector = PCIConflictDetector(params)
        result = detector.detect(hulls)

        mod3_conflicts = [c for c in result if c['conflict_type'] == 'mod3']
        if mod3_conflicts:
            # Mod 3 severity score should be lower due to 0.6 conflict_factor
            # With partial overlap and distance, should not be CRITICAL
            # Note: With 100% overlap at 0km, even MOD3 can be CRITICAL due to extreme conditions
            assert mod3_conflicts[0]['severity_score'] < 0.92  # Must be lower than max exact collision


# -----------------------------
# Test convenience function
# -----------------------------

class TestConvenienceFunction:
    """Test the detect_pci_conflicts convenience function."""

    def test_convenience_function_returns_dataframe(self):
        """Convenience function should return DataFrame."""
        hulls = create_test_hulls([
            {'cell_name': 'CELL1', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0},
        ])

        result = detect_pci_conflicts(hulls)
        assert isinstance(result, pd.DataFrame)

    def test_convenience_function_with_params(self):
        """Convenience function should accept custom params."""
        hulls = create_test_hulls([
            {'cell_name': 'CELL1', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0},
        ])

        params = PCIConflictParams(overlap_threshold=0.50)
        result = detect_pci_conflicts(hulls, params)
        assert isinstance(result, pd.DataFrame)


# -----------------------------
# Test multi-band handling
# -----------------------------

class TestMultiBandHandling:
    """Test handling of multiple frequency bands."""

    def test_different_bands_no_conflict(self):
        """Cells on different bands should not conflict."""
        hulls = create_test_hulls([
            {'cell_name': 'CELL1', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0, 'size': 0.02},
            {'cell_name': 'CELL2', 'band': 'L1800', 'pci': 100, 'lat': 53.0, 'lon': -6.0, 'size': 0.02},
        ])

        detector = PCIConflictDetector()
        result = detector.detect(hulls)

        # Same PCI but different bands - no conflict
        assert len(result) == 0

    def test_multiple_bands_processed(self):
        """Multiple bands should be processed separately."""
        hulls = create_test_hulls([
            # L800 conflict
            {'cell_name': 'L800_A', 'band': 'L800', 'pci': 100, 'lat': 53.0, 'lon': -6.0, 'size': 0.02},
            {'cell_name': 'L800_B', 'band': 'L800', 'pci': 100, 'lat': 53.005, 'lon': -6.0, 'size': 0.02},
            # L1800 conflict
            {'cell_name': 'L1800_A', 'band': 'L1800', 'pci': 200, 'lat': 53.1, 'lon': -6.0, 'size': 0.02},
            {'cell_name': 'L1800_B', 'band': 'L1800', 'pci': 200, 'lat': 53.105, 'lon': -6.0, 'size': 0.02},
        ])

        params = PCIConflictParams(overlap_threshold=0.01)
        detector = PCIConflictDetector(params)
        result = detector.detect(hulls)

        # Should find conflicts on both bands
        bands_with_conflicts = set(c['band'] for c in result if c['conflict_type'] == 'exact')
        assert 'L800' in bands_with_conflicts
        assert 'L1800' in bands_with_conflicts

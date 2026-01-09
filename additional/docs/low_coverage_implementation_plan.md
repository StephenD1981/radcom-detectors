# Low Coverage Detection - Implementation Plan

## Overview
Detect areas of low coverage where:
- Only 1 cell serves (single-server areas with no overlap from other cells)
- Average RSRP ≤ -115 dBm (configurable threshold)
- **Band-specific** (unlike no coverage which is band-agnostic)

## 1. Data Requirements

### Cell Hulls Data
Need to identify band information for each cell hull. Options:

**Option A: Band column in cell_hulls.csv**
```python
# Expected columns: cell_name, geometry, area_km2, band
# Example: CK002H1, POLYGON(...), 17.75, "Band 20"
```

**Option B: Join with GIS data**
```python
# cell_hulls has cell_name
# gis_data has cell_name + band/frequency
# Join on cell_name to get band info
hulls_with_band = cell_hulls.merge(
    gis_data[['cell_name', 'band']],
    on='cell_name',
    how='left'
)
```

**Option C: Extract from cell_name pattern**
```python
# If cell names follow pattern like "CK002H1_L20" (L20 = Band 20)
# Parse band from cell_name
```

### Grid Data
Need band-specific RSRP measurements:

```python
# Expected columns: geohash, lat, lon, avg_rsrp, band, cell_name
# Or multiple band columns: avg_rsrp_band20, avg_rsrp_band3, etc.
```

## 2. Architecture Design

### New Classes

```python
@dataclass
class LowCoverageParams:
    """Parameters for low coverage detection (band-specific)"""
    rsrp_threshold_dbm: float = -115
    k_ring_steps: int = 3
    min_missing_neighbors: int = 40
    hdbscan_min_cluster_size: int = 10
    alpha_shape_alpha: Optional[float] = None
    max_alphashape_points: int = 5000

    @classmethod
    def from_config(cls, config_path: Path, environment: str = "suburban") -> "LowCoverageParams":
        """Load from config file with environment overrides"""
        with open(config_path) as f:
            config = json.load(f)

        base_params = config.get("low_coverage_detection", {})
        env_overrides = config.get("environment_overrides", {}).get(environment, {})

        # Merge base + overrides
        params = {**base_params, **env_overrides}

        return cls(
            rsrp_threshold_dbm=params.get("rsrp_threshold_dbm", -115),
            k_ring_steps=params.get("k_ring_steps", 3),
            min_missing_neighbors=params.get("min_missing_neighbors", 40),
            hdbscan_min_cluster_size=params.get("hdbscan_min_cluster_size", 10),
            alpha_shape_alpha=params.get("alpha_shape_alpha"),
            max_alphashape_points=params.get("max_alphashape_points", 5000)
        )


class LowCoverageDetector:
    """Detect low coverage areas per band"""

    def __init__(self, params: LowCoverageParams):
        self.params = params
        self.logger = get_logger(__name__)

    def detect(
        self,
        hulls: gpd.GeoDataFrame,
        grid_data: pd.DataFrame,
        gis_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Detect low coverage clusters per band.

        Args:
            hulls: Cell convex hulls (requires 'band' column or join with gis_data)
            grid_data: Grid measurements with RSRP per band
            gis_data: Optional GIS data to join for band info

        Returns:
            Dict mapping band names to GeoDataFrames of low coverage clusters
            e.g., {'Band 20': gdf, 'Band 3': gdf}
        """
        # Step 1: Ensure hulls have band information
        hulls_with_band = self._add_band_info(hulls, gis_data)

        # Step 2: Process each band separately
        band_results = {}
        for band in hulls_with_band['band'].unique():
            if pd.isna(band):
                continue

            self.logger.info("processing_band", band=band)
            band_clusters = self._detect_band_low_coverage(
                hulls_with_band[hulls_with_band['band'] == band],
                grid_data,
                band
            )

            if len(band_clusters) > 0:
                band_results[band] = band_clusters

        return band_results

    def _add_band_info(
        self,
        hulls: gpd.GeoDataFrame,
        gis_data: Optional[pd.DataFrame]
    ) -> gpd.GeoDataFrame:
        """Add band information to hulls if not already present"""
        if 'band' in hulls.columns:
            return hulls

        if gis_data is None:
            raise ValueError("hulls missing 'band' column and no gis_data provided")

        # Join with GIS data to get band
        return hulls.merge(
            gis_data[['cell_name', 'band']],
            on='cell_name',
            how='left'
        )

    def _detect_band_low_coverage(
        self,
        band_hulls: gpd.GeoDataFrame,
        grid_data: pd.DataFrame,
        band: str
    ) -> gpd.GeoDataFrame:
        """Detect low coverage for a specific band"""
        # Step 1: Find single-server regions (no overlap)
        single_server_polygons = self._find_single_server_regions(band_hulls)

        # Step 2: Get geohashes in single-server regions
        candidate_geohashes = self._geohashes_in_polygons(single_server_polygons)

        # Step 3: Filter by RSRP threshold
        low_rsrp_geohashes = self._filter_by_rsrp(
            candidate_geohashes,
            grid_data,
            band
        )

        # Step 4: Apply k-ring density filtering
        dense_gaps = self._compute_kring_density(low_rsrp_geohashes)

        # Step 5: Cluster with HDBSCAN
        clustered_gaps = self._cluster_hdbscan(dense_gaps)

        # Step 6: Create alpha shape polygons
        cluster_polygons = self._create_cluster_polygons(clustered_gaps)

        # Step 7: Add band label
        cluster_polygons['band'] = band

        return cluster_polygons

    def _find_single_server_regions(
        self,
        band_hulls: gpd.GeoDataFrame
    ) -> List[Polygon]:
        """
        Find regions where only one cell provides coverage (no overlap).

        For each cell hull, subtract all overlapping areas from other cells.
        """
        single_server_regions = []

        for idx, hull_row in band_hulls.iterrows():
            cell_name = hull_row['cell_name']
            hull_geom = hull_row['geometry']

            # Find all other hulls that overlap
            other_hulls = band_hulls[band_hulls['cell_name'] != cell_name]
            overlapping = other_hulls[other_hulls.intersects(hull_geom)]

            if len(overlapping) == 0:
                # Entire hull is single-server
                single_server_regions.append(hull_geom)
            else:
                # Subtract overlapping areas
                overlap_union = unary_union(overlapping.geometry)
                single_server = hull_geom.difference(overlap_union)

                # Handle MultiPolygon results
                if isinstance(single_server, Polygon) and single_server.area > 0:
                    single_server_regions.append(single_server)
                elif isinstance(single_server, MultiPolygon):
                    single_server_regions.extend([p for p in single_server.geoms if p.area > 0])

        return single_server_regions

    def _filter_by_rsrp(
        self,
        geohashes: Set[str],
        grid_data: pd.DataFrame,
        band: str
    ) -> Set[str]:
        """Filter geohashes by RSRP threshold for specific band"""
        # Filter grid_data to this band
        band_grid = grid_data[grid_data['band'] == band]

        # Filter by RSRP threshold
        low_rsrp = band_grid[band_grid['avg_rsrp'] <= self.params.rsrp_threshold_dbm]

        # Intersect with candidate geohashes
        low_rsrp_geohashes = set(low_rsrp['geohash'].unique())

        return geohashes.intersection(low_rsrp_geohashes)

    # Reuse methods from CoverageGapDetector:
    # - _geohashes_in_polygons()
    # - _compute_kring_density()
    # - _cluster_hdbscan()
    # - _create_cluster_polygons()
```

## 3. Algorithm Detailed Steps

### Step 1: Find Single-Server Regions

```
For each cell hull H in band B:
    1. Get all other hulls in same band: others = hulls_B - {H}
    2. Find overlapping hulls: overlaps = [O for O in others if O.intersects(H)]
    3. If no overlaps:
        single_server_region = H
    4. Else:
        overlap_union = union(all overlapping geometries)
        single_server_region = H - overlap_union
    5. Add single_server_region to results (may be MultiPolygon)
```

**Example**:
- Cell A (Band 20) covers area [10 km²]
- Cell B (Band 20) overlaps 3 km² with Cell A
- Single-server region for Cell A = 7 km² (the non-overlapping part)

### Step 2: Get Candidate Geohashes

```
For each single_server_polygon:
    1. Get bounds: minx, miny, maxx, maxy
    2. Calculate step size for precision 7 (~153m)
    3. Generate grid of lat/lon points
    4. Convert to geohashes
    5. Filter to points actually inside polygon
    6. Add to candidate set
```

### Step 3: Filter by RSRP

```
1. Filter grid_data to current band: band_grid = grid_data[grid_data['band'] == band]
2. Filter by threshold: low_rsrp = band_grid[band_grid['avg_rsrp'] <= -115]
3. Get geohashes: low_rsrp_geohashes = set(low_rsrp['geohash'])
4. Intersect: valid_geohashes = candidates ∩ low_rsrp_geohashes
```

**Key**: This filtering happens BEFORE k-ring, so we only consider areas that are both:
- Single-server (no overlap)
- Poor signal (RSRP ≤ -115 dBm)

### Step 4-6: Same as No Coverage

Apply same k-ring density → HDBSCAN clustering → alpha shape polygon creation

## 4. Integration with Existing Code

### Refactor Shared Methods

Both `CoverageGapDetector` and `LowCoverageDetector` need:
- `_geohashes_in_polygons()`
- `_compute_kring_density()`
- `_cluster_hdbscan()`
- `_create_cluster_polygons()`

**Solution**: Extract to base class or utility module

```python
class GapDetectorBase:
    """Base class with shared gap detection methods"""

    def _geohashes_in_polygons(self, polygons: List[Polygon]) -> Set[str]:
        """Shared implementation"""
        pass

    def _compute_kring_density(self, geohashes: Set[str]) -> Set[str]:
        """Shared implementation"""
        pass

    def _cluster_hdbscan(self, geohashes: Set[str]) -> pd.DataFrame:
        """Shared implementation"""
        pass

    def _create_cluster_polygons(self, clustered: pd.DataFrame) -> gpd.GeoDataFrame:
        """Shared implementation"""
        pass


class CoverageGapDetector(GapDetectorBase):
    """Detect no coverage areas (band-agnostic)"""
    pass


class LowCoverageDetector(GapDetectorBase):
    """Detect low coverage areas (band-specific)"""
    pass
```

### Test Updates

```python
def test_coverage_and_low_coverage():
    """Test both no coverage and low coverage detection"""

    # Load data
    grid_data = load_grid_data(grid_file, sample_rows=50000, decode_geohash=True)
    gis_data = load_gis_data(gis_file)
    cell_hulls = load_cell_hulls(hulls_file)

    # Detect NO COVERAGE (band-agnostic)
    no_cov_params = CoverageGapParams.from_config(config_path, environment="suburban")
    no_cov_detector = CoverageGapDetector(no_cov_params)
    no_coverage_clusters = no_cov_detector.detect(cell_hulls)

    # Detect LOW COVERAGE (band-specific)
    low_cov_params = LowCoverageParams.from_config(config_path, environment="suburban")
    low_cov_detector = LowCoverageDetector(low_cov_params)
    low_coverage_by_band = low_cov_detector.detect(cell_hulls, grid_data, gis_data)

    # Visualize both
    create_coverage_map(no_coverage_clusters, low_coverage_by_band, cell_hulls, gis_data)
```

## 5. Map Visualization Strategy

### Layer Structure

```python
# Base map
m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

# Layer 1: Cell Coverage Hulls (grey, transparent)
hulls_layer = folium.FeatureGroup(name="Cell Coverage Hulls", show=True)
# ... add grey hulls with fillOpacity=0.15

# Layer 2: No Coverage Clusters (RED)
no_cov_layer = folium.FeatureGroup(name="No Coverage Areas", show=True)
for _, cluster in no_coverage_clusters.iterrows():
    folium.GeoJson(
        cluster['geometry'],
        style_function=lambda x: {
            'fillColor': 'red',
            'color': 'darkred',
            'weight': 2,
            'fillOpacity': 0.4
        },
        popup=f"No Coverage - Cluster {cluster['cluster_id']}"
    ).add_to(no_cov_layer)

# Layer 3+: Low Coverage per Band (different colors)
band_colors = {
    'Band 20': 'orange',
    'Band 3': 'yellow',
    'Band 7': 'purple',
    'Band 1': 'pink'
}

for band, clusters in low_coverage_by_band.items():
    band_layer = folium.FeatureGroup(name=f"Low Coverage - {band}", show=True)
    color = band_colors.get(band, 'orange')

    for _, cluster in clusters.iterrows():
        folium.GeoJson(
            cluster['geometry'],
            style_function=lambda x, c=color: {
                'fillColor': c,
                'color': c,
                'weight': 2,
                'fillOpacity': 0.3
            },
            popup=f"Low Coverage - {band}<br>Cluster {cluster['cluster_id']}<br>"
                  f"RSRP ≤ {low_cov_params.rsrp_threshold_dbm} dBm"
        ).add_to(band_layer)

    band_layer.add_to(m)

# Layer N: Cell Sites (blue markers)
sites_layer = folium.FeatureGroup(name="Cell Sites", show=True)
# ... add blue circle markers

# Add layer control
folium.LayerControl().add_to(m)
```

### Legend

Add custom legend showing color coding:
- 🔴 Red = No Coverage (any band)
- 🟠 Orange = Band 20 Low Coverage (RSRP ≤ -115 dBm)
- 🟡 Yellow = Band 3 Low Coverage (RSRP ≤ -115 dBm)
- ⬜ Grey = Cell Coverage Footprints

## 6. Edge Cases & Considerations

### Edge Case 1: Missing Band Information

**Problem**: Some cells don't have band info in GIS data

**Solution**:
```python
# Log warning and skip cells without band
if 'band' not in hulls.columns and gis_data is None:
    logger.warning("cannot_detect_low_coverage_without_band_info")
    return {}

hulls_with_band = hulls.merge(gis_data[['cell_name', 'band']], on='cell_name', how='left')
missing_band = hulls_with_band['band'].isna().sum()
if missing_band > 0:
    logger.warning("cells_missing_band_info", count=missing_band)
```

### Edge Case 2: Grid Data Band Format

**Problem**: Grid data may have different band column formats

**Options**:
```python
# Option A: Single 'band' column with multiple rows per geohash
# geohash,lat,lon,avg_rsrp,band,cell_name
# gc7x3r4,51.90,-8.47,-95,Band 20,CK002H1
# gc7x3r4,51.90,-8.47,-102,Band 3,CK003H1

# Option B: Multiple RSRP columns (one per band)
# geohash,lat,lon,avg_rsrp_band20,avg_rsrp_band3,serving_cell_band20,serving_cell_band3
# gc7x3r4,51.90,-8.47,-95,-102,CK002H1,CK003H1
```

**Solution**: Auto-detect format
```python
def _get_band_rsrp(self, grid_data: pd.DataFrame, band: str) -> pd.Series:
    """Get RSRP values for specific band, handling different formats"""
    if 'band' in grid_data.columns:
        # Option A: Filter by band column
        return grid_data[grid_data['band'] == band]['avg_rsrp']
    else:
        # Option B: Use band-specific column
        col_name = f'avg_rsrp_{band.lower().replace(" ", "")}'
        if col_name in grid_data.columns:
            return grid_data[col_name]
        else:
            raise ValueError(f"Cannot find RSRP data for {band}")
```

### Edge Case 3: Performance with Many Bands

**Problem**: If 10+ bands, could be slow

**Solution**:
- Process bands in parallel using multiprocessing
- Add progress logging
- Allow filtering to specific bands

```python
def detect(
    self,
    hulls: gpd.GeoDataFrame,
    grid_data: pd.DataFrame,
    gis_data: Optional[pd.DataFrame] = None,
    bands: Optional[List[str]] = None  # NEW: Filter to specific bands
) -> Dict[str, gpd.GeoDataFrame]:
    """Detect low coverage, optionally filtering to specific bands"""
    hulls_with_band = self._add_band_info(hulls, gis_data)

    available_bands = hulls_with_band['band'].unique()
    if bands is not None:
        available_bands = [b for b in available_bands if b in bands]

    self.logger.info("detecting_low_coverage", bands=list(available_bands))

    # ... process each band
```

### Edge Case 4: Cell Completely Covered by Others

**Problem**: Some cells may be completely overlapped by other cells

**Solution**: Skip cells where single_server_region is empty
```python
if isinstance(single_server, Polygon) and single_server.area > 0:
    single_server_regions.append(single_server)
elif single_server.is_empty:
    logger.debug("cell_completely_overlapped", cell_name=cell_name)
    continue
```

## 7. Implementation Checklist

- [ ] Refactor shared methods to `GapDetectorBase` class
- [ ] Implement `LowCoverageParams` dataclass with config loading
- [ ] Implement `LowCoverageDetector` class
  - [ ] `_add_band_info()` method
  - [ ] `_find_single_server_regions()` method
  - [ ] `_filter_by_rsrp()` method with auto-detection of band format
  - [ ] `detect()` orchestration method
- [ ] Update test file
  - [ ] Add low coverage detection
  - [ ] Update map visualization with band colors
  - [ ] Add legend
- [ ] Test with real data
  - [ ] Verify band information available
  - [ ] Check grid_data format for RSRP
  - [ ] Validate results make sense
- [ ] Add documentation
  - [ ] Update README with low coverage section
  - [ ] Add docstrings with examples
  - [ ] Create usage guide

## 8. Expected Results

After implementation, running the test should produce:

```
=== Coverage Detection Results ===
No Coverage Clusters:       2 (band-agnostic)
  Cluster 0: 98 grids at (51.9022, -9.3336)
  Cluster 1: 46 grids at (51.9079, -9.3499)

Low Coverage Clusters:
  Band 20: 3 clusters
    Cluster 0: 156 grids (avg RSRP: -117 dBm)
    Cluster 1: 89 grids (avg RSRP: -119 dBm)
    Cluster 2: 45 grids (avg RSRP: -116 dBm)

  Band 3: 1 cluster
    Cluster 0: 67 grids (avg RSRP: -118 dBm)

Map saved: data/vf-ie/output-data/maps/coverage_analysis_map.html
```

Map should show:
- Red polygons for no coverage
- Orange polygons for Band 20 low coverage
- Yellow polygons for Band 3 low coverage
- Grey transparent hulls for cell coverage
- Blue markers for cell sites
- Layer control to toggle visibility

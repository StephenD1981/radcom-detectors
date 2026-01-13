"""
Crossed Feeder Detection.

Detects likely crossed feeders (sector/feeder swaps) by identifying:
  1) Reciprocal swap patterns between cells at the same site+band (HIGH confidence)
  2) Multiple cells at same site+band with out-of-beam traffic (MEDIUM confidence)
  3) Single cells with out-of-beam traffic - likely azimuth issues (LOW confidence)

A true crossed feeder shows a SWAP PATTERN:
  - Cell A's traffic goes toward Cell B's azimuth direction
  - Cell B's traffic goes toward Cell A's azimuth direction
  - Both cells are at the same site on the same band
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)


# -----------------------------
# Geometry helpers
# -----------------------------

def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Initial bearing from (lat1,lon1) to (lat2,lon2), in degrees [0,360).
    Uses great-circle bearing formula.
    """
    phi1 = _deg2rad(lat1)
    phi2 = _deg2rad(lat2)
    dlon = _deg2rad(lon2 - lon1)

    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)

    brng = math.atan2(y, x)
    brng_deg = (brng * 180.0 / math.pi + 360.0) % 360.0
    return brng_deg


def bearing_deg_vec(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized version of bearing_deg."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)

    y = np.sin(dlon) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)

    brng = np.arctan2(y, x)
    return (np.degrees(brng) + 360.0) % 360.0


def circ_diff_deg(a: float, b: float) -> float:
    """
    Smallest circular difference between angles a and b in degrees, result in [0, 180].
    """
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def circ_diff_deg_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized version of circ_diff_deg."""
    d = np.abs((a - b) % 360.0)
    return np.minimum(d, 360.0 - d)


def is_in_beam(azimuth: float, target_angle: float, half_width: float) -> bool:
    """
    Checks if target_angle lies within azimuth +/- half_width (with circular wrap).
    """
    return circ_diff_deg(azimuth, target_angle) <= half_width


def weighted_circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate weighted circular mean of angles in degrees.
    Returns angle in [0, 360), or NaN if inputs are invalid.
    """
    if len(angles) == 0 or len(weights) == 0:
        return np.nan

    # Filter out NaN values from both arrays
    valid_mask = ~np.isnan(angles) & ~np.isnan(weights)
    if not valid_mask.any():
        return np.nan

    valid_angles = angles[valid_mask]
    valid_weights = weights[valid_mask]

    if valid_weights.sum() == 0:
        return np.nan

    # Convert to radians
    rad = np.radians(valid_angles)

    # Weighted sum of unit vectors
    x = np.sum(valid_weights * np.cos(rad))
    y = np.sum(valid_weights * np.sin(rad))

    # Convert back to degrees
    mean_rad = np.arctan2(y, x)
    mean_deg = (np.degrees(mean_rad) + 360.0) % 360.0

    return mean_deg


# -----------------------------
# Configuration
# -----------------------------

# Band-specific maximum radius thresholds (meters)
BAND_MAX_RADIUS_M = {
    'L700': 32000,
    'L800': 30000,
    'L900': 28000,
    'L1800': 25000,
    'L2100': 20000,
    'L2600': 15000,
    'N78': 10000,
    'N258': 2000,
}
DEFAULT_MAX_RADIUS_M = 32000

# Technology prefixes for same-tech filtering
TECH_PREFIXES = {
    'L': 'LTE',
    'N': 'NR',
    'U': 'UMTS',
    'G': 'GSM',
}


def _get_tech_from_band(band: str) -> str:
    """Extract technology from band string (e.g., 'L800' -> 'LTE')."""
    if not band or not isinstance(band, str):
        return 'UNKNOWN'
    band_upper = band.upper().strip()
    if not band_upper:
        return 'UNKNOWN'
    prefix = band_upper[0]
    return TECH_PREFIXES.get(prefix, 'UNKNOWN')


@dataclass
class CrossedFeederParams:
    """Parameters for crossed feeder detection."""
    # Distance filters
    max_radius_m: float = 32000.0
    min_distance_m: float = 500.0
    band_max_radius_m: Optional[Dict[str, float]] = None

    # Beamwidth parameters
    hbw_cap_deg: float = 60.0
    min_hbw_deg: float = 1.0
    max_hbw_deg: float = 179.0
    beamwidth_expansion_factor: float = 1.5

    # Scoring
    use_strength_col: str = "cell_perc_weight"

    # Swap detection thresholds
    swap_angle_tolerance_deg: float = 30.0  # How close traffic direction must be to another cell's azimuth
    min_out_of_beam_ratio: float = 0.5  # Minimum ratio of out-of-beam traffic to be considered anomalous (50%)
    min_out_of_beam_weight: float = 5.0  # Minimum total out-of-beam weight to be considered

    # Minimum relation count thresholds (for data confidence)
    min_total_relations: int = 5  # Need at least N relations to have confidence
    min_out_of_beam_relations: int = 3  # Need at least N out-of-beam relations to flag

    # Swap candidate thresholds (looser thresholds for initial swap detection)
    swap_candidate_min_oob_ratio: float = 0.5  # Min out-of-beam ratio for swap candidates
    swap_candidate_min_relations: int = 3  # Min total relations for swap candidates
    swap_candidate_min_oob_relations: int = 2  # Min out-of-beam relations for swap candidates

    # Data quality thresholds
    max_data_drop_ratio: float = 0.5
    max_detection_rate: float = 0.20

    # Severity score thresholds (0-1 scale, matching other detectors)
    severity_threshold_critical: float = 0.80
    severity_threshold_high: float = 0.60
    severity_threshold_medium: float = 0.40
    severity_threshold_low: float = 0.20

    # Output options
    top_k_relations_per_cell: int = 5

    def __post_init__(self):
        """Validate parameter ranges."""
        if self.min_distance_m < 0:
            raise ValueError("min_distance_m must be non-negative")
        if self.max_radius_m <= 0:
            raise ValueError("max_radius_m must be positive")
        if not 0 <= self.min_out_of_beam_ratio <= 1:
            raise ValueError("min_out_of_beam_ratio must be in [0, 1]")
        if not 0 <= self.swap_candidate_min_oob_ratio <= 1:
            raise ValueError("swap_candidate_min_oob_ratio must be in [0, 1]")
        if self.swap_angle_tolerance_deg < 0 or self.swap_angle_tolerance_deg > 180:
            raise ValueError("swap_angle_tolerance_deg must be in [0, 180]")
        if self.min_total_relations < 1:
            raise ValueError("min_total_relations must be at least 1")
        if self.swap_candidate_min_relations < 1:
            raise ValueError("swap_candidate_min_relations must be at least 1")
        if not 0 < self.beamwidth_expansion_factor <= 3:
            raise ValueError("beamwidth_expansion_factor must be in (0, 3]")

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> 'CrossedFeederParams':
        """Load parameters from config file or use defaults."""
        if config_path is None:
            config_path = "config/crossed_feeder_params.json"

        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return cls()

        try:
            with open(path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in config file {config_path}: {e}. Using defaults.")
            return cls()
        except OSError as e:
            logger.warning(f"Cannot read config file {config_path}: {e}. Using defaults.")
            return cls()

        if not isinstance(config, dict):
            logger.warning(f"Config file {config_path} does not contain a dict. Using defaults.")
            return cls()

        params = config.get('default', config)
        if not isinstance(params, dict):
            logger.warning(f"Config params is not a dict. Using defaults.")
            return cls()

        try:
            return cls(
                max_radius_m=float(params.get('max_radius_m', 32000.0)),
                min_distance_m=float(params.get('min_distance_m', 500.0)),
                band_max_radius_m=params.get('band_max_radius_m', None),
                hbw_cap_deg=float(params.get('hbw_cap_deg', 60.0)),
                min_hbw_deg=float(params.get('min_hbw_deg', 1.0)),
                max_hbw_deg=float(params.get('max_hbw_deg', 179.0)),
                beamwidth_expansion_factor=float(params.get('beamwidth_expansion_factor', 1.5)),
                use_strength_col=str(params.get('use_strength_col', 'cell_perc_weight')),
                swap_angle_tolerance_deg=float(params.get('swap_angle_tolerance_deg', 30.0)),
                min_out_of_beam_ratio=float(params.get('min_out_of_beam_ratio', 0.5)),
                min_out_of_beam_weight=float(params.get('min_out_of_beam_weight', 5.0)),
                min_total_relations=int(params.get('min_total_relations', 5)),
                min_out_of_beam_relations=int(params.get('min_out_of_beam_relations', 3)),
                swap_candidate_min_oob_ratio=float(params.get('swap_candidate_min_oob_ratio', 0.5)),
                swap_candidate_min_relations=int(params.get('swap_candidate_min_relations', 3)),
                swap_candidate_min_oob_relations=int(params.get('swap_candidate_min_oob_relations', 2)),
                max_data_drop_ratio=float(params.get('max_data_drop_ratio', 0.5)),
                max_detection_rate=float(params.get('max_detection_rate', 0.20)),
                severity_threshold_critical=float(params.get('severity_threshold_critical', 0.80)),
                severity_threshold_high=float(params.get('severity_threshold_high', 0.60)),
                severity_threshold_medium=float(params.get('severity_threshold_medium', 0.40)),
                severity_threshold_low=float(params.get('severity_threshold_low', 0.20)),
                top_k_relations_per_cell=int(params.get('top_k_relations_per_cell', 5)),
            )
        except (TypeError, ValueError) as e:
            logger.warning(f"Invalid config parameter type: {e}. Using defaults.")
            return cls()


REQUIRED_REL_COLS = {
    "cell_name", "to_cell_name", "distance", "band", "to_band",
    "intra_site", "intra_cell", "weight",
}
REQUIRED_GIS_COLS = {"cell_name", "site", "band", "bearing", "hbw", "latitude", "longitude"}


def _norm_yn_vec(series: pd.Series) -> pd.Series:
    """Vectorized normalization of y/n values."""
    s = series.fillna("n").astype(str).str.strip().str.lower()
    return s.isin(["y", "yes", "true", "1"]).map({True: "y", False: "n"})


def _validate_columns(df: pd.DataFrame, required: set, name: str) -> None:
    """Validate required columns are present."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _pick_strength_series(rel: pd.DataFrame, col: str) -> pd.Series:
    """Returns a non-negative strength series."""
    if col in rel.columns:
        s = pd.to_numeric(rel[col], errors="coerce").fillna(0.0)
    else:
        s = pd.to_numeric(rel["weight"], errors="coerce").fillna(0.0)
    return s.clip(lower=0.0)


class CrossedFeederDetector:
    """
    Detects crossed feeders using swap pattern analysis.

    Confidence levels:
    - HIGH: Reciprocal swap pattern detected between 2+ cells at same site+band
    - MEDIUM: Multiple cells at same site+band have out-of-beam anomalies (no clean swap)
    - LOW: Single cell has out-of-beam anomaly - likely azimuth misconfiguration
    """

    def __init__(self, params: Optional[CrossedFeederParams] = None):
        self.params = params or CrossedFeederParams()
        logger.info(
            "Crossed Feeder detector initialized",
            swap_tolerance_deg=self.params.swap_angle_tolerance_deg,
            min_out_of_beam_ratio=self.params.min_out_of_beam_ratio,
            min_total_relations=self.params.min_total_relations,
            min_out_of_beam_relations=self.params.min_out_of_beam_relations,
        )

    def detect(
        self,
        relations_df: pd.DataFrame,
        gis_df: pd.DataFrame,
        band_filter: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Detect crossed feeders using swap pattern analysis.

        Returns:
            Dictionary with keys:
            - 'cells': Per-cell results with swap detection
            - 'sites': Per-site summary
            - 'swap_pairs': Detected swap pairs (HIGH confidence)
            - 'relation_details': Detailed relation-level data
        """
        logger.info("Starting crossed feeder detection")

        _validate_columns(relations_df, REQUIRED_REL_COLS, "relations")
        _validate_columns(gis_df, REQUIRED_GIS_COLS, "gis")

        rel = relations_df.copy()
        gis = gis_df.copy()

        rel["intra_site"] = _norm_yn_vec(rel["intra_site"])
        rel["intra_cell"] = _norm_yn_vec(rel["intra_cell"])

        if band_filter is not None:
            rel = rel[rel["band"].astype(str) == str(band_filter)].copy()
            gis = gis[gis["band"].astype(str) == str(band_filter)].copy()
            logger.info(f"Filtered to band={band_filter}: relations={len(rel)}, gis={len(gis)}")

        # Build relation-level geometry
        rel_scores = self._build_relation_geometry(rel, gis)

        if len(rel_scores) == 0:
            logger.warning("No valid relations after filtering")
            return {
                'cells': pd.DataFrame(),
                'sites': pd.DataFrame(),
                'swap_pairs': pd.DataFrame(),
                'relation_details': pd.DataFrame(),
            }

        # Calculate per-cell traffic direction and anomaly metrics
        cell_metrics = self._calculate_cell_metrics(rel_scores, gis)

        # Detect swap patterns at each site+band
        cell_results, swap_pairs = self._detect_swap_patterns(cell_metrics, gis)

        # Build site summary
        site_summary = self._build_site_summary(cell_results)

        # Log results
        high_swap = len(cell_results[cell_results['confidence_level'] == 'HIGH_POTENTIAL_SWAP'])
        possible_swap = len(cell_results[cell_results['confidence_level'] == 'POSSIBLE_SWAP'])
        single_anomaly = len(cell_results[cell_results['confidence_level'] == 'SINGLE_ANOMALY'])
        repan = len(cell_results[cell_results['confidence_level'] == 'REPAN'])

        logger.info(
            "Crossed feeder detection complete",
            high_potential_swap=high_swap,
            possible_swap=possible_swap,
            single_anomaly=single_anomaly,
            repan_candidates=repan,
            swap_pairs=len(swap_pairs),
        )

        return {
            'cells': cell_results,
            'sites': site_summary,
            'swap_pairs': swap_pairs,
            'relation_details': rel_scores,
        }

    def _build_relation_geometry(
        self,
        rel: pd.DataFrame,
        gis: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build per-relation geometry with in-beam/out-of-beam classification."""
        cfg = self.params

        # Filter to same-technology relations
        rel["_src_tech"] = rel["band"].apply(_get_tech_from_band)
        rel["_tgt_tech"] = rel["to_band"].apply(_get_tech_from_band)

        total_before = len(rel)
        rel = rel[rel["_src_tech"] == rel["_tgt_tech"]].copy()
        cross_tech_dropped = total_before - len(rel)

        if cross_tech_dropped > 0:
            logger.info(
                "Filtered to same-technology relations",
                same_tech=len(rel),
                cross_tech_dropped=cross_tech_dropped,
            )

        rel = rel.drop(columns=["_src_tech", "_tgt_tech"])

        # Join GIS for serving cell
        gis_s = gis.rename(columns={
            "site": "site",
            "bearing": "bearing",
            "hbw": "hbw",
            "latitude": "lat",
            "longitude": "lon",
        })[["cell_name", "site", "bearing", "hbw", "lat", "lon"]].copy()

        # Join GIS for neighbour cell
        gis_n = gis.rename(columns={
            "cell_name": "to_cell_name",
            "latitude": "to_lat",
            "longitude": "to_lon",
        })[["to_cell_name", "to_lat", "to_lon"]].copy()

        # Drop conflicting columns
        drop_cols = [c for c in ['site', 'bearing', 'hbw', 'lat', 'lon'] if c in rel.columns]
        if drop_cols:
            rel = rel.drop(columns=drop_cols)

        df = rel.merge(gis_s, on="cell_name", how="left").merge(gis_n, on="to_cell_name", how="left")

        # Convert to numeric
        for col in ["distance", "bearing", "hbw", "lat", "lon", "to_lat", "to_lon"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["distance"] = df["distance"].fillna(0.0)

        # Normalize bearing to [0, 360) range with float precision rounding
        df["bearing"] = (df["bearing"] % 360.0).round(6)

        # Drop rows missing essential geometry
        before = len(df)
        df = df.dropna(subset=["bearing", "hbw", "lat", "lon", "to_lat", "to_lon"]).copy()
        dropped = before - len(df)

        if dropped > 0:
            logger.warning(f"Dropped {dropped} same-tech relations due to missing GIS geometry")
            drop_ratio = dropped / before if before > 0 else 0
            if drop_ratio > cfg.max_data_drop_ratio:
                logger.error(
                    "data_quality_warning",
                    dropped=dropped,
                    total=before,
                    drop_ratio=round(drop_ratio, 2),
                    message=f"High same-tech data drop rate ({drop_ratio:.1%})"
                )

        # Beam sanity filter
        df = df[(df["hbw"] >= cfg.min_hbw_deg) & (df["hbw"] <= cfg.max_hbw_deg)].copy()

        # Band-specific distance filter
        band_radii = cfg.band_max_radius_m or BAND_MAX_RADIUS_M

        def get_max_radius(band: str) -> float:
            band_upper = str(band).upper()
            if band_upper in band_radii:
                return band_radii[band_upper]
            for key, val in band_radii.items():
                if key in band_upper or band_upper in key:
                    return val
            return DEFAULT_MAX_RADIUS_M

        df["band_max_radius"] = df["band"].apply(get_max_radius)
        df = df[
            (df["distance"] <= df["band_max_radius"]) &
            (df["distance"] >= cfg.min_distance_m)
        ].copy()

        # Inter-site relations only (feeders connect to external neighbors)
        df = df[df["intra_site"] == "n"].copy()

        if len(df) == 0:
            return df

        # Strength
        df["weight"] = _pick_strength_series(df, cfg.use_strength_col)

        # Angle to neighbor
        df["angle_to_neighbor"] = bearing_deg_vec(
            df["lat"].values, df["lon"].values,
            df["to_lat"].values, df["to_lon"].values
        )

        # In-beam test with expansion factor
        df["half_width"] = (df["hbw"] * cfg.beamwidth_expansion_factor).clip(upper=cfg.hbw_cap_deg)
        df["angle_diff"] = circ_diff_deg_vec(df["bearing"].values, df["angle_to_neighbor"].values)
        df["in_beam"] = df["angle_diff"] <= df["half_width"]

        return df

    def _calculate_cell_metrics(
        self,
        rel: pd.DataFrame,
        gis: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate per-cell traffic direction and anomaly metrics."""
        cfg = self.params

        # Get cell info from GIS
        cell_info = gis[["cell_name", "site", "band", "bearing", "hbw", "latitude", "longitude"]].copy()
        cell_info = cell_info.rename(columns={"latitude": "lat", "longitude": "lon"})

        # Calculate per-cell metrics
        metrics = []

        for cell_name, cell_rel in rel.groupby("cell_name"):
            total_weight = cell_rel["weight"].sum()
            in_beam_weight = cell_rel[cell_rel["in_beam"]]["weight"].sum()
            out_of_beam_weight = cell_rel[~cell_rel["in_beam"]]["weight"].sum()

            total_relations = len(cell_rel)
            out_of_beam_relations = len(cell_rel[~cell_rel["in_beam"]])

            out_of_beam_ratio = out_of_beam_weight / total_weight if total_weight > 0 else 0

            # Calculate dominant traffic direction (weighted mean of out-of-beam neighbors)
            oob = cell_rel[~cell_rel["in_beam"]]
            if len(oob) > 0 and oob["weight"].sum() > 0:
                dominant_direction = weighted_circular_mean(
                    oob["angle_to_neighbor"].values,
                    oob["weight"].values
                )
            else:
                dominant_direction = np.nan

            # Get top suspicious relations
            top_oob = cell_rel[~cell_rel["in_beam"]].nlargest(cfg.top_k_relations_per_cell, "weight")
            top_relations = " | ".join([
                f"{row['to_cell_name']} (d={int(row['distance'])}m, w={row['weight']:.1f}, angle={row['angle_to_neighbor']:.0f}°)"
                for _, row in top_oob.iterrows()
            ])

            metrics.append({
                "cell_name": cell_name,
                "total_weight": total_weight,
                "in_beam_weight": in_beam_weight,
                "out_of_beam_weight": out_of_beam_weight,
                "out_of_beam_ratio": out_of_beam_ratio,
                "total_relations": total_relations,
                "out_of_beam_relations": out_of_beam_relations,
                "dominant_traffic_direction": dominant_direction,
                "top_suspicious_relations": top_relations,
            })

        metrics_df = pd.DataFrame(metrics)

        # Merge with cell info
        metrics_df = metrics_df.merge(cell_info, on="cell_name", how="left")

        return metrics_df

    def _detect_swap_patterns(
        self,
        cell_metrics: pd.DataFrame,
        gis: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect swap patterns at each site+band combination."""
        cfg = self.params

        results = []
        swap_pairs = []

        # Count cells per site+band from GIS (actual cell count, not just those with relations)
        site_band_cell_count = gis.groupby(["site", "band"]).size().to_dict()

        # Process each site+band group
        for (site, band), group in cell_metrics.groupby(["site", "band"]):
            n_cells = len(group)
            # Get actual cell count from GIS for this site+band
            n_cells_on_band = site_band_cell_count.get((site, band), n_cells)

            # For swap detection: use looser thresholds (configurable)
            # because the reciprocal swap pattern itself is strong evidence
            swap_candidates = group[
                (group["out_of_beam_ratio"] >= cfg.swap_candidate_min_oob_ratio) &
                (group["total_relations"] >= cfg.swap_candidate_min_relations) &
                (group["out_of_beam_relations"] >= cfg.swap_candidate_min_oob_relations)
            ]

            # For MEDIUM/LOW flagging: use stricter thresholds
            # Must meet ALL thresholds: ratio, weight, AND relation counts
            anomalous = group[
                (group["out_of_beam_ratio"] >= cfg.min_out_of_beam_ratio) &
                (group["out_of_beam_weight"] >= cfg.min_out_of_beam_weight) &
                (group["total_relations"] >= cfg.min_total_relations) &
                (group["out_of_beam_relations"] >= cfg.min_out_of_beam_relations)
            ]

            n_anomalous = len(anomalous)
            n_swap_candidates = len(swap_candidates)

            # Pre-compute set for O(1) lookups instead of O(n) array search
            anomalous_cells = set(anomalous["cell_name"])

            # Handle single-cell-on-band case: can't be crossed feeders (nothing to swap with)
            # But flag as antenna repan candidates if significant out-of-beam traffic
            if n_cells_on_band == 1:
                for _, cell in group.iterrows():
                    cell_name = cell["cell_name"]
                    if cell_name in anomalous_cells:
                        # Significant out-of-beam but only 1 cell on band - repan candidate
                        results.append(self._build_cell_result(
                            cell, "REPAN", None,
                            "Single cell on band with out-of-beam traffic. Cannot be crossed feeder. Review antenna azimuth/pan."
                        ))
                    else:
                        results.append(self._build_cell_result(cell, "NONE", None, "No out-of-beam anomaly detected"))
                continue

            # Check for swap patterns using looser thresholds (swap_candidates)
            # The reciprocal swap pattern itself is strong evidence
            detected_swaps = set()
            cell_swap_partners = {}

            for i, cell_a in swap_candidates.iterrows():
                if pd.isna(cell_a["dominant_traffic_direction"]):
                    continue

                for j, cell_b in swap_candidates.iterrows():
                    if i >= j:  # Avoid duplicate pairs
                        continue
                    if pd.isna(cell_b["dominant_traffic_direction"]):
                        continue

                    # Check if Cell A's traffic points toward Cell B's azimuth
                    a_to_b_diff = circ_diff_deg(cell_a["dominant_traffic_direction"], cell_b["bearing"])
                    # Check if Cell B's traffic points toward Cell A's azimuth
                    b_to_a_diff = circ_diff_deg(cell_b["dominant_traffic_direction"], cell_a["bearing"])

                    # Both directions must match for a swap
                    if (a_to_b_diff <= cfg.swap_angle_tolerance_deg and
                        b_to_a_diff <= cfg.swap_angle_tolerance_deg):

                        # Found a swap pair!
                        detected_swaps.add((cell_a["cell_name"], cell_b["cell_name"]))
                        cell_swap_partners[cell_a["cell_name"]] = cell_b["cell_name"]
                        cell_swap_partners[cell_b["cell_name"]] = cell_a["cell_name"]

                        swap_pairs.append({
                            "site": site,
                            "band": band,
                            "cell_a": cell_a["cell_name"],
                            "cell_b": cell_b["cell_name"],
                            "cell_a_azimuth": cell_a["bearing"],
                            "cell_b_azimuth": cell_b["bearing"],
                            "cell_a_traffic_dir": cell_a["dominant_traffic_direction"],
                            "cell_b_traffic_dir": cell_b["dominant_traffic_direction"],
                            "a_to_b_angle_diff": a_to_b_diff,
                            "b_to_a_angle_diff": b_to_a_diff,
                        })

            # Classify all cells at this site+band
            for _, cell in group.iterrows():
                cell_name = cell["cell_name"]

                if cell_name in cell_swap_partners:
                    # HIGH_POTENTIAL_SWAP: part of a detected swap pair
                    partner = cell_swap_partners[cell_name]
                    results.append(self._build_cell_result(
                        cell, "HIGH_POTENTIAL_SWAP", partner,
                        f"Reciprocal swap pattern detected with {partner}. Check feeder connections."
                    ))
                elif cell_name in anomalous_cells:
                    if n_anomalous >= 2:
                        # POSSIBLE_SWAP: multiple anomalies but no clean swap
                        results.append(self._build_cell_result(
                            cell, "POSSIBLE_SWAP", None,
                            f"Out-of-beam anomaly at site with {n_anomalous} anomalous cells. Investigate feeder routing."
                        ))
                    else:
                        # SINGLE_ANOMALY: single cell anomaly
                        results.append(self._build_cell_result(
                            cell, "SINGLE_ANOMALY", None,
                            "Single cell out-of-beam anomaly. Review azimuth configuration or terrain effects."
                        ))
                else:
                    # No anomaly for this cell
                    results.append(self._build_cell_result(cell, "NONE", None, "No out-of-beam anomaly detected"))

        results_df = pd.DataFrame(results)
        swap_pairs_df = pd.DataFrame(swap_pairs)

        # Sort results
        confidence_order = {"HIGH_POTENTIAL_SWAP": 0, "POSSIBLE_SWAP": 1, "SINGLE_ANOMALY": 2, "REPAN": 3, "NONE": 4}
        if len(results_df) > 0:
            results_df["_sort"] = results_df["confidence_level"].map(confidence_order)
            results_df = results_df.sort_values(["_sort", "out_of_beam_ratio"], ascending=[True, False])
            results_df = results_df.drop(columns=["_sort"])

        return results_df, swap_pairs_df

    def _calculate_cell_severity(self, confidence_level: str, out_of_beam_ratio: float) -> tuple:
        """
        Calculate severity score (0-1) and category for a cell.

        Severity is based on:
        - Confidence level: HIGH_POTENTIAL_SWAP contributes most, NONE contributes least
        - Out-of-beam ratio: Higher ratio = more severe

        Args:
            confidence_level: Detection confidence (HIGH_POTENTIAL_SWAP, POSSIBLE_SWAP, etc.)
            out_of_beam_ratio: Ratio of out-of-beam traffic (0-1)

        Returns:
            Tuple of (severity_score: float, severity_category: str)
        """
        # Base score from confidence level
        confidence_scores = {
            "HIGH_POTENTIAL_SWAP": 0.90,
            "POSSIBLE_SWAP": 0.70,
            "SINGLE_ANOMALY": 0.50,
            "REPAN": 0.30,
            "NONE": 0.0,
        }
        base_score = confidence_scores.get(confidence_level, 0.0)

        # Add out-of-beam ratio component (up to 0.10 bonus)
        oob_bonus = min(out_of_beam_ratio * 0.10, 0.10)

        # Final severity score
        severity_score = round(min(1.0, base_score + oob_bonus), 4)

        # Determine severity category
        if severity_score >= self.params.severity_threshold_critical:
            severity_category = "CRITICAL"
        elif severity_score >= self.params.severity_threshold_high:
            severity_category = "HIGH"
        elif severity_score >= self.params.severity_threshold_medium:
            severity_category = "MEDIUM"
        elif severity_score >= self.params.severity_threshold_low:
            severity_category = "LOW"
        else:
            severity_category = "MINIMAL"

        return severity_score, severity_category

    def _build_cell_result(
        self,
        cell: pd.Series,
        confidence_level: str,
        swap_partner: Optional[str],
        recommendation: str,
    ) -> Dict:
        """Build a cell result dictionary."""
        severity_score, severity_category = self._calculate_cell_severity(
            confidence_level, cell["out_of_beam_ratio"]
        )
        return {
            "cell_name": cell["cell_name"],
            "site": cell["site"],
            "band": cell["band"],
            "bearing": cell["bearing"],
            "hbw": cell["hbw"],
            "total_relations": cell["total_relations"],
            "in_beam_relations": cell["total_relations"] - cell["out_of_beam_relations"],
            "out_of_beam_relations": cell["out_of_beam_relations"],
            "total_weight": cell["total_weight"],
            "in_beam_weight": cell["in_beam_weight"],
            "out_of_beam_weight": cell["out_of_beam_weight"],
            "out_of_beam_ratio": round(cell["out_of_beam_ratio"], 3),
            "dominant_traffic_direction": cell["dominant_traffic_direction"],
            "confidence_level": confidence_level,
            "severity_score": severity_score,
            "severity_category": severity_category,
            "swap_partner": swap_partner,
            "recommendation": recommendation,
            "top_suspicious_relations": cell["top_suspicious_relations"],
            "flagged": confidence_level in ("HIGH_POTENTIAL_SWAP", "POSSIBLE_SWAP", "SINGLE_ANOMALY", "REPAN"),
        }

    def _build_site_summary(self, cell_results: pd.DataFrame) -> pd.DataFrame:
        """Build per-site summary."""
        if len(cell_results) == 0:
            return pd.DataFrame()

        summary = []

        for (site, band), group in cell_results.groupby(["site", "band"]):
            n_cells = len(group)
            # Map actual confidence levels to summary categories
            high_conf = len(group[group["confidence_level"] == "HIGH_POTENTIAL_SWAP"])
            medium_conf = len(group[group["confidence_level"].isin(["POSSIBLE_SWAP", "SINGLE_ANOMALY"])])
            low_conf = len(group[group["confidence_level"] == "REPAN"])

            # Get max severity_score from cells in the group
            max_severity_score = round(group["severity_score"].max(), 4) if "severity_score" in group.columns else 0.0

            # Determine site-level classification and severity category
            if high_conf > 0:
                classification = f"CROSSED FEEDER: {high_conf} swap pair(s) detected"
                severity_category = "HIGH"
            elif medium_conf > 0:
                classification = f"POSSIBLE ISSUE: {medium_conf} cells with out-of-beam anomalies"
                severity_category = "MEDIUM"
            elif low_conf > 0:
                classification = f"AZIMUTH REVIEW: {low_conf} cell(s) may need azimuth adjustment"
                severity_category = "LOW"
            else:
                classification = "No issues detected"
                severity_category = "MINIMAL"

            summary.append({
                "site": site,
                "band": band,
                "total_cells": n_cells,
                "high_confidence": high_conf,
                "medium_confidence": medium_conf,
                "low_confidence": low_conf,
                "severity_score": max_severity_score,
                "severity_category": severity_category,
                "classification": classification,
            })

        summary_df = pd.DataFrame(summary)

        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "MINIMAL": 3}
        summary_df["_sort"] = summary_df["severity_category"].map(severity_order)
        summary_df = summary_df.sort_values(["_sort", "high_confidence", "medium_confidence"], ascending=[True, False, False])
        summary_df = summary_df.drop(columns=["_sort"])

        return summary_df


def detect_crossed_feeders(
    relations_df: pd.DataFrame,
    gis_df: pd.DataFrame,
    params: Optional[CrossedFeederParams] = None,
    band_filter: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to detect crossed feeders.

    Returns dictionary with:
    - 'cells': Per-cell results with confidence level and swap partner
    - 'sites': Per-site summary
    - 'swap_pairs': Detected swap pairs
    - 'relation_details': Relation-level data
    """
    detector = CrossedFeederDetector(params)
    return detector.detect(relations_df, gis_df, band_filter)

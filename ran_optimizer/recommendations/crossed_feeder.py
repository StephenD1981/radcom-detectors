"""
Crossed Feeder Detection.

Detects likely crossed feeders (sector/feeder swaps) using:
  1) Directed neighbour relations with strength (e.g., weekly handovers)
  2) Cell GIS / inventory with azimuth (bearing), beamwidth (hbw), and lat/lon

Primary signal:
  A cell shows strong neighbour relations to cells located OUTSIDE its expected main lobe.

Secondary signal (optional):
  Co-sectored cross-band anomaly: within the same site, the expected co-sectored
  cross-band relation is weak compared to a non-co-sectored cross-band relation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


# -----------------------------
# Configuration
# -----------------------------

# Band-specific maximum radius thresholds (meters)
# Different bands have different propagation characteristics
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
DEFAULT_MAX_RADIUS_M = 32000  # Default for unknown bands

# Technology prefixes for same-tech filtering
TECH_PREFIXES = {
    'L': 'LTE',    # L700, L800, L900, L1800, L2100, L2600
    'N': 'NR',     # N78, N258 (5G NR)
    'U': 'UMTS',   # U900, U2100
    'G': 'GSM',    # G900, G1800
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
    max_radius_m: float = 32000.0  # Default, overridden by band-specific values
    min_distance_m: float = 500.0
    hbw_cap_deg: float = 60.0  # cap the "half-width" used for in-beam test
    percentile: float = 0.95
    min_hbw_deg: float = 1.0
    max_hbw_deg: float = 179.0  # exclude broken/omni-like hbw >= 180
    top_k_relations_per_cell: int = 5

    # In-beam expansion factor (1.5x 3dB beamwidth for softer boundary)
    beamwidth_expansion_factor: float = 1.5

    # Scoring
    use_strength_col: str = "cell_perc_weight"  # fallback to weight if missing
    distance_weighting: bool = True
    angle_weighting: bool = True

    # Co-sectored cross-band anomaly (secondary output)
    enable_cosectored_check: bool = False
    cosectored_strength_col: str = "weight"
    cosectored_min_strength: float = 10.0
    cosectored_diff_thresh: float = -10.0

    # Classification thresholds (ratio-based for multi-sector sites)
    high_ratio_threshold: float = 0.8  # >=80% of sectors flagged = high potential
    medium_ratio_threshold: float = 0.5  # >=50% of sectors flagged = medium potential

    # Data quality thresholds
    max_data_drop_ratio: float = 0.5  # Warn if >50% of data dropped
    max_detection_rate: float = 0.20  # Warn if >20% of cells flagged

    # Band-specific radius overrides (loaded from config)
    band_max_radius_m: Optional[Dict[str, float]] = None

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> 'CrossedFeederParams':
        """Load parameters from config file or use defaults."""
        if config_path is None:
            config_path = "config/crossed_feeder_params.json"

        try:
            import json
            from pathlib import Path

            path = Path(config_path)
            if not path.exists():
                logger.warning(f"Config file not found: {config_path}. Using defaults.")
                return cls()

            with open(path, 'r') as f:
                config = json.load(f)

            params = config.get('default', config)

            return cls(
                max_radius_m=params.get('max_radius_m', 32000.0),
                min_distance_m=params.get('min_distance_m', 500.0),
                hbw_cap_deg=params.get('hbw_cap_deg', 60.0),
                percentile=params.get('percentile', 0.95),
                min_hbw_deg=params.get('min_hbw_deg', 1.0),
                max_hbw_deg=params.get('max_hbw_deg', 179.0),
                top_k_relations_per_cell=params.get('top_k_relations_per_cell', 5),
                beamwidth_expansion_factor=params.get('beamwidth_expansion_factor', 1.5),
                use_strength_col=params.get('use_strength_col', 'cell_perc_weight'),
                distance_weighting=params.get('distance_weighting', True),
                angle_weighting=params.get('angle_weighting', True),
                enable_cosectored_check=params.get('enable_cosectored_check', False),
                cosectored_strength_col=params.get('cosectored_strength_col', 'weight'),
                cosectored_min_strength=params.get('cosectored_min_strength', 10.0),
                cosectored_diff_thresh=params.get('cosectored_diff_thresh', -10.0),
                high_ratio_threshold=params.get('high_ratio_threshold', 0.8),
                medium_ratio_threshold=params.get('medium_ratio_threshold', 0.5),
                max_data_drop_ratio=params.get('max_data_drop_ratio', 0.5),
                max_detection_rate=params.get('max_detection_rate', 0.20),
                band_max_radius_m=params.get('band_max_radius_m', None),
            )
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return cls()


REQUIRED_REL_COLS = {
    "cell_name", "to_cell_name", "distance", "band", "to_band",
    "intra_site", "intra_cell", "weight",
}
REQUIRED_GIS_COLS = {"cell_name", "site", "band", "bearing", "hbw", "latitude", "longitude"}


def _norm_yn(x) -> str:
    """Normalize y/n values."""
    if pd.isna(x):
        return "n"
    s = str(x).strip().lower()
    return "y" if s in ("y", "yes", "true", "1") else "n"


def _norm_yn_vec(series: pd.Series) -> pd.Series:
    """Vectorized version of _norm_yn."""
    s = series.fillna("n").astype(str).str.strip().str.lower()
    return s.isin(["y", "yes", "true", "1"]).map({True: "y", False: "n"})


def _validate_columns(df: pd.DataFrame, required: set, name: str) -> None:
    """Validate required columns are present."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _pick_strength_series(rel: pd.DataFrame, col: str) -> pd.Series:
    """
    Returns a non-negative strength series.
    Prefers rel[col] if exists, otherwise fallback to rel['weight'].
    """
    if col in rel.columns:
        s = pd.to_numeric(rel[col], errors="coerce").fillna(0.0)
    else:
        s = pd.to_numeric(rel["weight"], errors="coerce").fillna(0.0)
    return s.clip(lower=0.0)


class CrossedFeederDetector:
    """Detects crossed feeders using neighbour relation geometry analysis.

    Required relation columns (canonical names):
        cell_name, to_cell_name, distance, band, to_band, intra_site, intra_cell, weight
    Optional relation columns:
        cell_perc_weight

    Required GIS columns (canonical names):
        cell_name, site, band, bearing, hbw, latitude, longitude
    """

    def __init__(self, params: Optional[CrossedFeederParams] = None):
        """
        Initialize the Crossed Feeder Detector.

        Args:
            params: Detection parameters (uses defaults if None)
        """
        self.params = params or CrossedFeederParams()

        logger.info(
            "Crossed Feeder detector initialized",
            max_radius_m=self.params.max_radius_m,
            min_distance_m=self.params.min_distance_m,
            percentile=self.params.percentile,
        )

    def detect(
        self,
        relations_df: pd.DataFrame,
        gis_df: pd.DataFrame,
        band_filter: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Detect crossed feeders using neighbour relations and GIS data.

        Args:
            relations_df: DataFrame with neighbour relations (canonical column names)
            gis_df: DataFrame with cell GIS data (canonical column names)
            band_filter: Optional filter on serving band (e.g., 'L800')

        Returns:
            Dictionary with keys:
            - 'relation_scores': Per-relation geometry and scores
            - 'cell_scores': Per-cell aggregated scores
            - 'site_summary': Per-site summary with classification
            - 'cosectored_anomalies': Co-sectored cross-band anomalies (if enabled)
        """
        logger.info("Starting crossed feeder detection")

        # Validate columns
        _validate_columns(relations_df, REQUIRED_REL_COLS, "relations")
        _validate_columns(gis_df, REQUIRED_GIS_COLS, "gis")

        # Make copies and normalize using vectorized operations
        rel = relations_df.copy()
        gis = gis_df.copy()

        rel["intra_site"] = _norm_yn_vec(rel["intra_site"])
        rel["intra_cell"] = _norm_yn_vec(rel["intra_cell"])

        # Optional band filter
        if band_filter is not None:
            rel = rel[rel["band"].astype(str) == str(band_filter)].copy()
            gis = gis[gis["band"].astype(str) == str(band_filter)].copy()
            logger.info(f"Filtered to band={band_filter}: relations={len(rel)}, gis={len(gis)}")

        # Build relation scores
        logger.info("Building relation-level scores")
        rel_scores = self._build_relation_scores(rel, gis)

        # Build cell and site outputs
        logger.info("Aggregating to cell and site outputs")
        cell_scores, site_summary = self._build_cell_and_site_outputs(rel_scores)

        results = {
            'relation_scores': rel_scores,
            'cell_scores': cell_scores,
            'site_summary': site_summary,
        }

        # Optional co-sectored check
        if self.params.enable_cosectored_check:
            logger.info("Running co-sectored cross-band anomaly check")
            cosec = self._cosectored_cross_band_anomalies(relations_df, gis_df)
            results['cosectored_anomalies'] = cosec

        # Log summary and detection rate sanity check
        flagged_cells = int(cell_scores["flagged"].sum()) if "flagged" in cell_scores.columns else 0
        total_cells = len(cell_scores)
        detection_rate = flagged_cells / total_cells if total_cells > 0 else 0

        # Sanity check: warn if detection rate is unusually high
        if detection_rate > self.params.max_detection_rate:
            logger.warning(
                "high_detection_rate_warning",
                flagged_cells=flagged_cells,
                total_cells=total_cells,
                detection_rate=round(detection_rate, 3),
                threshold=self.params.max_detection_rate,
                message=f"Unusually high detection rate ({detection_rate:.1%}) - review thresholds or data quality"
            )

        logger.info(
            "Crossed feeder detection complete",
            relations_scored=len(rel_scores),
            flagged_cells=flagged_cells,
            total_cells=total_cells,
            detection_rate=round(detection_rate, 3),
        )

        return results

    def _build_relation_scores(
        self,
        rel: pd.DataFrame,
        gis: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build per-relation geometry and suspiciousness score."""
        cfg = self.params

        # Filter to same-technology relations before GIS merge
        # This ensures data quality metrics only consider valid same-tech relations
        rel["_src_tech"] = rel["band"].apply(_get_tech_from_band)
        rel["_tgt_tech"] = rel["to_band"].apply(_get_tech_from_band)

        total_before_tech_filter = len(rel)
        rel = rel[rel["_src_tech"] == rel["_tgt_tech"]].copy()
        cross_tech_dropped = total_before_tech_filter - len(rel)

        if cross_tech_dropped > 0:
            logger.info(
                "Filtered to same-technology relations",
                same_tech_relations=len(rel),
                cross_tech_dropped=cross_tech_dropped,
                total_before=total_before_tech_filter,
            )

        # Clean up temp columns
        rel = rel.drop(columns=["_src_tech", "_tgt_tech"])

        # Join GIS for serving cell
        gis_s = gis.rename(
            columns={
                "cell_name": "cell_name",
                "site": "site",
                "band": "band_gis",
                "bearing": "bearing",
                "hbw": "hbw",
                "latitude": "lat",
                "longitude": "lon",
            }
        )[["cell_name", "site", "band_gis", "bearing", "hbw", "lat", "lon"]].copy()

        # Join GIS for neighbour cell
        gis_n = gis.rename(
            columns={
                "cell_name": "to_cell_name",
                "site": "to_site",
                "band": "to_band_gis",
                "latitude": "to_lat",
                "longitude": "to_lon",
            }
        )[["to_cell_name", "to_site", "to_band_gis", "to_lat", "to_lon"]].copy()

        # Drop any conflicting columns from rel before merge
        rel_cols_to_drop = [c for c in ['site', 'band_gis', 'bearing', 'hbw', 'lat', 'lon'] if c in rel.columns]
        if rel_cols_to_drop:
            rel = rel.drop(columns=rel_cols_to_drop)

        df = rel.merge(gis_s, on="cell_name", how="left").merge(gis_n, on="to_cell_name", how="left")

        # Basic sanitation / filters (vectorized)
        df["intra_site"] = _norm_yn_vec(df["intra_site"])
        df["intra_cell"] = _norm_yn_vec(df["intra_cell"])

        df["distance"] = pd.to_numeric(df["distance"], errors="coerce").fillna(0.0)
        df["bearing"] = pd.to_numeric(df["bearing"], errors="coerce")
        df["hbw"] = pd.to_numeric(df["hbw"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df["to_lat"] = pd.to_numeric(df["to_lat"], errors="coerce")
        df["to_lon"] = pd.to_numeric(df["to_lon"], errors="coerce")

        # Drop rows missing essential geometry
        before = len(df)
        df = df.dropna(subset=["bearing", "hbw", "lat", "lon", "to_lat", "to_lon"]).copy()
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} same-tech relations due to missing GIS geometry")

            # Data quality check: warn if too much same-tech data dropped
            drop_ratio = dropped / before if before > 0 else 0
            if drop_ratio > cfg.max_data_drop_ratio:
                logger.error(
                    "data_quality_warning",
                    dropped=dropped,
                    total=before,
                    drop_ratio=round(drop_ratio, 2),
                    threshold=cfg.max_data_drop_ratio,
                    message=f"High same-tech data drop rate ({drop_ratio:.1%}) - check GIS data quality"
                )

        # Beam sanity filter
        df = df[(df["hbw"] >= cfg.min_hbw_deg) & (df["hbw"] <= cfg.max_hbw_deg)].copy()

        # Apply band-specific distance filter
        # Get band-specific max radius from config or use defaults
        band_radii = cfg.band_max_radius_m if cfg.band_max_radius_m else BAND_MAX_RADIUS_M

        def get_max_radius_for_band(band: str) -> float:
            """Get max radius for a specific band."""
            band_upper = str(band).upper()
            if band_upper in band_radii:
                return band_radii[band_upper]
            # Try partial match (e.g., 'L800' matches '800')
            for key, val in band_radii.items():
                if key in band_upper or band_upper in key:
                    return val
            return DEFAULT_MAX_RADIUS_M

        # Apply band-specific max radius
        df["band_max_radius"] = df["band"].apply(get_max_radius_for_band)
        df = df[
            (df["distance"] <= df["band_max_radius"]) &
            (df["distance"] >= cfg.min_distance_m)
        ].copy()

        # Primary detection uses INTER-SITE relations
        df = df[df["intra_site"] == "n"].copy()

        # Strength
        df["strength"] = _pick_strength_series(df, cfg.use_strength_col)

        # Compute angle to neighbour (vectorized)
        df["angle_to_neighbor"] = bearing_deg_vec(
            df["lat"].values, df["lon"].values,
            df["to_lat"].values, df["to_lon"].values
        )

        # In-beam test (vectorized) with expansion factor for softer boundary
        # Use 1.5x the 3dB beamwidth to account for RF energy beyond main lobe
        df["half_width_deg"] = (df["hbw"] * cfg.beamwidth_expansion_factor).clip(upper=cfg.hbw_cap_deg)
        df["angle_diff_deg"] = circ_diff_deg_vec(df["bearing"].values, df["angle_to_neighbor"].values)
        df["in_beam"] = df["angle_diff_deg"] <= df["half_width_deg"]

        # Distance normalization per serving cell
        max_dist = df.groupby("cell_name")["distance"].max().rename("max_neigh_distance_m")
        df = df.merge(max_dist, on="cell_name", how="left")
        df["dist_factor"] = (df["distance"] / (df["max_neigh_distance_m"].replace(0, pd.NA))).fillna(0.0)
        df["dist_factor"] = df["dist_factor"].clip(lower=0.0, upper=1.0)

        # Angle factor
        df["angle_factor"] = (df["angle_diff_deg"] / 180.0).clip(lower=0.0, upper=1.0)

        # Suspicious score: only out-of-beam relations contribute
        df["score"] = 0.0
        out_mask = ~df["in_beam"]
        score = df.loc[out_mask, "strength"].copy()

        if cfg.distance_weighting:
            score = score * df.loc[out_mask, "dist_factor"]
        if cfg.angle_weighting:
            score = score * df.loc[out_mask, "angle_factor"]

        df.loc[out_mask, "score"] = score

        # Keep relevant columns
        keep_cols = [
            "cell_name", "band", "site",
            "to_cell_name", "to_band", "to_site",
            "distance",
            "strength",
            "bearing", "hbw", "half_width_deg",
            "angle_to_neighbor", "angle_diff_deg", "in_beam",
            "dist_factor", "angle_factor",
            "score",
        ]
        # Include metadata if present
        for extra in ["pci", "to_pci", "cell_perc_weight", "weight"]:
            if extra in df.columns and extra not in keep_cols:
                keep_cols.append(extra)

        df = df[keep_cols].copy()
        df = df.sort_values(["score", "strength", "distance"], ascending=[False, False, False])
        return df

    def _build_cell_and_site_outputs(
        self,
        rel_scores: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate relation scores into per-cell and per-site summaries."""
        cfg = self.params

        # Per-cell score with additional metrics for confidence calculation
        cell = (
            rel_scores.groupby(["cell_name", "site", "band"], as_index=False)
            .agg(
                cell_score=("score", "sum"),
                out_of_beam_relations=("score", lambda s: int((s > 0).sum())),
                total_relations=("score", "size"),
                max_angle_diff=("angle_diff_deg", "max"),
                avg_angle_diff=("angle_diff_deg", "mean"),
                max_out_of_beam_score=("score", "max"),
            )
        )

        # Calculate out-of-beam ratio
        cell["out_of_beam_ratio"] = cell["out_of_beam_relations"] / cell["total_relations"].replace(0, 1)

        # Percentile threshold per band
        cell["flagged"] = False
        cell["threshold"] = 0.0
        cell["percentile"] = cfg.percentile

        for b, sub in cell.groupby("band"):
            if len(sub) == 0:
                continue
            thr = float(sub["cell_score"].quantile(cfg.percentile))
            idx = cell["band"] == b
            cell.loc[idx, "threshold"] = thr
            cell.loc[idx, "flagged"] = cell.loc[idx, "cell_score"] >= thr

        # Calculate confidence score (0-100%)
        # Based on: score strength, number of out-of-beam relations, angle deviation consistency
        cell["confidence"] = 0.0
        flagged_mask = cell["flagged"]

        if flagged_mask.any():
            # Normalize cell_score to 0-1 within flagged cells
            max_score = cell.loc[flagged_mask, "cell_score"].max()
            if max_score > 0:
                score_factor = (cell.loc[flagged_mask, "cell_score"] / max_score).clip(0, 1)
            else:
                score_factor = 0.0

            # Out-of-beam ratio factor (more out-of-beam = higher confidence)
            oob_factor = cell.loc[flagged_mask, "out_of_beam_ratio"].clip(0, 1)

            # Angle deviation factor (larger deviations = higher confidence)
            # Normalize by 180 degrees (max possible deviation)
            angle_factor = (cell.loc[flagged_mask, "avg_angle_diff"] / 180.0).clip(0, 1)

            # Weighted confidence: 40% score, 30% out-of-beam ratio, 30% angle deviation
            cell.loc[flagged_mask, "confidence"] = (
                0.40 * score_factor +
                0.30 * oob_factor +
                0.30 * angle_factor
            ) * 100

            cell["confidence"] = cell["confidence"].round(1)

        # Attach top-K suspicious relations per cell
        topk = (
            rel_scores[rel_scores["score"] > 0]
            .groupby("cell_name")
            .head(cfg.top_k_relations_per_cell)
            .copy()
        )
        topk["rel_str"] = (
            topk["to_cell_name"].astype(str)
            + " (d="
            + topk["distance"].round(0).astype(int).astype(str)
            + "m, score="
            + topk["score"].round(6).astype(str)
            + ")"
        )
        topk_agg = topk.groupby("cell_name")["rel_str"].apply(lambda x: " | ".join(x)).rename("top_suspicious_relations")
        cell = cell.merge(topk_agg, on="cell_name", how="left")
        cell["top_suspicious_relations"] = cell["top_suspicious_relations"].fillna("")

        # Site summary (per site+band)
        site = (
            cell.groupby(["site", "band"], as_index=False)
            .agg(
                flagged_cells=("flagged", "sum"),
                total_sectors=("cell_name", "nunique"),
                sum_cell_score=("cell_score", "sum"),
                max_cell_score=("cell_score", "max"),
                avg_out_of_beam_ratio=("out_of_beam_relations", lambda x: x.mean()),
            )
        )

        # Ratio-based classification for multi-sector sites
        site["flagged_ratio"] = site["flagged_cells"] / site["total_sectors"].replace(0, 1)

        def classify(row) -> str:
            flagged = int(row["flagged_cells"])
            total = int(row["total_sectors"])
            ratio = row["flagged_ratio"]

            if flagged == 0:
                return "No evidence"
            if ratio >= cfg.high_ratio_threshold:
                return f"High potential: {flagged}/{total} sectors flagged ({ratio:.0%})"
            if ratio >= cfg.medium_ratio_threshold:
                return f"Medium potential: {flagged}/{total} sectors flagged ({ratio:.0%})"
            if flagged >= 2:
                return f"Medium potential: {flagged} sectors flagged"
            return f"Low potential: {flagged} sector flagged"

        # Handle empty site DataFrame
        if len(site) > 0:
            site["classification"] = site.apply(classify, axis=1)
            site = site.sort_values(["flagged_ratio", "flagged_cells", "max_cell_score"], ascending=[False, False, False])
        else:
            site["classification"] = ""

        cell = cell.sort_values(["flagged", "cell_score"], ascending=[False, False])
        return cell, site

    def _cosectored_cross_band_anomalies(
        self,
        rel: pd.DataFrame,
        gis: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Secondary detector: compare co-sectored vs non-co-sectored cross-band relations.

        Returns a table of suspicious cases.
        """
        cfg = self.params

        # Only intra-site, cross-band (vectorized normalization)
        df = rel.copy()
        df["intra_site"] = _norm_yn_vec(df["intra_site"])
        df["intra_cell"] = _norm_yn_vec(df["intra_cell"])
        df = df[(df["intra_site"] == "y") & (df["band"].astype(str) != df["to_band"].astype(str))].copy()

        if df.empty:
            return pd.DataFrame()

        # Attach serving site via gis
        gis_s = gis[["cell_name", "site", "band"]].rename(columns={"band": "band_gis"})
        df = df.merge(gis_s, on="cell_name", how="left")

        # Strength metric for comparison
        strength = _pick_strength_series(df, cfg.cosectored_strength_col)
        df["strength"] = strength

        # Aggregate per serving cell and target band
        key = ["site", "cell_name", "band", "to_band"]
        agg = (
            df.groupby(key + ["intra_cell"], as_index=False)
            .agg(max_strength=("strength", "max"), count=("strength", "size"))
        )

        # Pivot intra_cell => columns
        piv = agg.pivot_table(
            index=key,
            columns="intra_cell",
            values="max_strength",
            aggfunc="max",
            fill_value=0.0
        ).reset_index()

        # Ensure both columns exist
        if "y" not in piv.columns:
            piv["y"] = 0.0
        if "n" not in piv.columns:
            piv["n"] = 0.0

        piv = piv.rename(columns={"y": "co_strength", "n": "non_strength"})
        piv["diff"] = 0.0
        denom = piv["co_strength"] + piv["non_strength"]
        piv.loc[denom > 0, "diff"] = 100.0 * (piv.loc[denom > 0, "co_strength"] - piv.loc[denom > 0, "non_strength"]) / denom[denom > 0]

        # Flagging rule
        piv["flag"] = (
            (piv[["co_strength", "non_strength"]].max(axis=1) >= cfg.cosectored_min_strength)
            & (piv["diff"] < cfg.cosectored_diff_thresh)
        )

        piv = piv.sort_values(["flag", "diff"], ascending=[False, True])
        return piv


def detect_crossed_feeders(
    relations_df: pd.DataFrame,
    gis_df: pd.DataFrame,
    params: Optional[CrossedFeederParams] = None,
    band_filter: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to detect crossed feeders.

    Args:
        relations_df: DataFrame with neighbour relations
        gis_df: DataFrame with cell GIS data
        params: Optional detection parameters
        band_filter: Optional filter on serving band

    Returns:
        Dictionary with detection results
    """
    detector = CrossedFeederDetector(params)
    return detector.detect(relations_df, gis_df, band_filter)

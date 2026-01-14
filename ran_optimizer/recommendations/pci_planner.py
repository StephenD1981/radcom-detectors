"""
PCI Confusion & Collision Planner - Handover Relation Based PCI Analysis.

This detector identifies PCI issues using directed handover relations:
1. PCI Confusion: Serving cell has 2+ neighbors sharing the same PCI (measurement ambiguity)
2. PCI Collision: Relevant cells share the same PCI (neighbors + optional 2-hop)
3. Mod 3 Conflict: Same PCI mod 3 causes PSS interference (3GPP TS 36.211 Section 6.11)
4. Mod 30 Conflict: Same PCI mod 30 causes RS interference (3GPP TS 36.211 Section 6.10)
5. Blacklist Suggestions: Identifies dead or low-activity relations to remove

Key Features:
- Traffic-weighted severity (based on actual HO volumes)
- Co-sectored cell coupling (force same PCI across bands)
- Conservative auto-blacklisting (dead both ways only)
- Distance-based collision filtering (default: 30 km radius)
- 2-hop collision detection for comprehensive analysis
- PCI range validation per 3GPP (LTE: 0-503, NR: 0-1007)
- Per-band PCI domain tracking for mixed LTE/NR networks
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional, Any

import numpy as np
import pandas as pd

from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)

# Constants per 3GPP TS 36.211 (LTE) and TS 38.211 (NR)
LTE_PCI_MAX = 503   # LTE: 504 PCIs (0-503)
NR_PCI_MAX = 1007   # NR: 1008 PCIs (0-1007)

# Performance constants
MAX_TWO_HOP_PAIRS = 1_000_000  # Memory safeguard for 2-hop pair generation
SHARE_EPSILON = 1e-9  # Prevent division by zero in share calculation
TWO_HOP_SEVERITY_FACTOR = 0.25  # Severity multiplier for 2-hop conflicts (lower priority than direct)

# Default configuration path
DEFAULT_CONFIG_PATH = "config/pci_planner_params.json"


def get_pci_max_for_band(band: str) -> int:
    """
    Get maximum valid PCI for a specific band per 3GPP specifications.

    Args:
        band: Band identifier (e.g., 'L800', 'N78', 'B1')

    Returns:
        Maximum PCI value (503 for LTE, 1007 for NR)
    """
    band_upper = str(band).strip().upper()
    # NR bands start with 'N' (e.g., N78, N260) or contain 'NR'
    if band_upper.startswith('N') or 'NR' in band_upper:
        return NR_PCI_MAX
    return LTE_PCI_MAX


def validate_pci_for_band(pci: int, band: str) -> bool:
    """
    Validate that PCI is within valid range for the band's technology.

    Args:
        pci: PCI value to validate
        band: Band identifier

    Returns:
        True if PCI is valid, False otherwise
    """
    if pci < 0:
        return False
    pci_max = get_pci_max_for_band(band)
    return pci <= pci_max


def log1p_safe(x: float) -> float:
    """Safely compute log(1 + x) for non-negative x."""
    return math.log(1.0 + max(0.0, float(x)))


def norm_yn(val) -> str:
    """Normalize yes/no values to 'y' or 'n'."""
    if pd.isna(val):
        return "n"
    s = str(val).strip().lower()
    return "y" if s in ("y", "yes", "true", "1") else "n"


def infer_pci_domain_from_bands(bands: Set[str]) -> int:
    """
    Infer PCI domain (LTE=503, NR=1007) from band names.
    LTE bands: Lxxx (L800, L1800)
    NR bands: Nxx (N78, N260)
    """
    for b in bands:
        bb = str(b).strip().upper()
        if bb.startswith("N") or "NR" in bb:
            return NR_PCI_MAX
    return LTE_PCI_MAX


def build_union_find_groups(pairs: List[Tuple[str, str]]) -> Tuple[Dict[str, int], Dict[int, Set[str]]]:
    """
    Union-Find algorithm to create co-sector groups from cell pairs.

    Uses iterative path compression to avoid recursion depth issues.

    Returns:
        - cell_to_gid: Maps each cell to its group ID
        - gid_to_members: Maps group ID to set of member cells
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        """Find root with iterative path compression."""
        if x not in parent:
            parent[x] = x
        # Find root (iterative)
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression (iterative) - point all nodes to root
        while parent[x] != root:
            next_x = parent[x]
            parent[x] = root
            x = next_x
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Build groups
    for a, b in pairs:
        union(a, b)

    # Assign compact group IDs
    root_to_gid: Dict[str, int] = {}
    cell_to_gid: Dict[str, int] = {}
    gid_to_members: Dict[int, Set[str]] = {}

    gid = 0
    for c in list(parent.keys()):
        r = find(c)
        if r not in root_to_gid:
            root_to_gid[r] = gid
            gid += 1
        g = root_to_gid[r]
        cell_to_gid[c] = g
        gid_to_members.setdefault(g, set()).add(c)

    return cell_to_gid, gid_to_members


@dataclass
class BlacklistDecision:
    """Represents a blacklist decision for a neighbor relation."""
    serving: str
    neighbor: str
    reason: str  # AUTO_DEAD_RELATION, SUGGEST_LOW_ACTIVITY_REVIEW, REJECT_*
    out_ho: float
    in_ho: float
    act_ho: float
    share: float
    confusion_pci: int
    confusion_group_size_before: int
    remaining_active_neighbors_after: int


@dataclass
class PCIPlannerParams:
    """Parameters for PCI planner detection."""
    couple_cosectors: bool = False
    min_active_neighbors_after_blacklist: int = 2
    max_collision_radius_m: float = 30000.0
    two_hop_factor: float = 0.25
    confusion_alpha: float = 1.0
    collision_beta: float = 1.0
    pci_change_cost: float = 5.0

    # Blacklist suggestion thresholds
    low_activity_max_act: float = 5.0
    low_activity_max_share: float = 0.001
    low_activity_max_single_dir: float = 5.0

    # Mod 3/30 interference detection per 3GPP TS 36.211
    # Mod 3: PSS (Primary Sync Signal) uses 3 sequences - same mod 3 causes cell search confusion
    # Mod 30: RS (Reference Signal) uses 30 cyclic shifts - same mod 30 causes RSRP errors
    detect_mod3_conflicts: bool = True   # PSS interference detection
    detect_mod30_conflicts: bool = False  # RS interference (disabled by default, many hits)
    mod3_severity_factor: float = 0.5    # Severity multiplier for mod 3 conflicts
    mod30_severity_factor: float = 0.3   # Severity multiplier for mod 30 conflicts

    # PCI validation
    validate_pci_range: bool = True      # Validate PCI within 3GPP range per band

    # Output filtering
    include_mod3_inter_site: bool = False  # If True, include inter-site mod3 in filtered output

    # Severity thresholds (0-1 scale, matching other detectors)
    severity_threshold_critical: float = 0.80
    severity_threshold_high: float = 0.60
    severity_threshold_medium: float = 0.40
    severity_threshold_low: float = 0.20

    # Intra-site severity adjustments (0-1 scale)
    # MOD3/MOD30 intra-site: MORE severe because cells are co-located, UEs see both with strong signals
    # EXACT intra-site: Slightly less severe because operators can manage internally
    # Note: MOD3 already has mod3_severity_factor=0.5 applied to base severity
    intra_site_bonus_mod3: float = 0.25    # MOD3 intra-site -> HIGH (co-located = severe interference)
    intra_site_bonus_mod30: float = 0.15   # MOD30 intra-site -> MEDIUM (co-located but less severe)
    intra_site_penalty_exact: float = 0.10 # EXACT intra-site -> still CRITICAL but slightly lower

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> 'PCIPlannerParams':
        """Load parameters from config file or use defaults."""
        import json
        from pathlib import Path

        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH

        try:
            path = Path(config_path)
            if not path.exists():
                logger.warning("config_file_not_found", path=config_path)
                return cls()

            with open(path, 'r') as f:
                config = json.load(f)

            params = config.get('default', config)

            return cls(
                couple_cosectors=bool(params.get('couple_cosectors', False)),
                min_active_neighbors_after_blacklist=int(params.get('min_active_neighbors_after_blacklist', 2)),
                max_collision_radius_m=float(params.get('max_collision_radius_m', 30000.0)),
                two_hop_factor=float(params.get('two_hop_factor', 0.25)),
                confusion_alpha=float(params.get('confusion_alpha', 1.0)),
                collision_beta=float(params.get('collision_beta', 1.0)),
                pci_change_cost=float(params.get('pci_change_cost', 5.0)),
                low_activity_max_act=float(params.get('low_activity_max_act', 5.0)),
                low_activity_max_share=float(params.get('low_activity_max_share', 0.001)),
                low_activity_max_single_dir=float(params.get('low_activity_max_single_dir', 5.0)),
                detect_mod3_conflicts=bool(params.get('detect_mod3_conflicts', True)),
                detect_mod30_conflicts=bool(params.get('detect_mod30_conflicts', False)),
                mod3_severity_factor=float(params.get('mod3_severity_factor', 0.5)),
                mod30_severity_factor=float(params.get('mod30_severity_factor', 0.3)),
                validate_pci_range=bool(params.get('validate_pci_range', True)),
                include_mod3_inter_site=bool(params.get('include_mod3_inter_site', False)),
                severity_threshold_critical=float(params.get('severity_threshold_critical', 0.80)),
                severity_threshold_high=float(params.get('severity_threshold_high', 0.60)),
                severity_threshold_medium=float(params.get('severity_threshold_medium', 0.40)),
                severity_threshold_low=float(params.get('severity_threshold_low', 0.20)),
                intra_site_bonus_mod3=float(params.get('intra_site_bonus_mod3', 0.25)),
                intra_site_bonus_mod30=float(params.get('intra_site_bonus_mod30', 0.15)),
                intra_site_penalty_exact=float(params.get('intra_site_penalty_exact', 0.10)),
            )
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("config_load_failed", error=str(e), using="defaults")
            return cls()


class PCIPlanner:
    """
    PCI Confusion & Collision Planner using directed handover relations.

    This class provides comprehensive PCI analysis including:
    - Confusion detection (serving cell with multiple neighbors on same PCI)
    - Collision detection (relevant cell pairs with same PCI)
    - Blacklist suggestions (dead/low-activity relations)

    Required columns (canonical names):
        cell_name, to_cell_name, pci, to_pci, band, to_band, weight
    Optional columns:
        distance, intra_site, intra_cell
    """

    def __init__(
        self,
        df: pd.DataFrame,
        params: Optional[PCIPlannerParams] = None,
        pci_max: Optional[int] = None,
    ):
        """
        Initialize the PCI Planner.

        Args:
            df: DataFrame with columns: cell_name, to_cell_name, pci, to_pci, band, to_band,
                weight, distance, intra_site, intra_cell
            params: Detection parameters (uses defaults if None)
            pci_max: Maximum PCI value (auto-detected from bands if None)
        """
        self.params = params or PCIPlannerParams()
        self.df_raw = df.copy()

        # Infer PCI domain if not specified
        if pci_max is None:
            bands = set(self.df_raw["band"].astype(str).unique())
            pci_max = infer_pci_domain_from_bands(bands)
        self.pci_max = pci_max

        # Configuration
        self.K = self.params.min_active_neighbors_after_blacklist
        self.max_collision_radius_m = self.params.max_collision_radius_m
        self.k2 = self.params.two_hop_factor

        # Co-sector groups
        self.cosector_gid: Dict[str, int] = {}
        self.cosector_members: Dict[int, Set[str]] = {}

        # Network structure
        self.df = pd.DataFrame()
        self.cells: Set[str] = set()
        self.nbrs_out: Dict[str, Set[str]] = {}  # serving -> set of neighbors
        self.rev_nbrs: Dict[str, Set[str]] = {}  # neighbor -> set of serving cells

        # Handover metrics (directed)
        self.out_ho: Dict[Tuple[str, str], float] = {}   # HO from S to N
        self.in_ho: Dict[Tuple[str, str], float] = {}    # HO from N to S
        self.act_ho: Dict[Tuple[str, str], float] = {}   # Total activity (out + in)
        self.share: Dict[Tuple[str, str], float] = {}    # Share of total from serving

        # Edge attributes
        self.intra_site_edge: Dict[Tuple[str, str], bool] = {}
        self.distance_m: Dict[Tuple[str, str], float] = {}

        # Cell attributes
        self.band_of: Dict[str, str] = {}
        self.cell_pci: Dict[str, int] = {}
        self.base_pci: Dict[str, int] = {}

        # Blacklist tracking
        self.blacklisted: Set[Tuple[str, str]] = set()

        # Collision relevance pairs (undirected, weighted)
        self.pair_w: Dict[Tuple[str, str], float] = {}
        self.pairs_by_cell: Dict[str, List[Tuple[str, float]]] = {}

        # Initialize
        self._prepare()

        logger.info(
            "pci_planner_initialized",
            cells=len(self.cells),
            pci_domain=f"0-{self.pci_max}",
            couple_cosectors=self.params.couple_cosectors,
            collision_radius_km=self.max_collision_radius_m / 1000.0,
        )

    def _prepare(self) -> None:
        """Prepare data structures from input DataFrame."""
        df = self.df_raw.copy()

        # Validate required columns
        required = ["cell_name", "to_cell_name", "pci", "to_pci", "band", "to_band", "weight"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Normalize flags using vectorized operations
        if "intra_site" in df.columns:
            df["intra_site"] = df["intra_site"].fillna("n").astype(str).str.strip().str.lower()
            df["intra_site"] = df["intra_site"].isin(["y", "yes", "true", "1"]).map({True: "y", False: "n"})
        else:
            df["intra_site"] = "n"

        if "intra_cell" in df.columns:
            df["intra_cell"] = df["intra_cell"].fillna("n").astype(str).str.strip().str.lower()
            df["intra_cell"] = df["intra_cell"].isin(["y", "yes", "true", "1"]).map({True: "y", False: "n"})
        else:
            df["intra_cell"] = "n"

        df["distance"] = pd.to_numeric(df.get("distance", 0.0), errors="coerce").fillna(0.0)

        # Build co-sector groups BEFORE removing intra_cell rows
        if self.params.couple_cosectors:
            link_df = df[(df["intra_site"] == "y") & (df["intra_cell"] == "y") & (df["distance"] == 0)]
            # Vectorized pair extraction
            cosector_pairs = list(zip(
                link_df["cell_name"].astype(str),
                link_df["to_cell_name"].astype(str)
            ))
            self.cosector_gid, self.cosector_members = build_union_find_groups(cosector_pairs)
            logger.info("cosector_groups_found", count=len(self.cosector_members))
        else:
            self.cosector_gid, self.cosector_members = {}, {}

        # Filter out intra_cell rows from mobility graph
        df = df[df["intra_cell"] != "y"].copy()

        # Ensure numeric types
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
        df["pci"] = pd.to_numeric(df["pci"], errors="coerce").fillna(-1).astype(int)
        df["to_pci"] = pd.to_numeric(df["to_pci"], errors="coerce").fillna(-1).astype(int)

        # Validate PCI ranges per 3GPP standards (per-band validation)
        if self.params.validate_pci_range:
            initial_count = len(df)

            # Validate serving cell PCI
            df["_pci_max"] = df["band"].apply(get_pci_max_for_band)
            invalid_serving_pci = (df["pci"] < 0) | (df["pci"] > df["_pci_max"])

            # Validate neighbor cell PCI
            df["_to_pci_max"] = df["to_band"].apply(get_pci_max_for_band)
            invalid_neighbor_pci = (df["to_pci"] < 0) | (df["to_pci"] > df["_to_pci_max"])

            invalid_pci_mask = invalid_serving_pci | invalid_neighbor_pci

            if invalid_pci_mask.any():
                invalid_count = invalid_pci_mask.sum()
                # Log examples of invalid PCIs
                invalid_examples = df[invalid_pci_mask].head(5)
                for _, row in invalid_examples.iterrows():
                    if row["pci"] < 0 or row["pci"] > row["_pci_max"]:
                        logger.warning(
                            "invalid_serving_pci",
                            cell=row["cell_name"],
                            pci=row["pci"],
                            band=row["band"],
                            max_valid=row["_pci_max"],
                        )
                    if row["to_pci"] < 0 or row["to_pci"] > row["_to_pci_max"]:
                        logger.warning(
                            "invalid_neighbor_pci",
                            cell=row["to_cell_name"],
                            pci=row["to_pci"],
                            band=row["to_band"],
                            max_valid=row["_to_pci_max"],
                        )

                df = df[~invalid_pci_mask].copy()
                logger.warning(
                    "pci_validation_filtered",
                    invalid_rows=invalid_count,
                    remaining_rows=len(df),
                    percent_filtered=round(100 * invalid_count / initial_count, 2) if initial_count > 0 else 0,
                )

            # Clean up temporary columns
            df = df.drop(columns=["_pci_max", "_to_pci_max"])

        self.df = df

        # Build main data structures using vectorized operations where possible

        # Get all unique cells
        self.cells = set(df["cell_name"].astype(str).unique()) | set(df["to_cell_name"].astype(str).unique())

        # Build cell properties from first occurrence (vectorized)
        serving_first = df.drop_duplicates("cell_name", keep="first")
        self.cell_pci = dict(zip(serving_first["cell_name"].astype(str), serving_first["pci"].astype(int)))
        self.base_pci = self.cell_pci.copy()
        self.band_of = dict(zip(serving_first["cell_name"].astype(str), serving_first["band"].astype(str)))

        # Add neighbor cell properties (only if not already present) - vectorized
        neighbor_first = df.drop_duplicates("to_cell_name", keep="first")
        for n, pci, band in zip(
            neighbor_first["to_cell_name"].astype(str),
            neighbor_first["to_pci"].astype(int),
            neighbor_first["to_band"].astype(str)
        ):
            if n not in self.cell_pci:
                # Log warning for invalid PCIs (negative values from fillna(-1))
                if pci < 0:
                    logger.debug("neighbor_cell_invalid_pci", cell=n, pci=pci)
                self.cell_pci[n] = pci
                self.base_pci[n] = pci
            if n not in self.band_of:
                self.band_of[n] = band

        # Build out_ho using groupby (sum weights per edge)
        edge_weights = df.groupby(["cell_name", "to_cell_name"])["weight"].sum()
        self.out_ho = {(str(s), str(n)): w for (s, n), w in edge_weights.items()}

        # Build neighbor sets using groupby
        for s, group in df.groupby("cell_name"):
            self.nbrs_out[str(s)] = set(group["to_cell_name"].astype(str).unique())
        for n, group in df.groupby("to_cell_name"):
            self.rev_nbrs[str(n)] = set(group["cell_name"].astype(str).unique())

        # Build edge attributes (take first occurrence for intra_site and distance) - vectorized
        edge_attrs = df.drop_duplicates(["cell_name", "to_cell_name"], keep="first")
        self.intra_site_edge = {
            (str(s), str(n)): is_intra == "y"
            for s, n, is_intra in zip(
                edge_attrs["cell_name"],
                edge_attrs["to_cell_name"],
                edge_attrs["intra_site"]
            )
        }
        self.distance_m = {
            (str(s), str(n)): float(d)
            for s, n, d in zip(
                edge_attrs["cell_name"],
                edge_attrs["to_cell_name"],
                edge_attrs["distance"]
            )
        }

        # Compute directed in_ho and act_ho
        for (S, N), out in self.out_ho.items():
            inv = self.out_ho.get((N, S), 0.0)
            self.in_ho[(S, N)] = inv
            self.act_ho[(S, N)] = out + inv

        # Compute shares per serving cell
        self._recompute_shares()

        # Build collision relevance pairs
        self._build_collision_pairs()

        logger.info(
            "pci_planner_prepared",
            cells=len(self.cells),
            relations=len(self.out_ho),
            collision_pairs=len(self.pair_w),
        )

    def _recompute_shares(self) -> None:
        """Recompute share of each neighbor from serving cell perspective."""
        for S in self.nbrs_out.keys():
            tot = 0.0
            for N in self.nbrs_out[S]:
                if (S, N) in self.blacklisted:
                    continue
                tot += self.act_ho.get((S, N), 0.0)

            for N in self.nbrs_out[S]:
                if (S, N) in self.blacklisted:
                    continue
                act = self.act_ho.get((S, N), 0.0)
                self.share[(S, N)] = act / tot if tot > 0 else 0.0

    def _build_collision_pairs(self) -> None:
        """
        Build collision relevance pairs using 1-hop and optional 2-hop neighbors.
        Uses distance filtering and logarithmic weighting of HO volumes.
        """
        # 1-hop pairs (direct neighbors)
        w1: Dict[Tuple[str, str], float] = {}
        for (S, N), out in self.out_ho.items():
            if (S, N) in self.blacklisted:
                continue

            # Distance filter
            dist_m = self.distance_m.get((S, N), 0.0)
            if dist_m > self.max_collision_radius_m:
                continue

            # Create undirected pair
            a, b = (S, N) if S < N else (N, S)
            out_ba = self.out_ho.get((N, S), 0.0)
            base = log1p_safe(out + out_ba)
            w1[(a, b)] = max(w1.get((a, b), 0.0), base)

        # 2-hop pairs (neighbors of neighbors) with memory safeguard
        w2: Dict[Tuple[str, str], float] = {}
        if self.k2 > 0:
            pairs_added = 0
            limit_reached = False
            for i in self.cells:
                if limit_reached:
                    break
                for k in self.nbrs_out.get(i, set()):
                    if limit_reached:
                        break
                    if (i, k) in self.blacklisted:
                        continue
                    if self.distance_m.get((i, k), 0.0) > self.max_collision_radius_m:
                        continue

                    for j in self.nbrs_out.get(k, set()):
                        if j == i or (k, j) in self.blacklisted:
                            continue
                        if self.distance_m.get((k, j), 0.0) > self.max_collision_radius_m:
                            continue

                        # Get weights for both hops
                        a1, b1 = (i, k) if i < k else (k, i)
                        a2, b2 = (k, j) if k < j else (j, k)
                        w_ik = w1.get((a1, b1), 0.0)
                        w_kj = w1.get((a2, b2), 0.0)

                        if w_ik <= 0 or w_kj <= 0:
                            continue

                        # Create undirected 2-hop pair
                        a, b = (i, j) if i < j else (j, i)
                        if (a, b) not in w2:
                            pairs_added += 1
                            # Memory safeguard: limit 2-hop pairs
                            if pairs_added > MAX_TWO_HOP_PAIRS:
                                logger.warning(
                                    "two_hop_pairs_limit_reached",
                                    limit=MAX_TWO_HOP_PAIRS,
                                    message="Truncating 2-hop pair generation to prevent memory exhaustion"
                                )
                                limit_reached = True
                                break
                        w2[(a, b)] = w2.get((a, b), 0.0) + self.k2 * min(w_ik, w_kj)

        # Combine 1-hop and 2-hop weights
        self.pair_w = {}
        for k, v in w1.items():
            self.pair_w[k] = self.pair_w.get(k, 0.0) + v
        for k, v in w2.items():
            self.pair_w[k] = self.pair_w.get(k, 0.0) + v

        # Index by cell for fast lookup
        self.pairs_by_cell = {c: [] for c in self.cells}
        for (a, b), w in self.pair_w.items():
            self.pairs_by_cell[a].append((b, w))
            self.pairs_by_cell[b].append((a, w))

    def detect_confusions(self) -> pd.DataFrame:
        """
        Detect PCI confusions: serving cells with 2+ neighbors sharing the same PCI on the same band.

        Returns:
            DataFrame with columns: serving, confusion_pci, group_size, neighbors,
            severity_act_sum_excl_max, min_act, max_act
        """
        rows = []
        for S, nbrs in self.nbrs_out.items():
            # Group neighbors by (PCI, band) tuple - confusions only valid on same band
            groups: Dict[Tuple[int, str], List[str]] = {}
            for N in nbrs:
                if (S, N) in self.blacklisted:
                    continue
                p = self.cell_pci.get(N, -1)
                band_N = self.band_of.get(N, "")
                if p >= 0 and band_N:
                    groups.setdefault((p, band_N), []).append(N)

            # Find confusions (2+ neighbors with same PCI on same band)
            for (p, band), members in groups.items():
                if len(members) < 2:
                    continue

                # Calculate severity (sum of all except max)
                acts = [(N, self.act_ho.get((S, N), 0.0), self.share.get((S, N), 0.0)) for N in members]
                acts_sorted = sorted(acts, key=lambda x: x[1])
                severity = sum(a for _, a, _ in acts_sorted[:-1])  # All except strongest

                rows.append({
                    "serving": S,
                    "band": band,
                    "confusion_pci": p,
                    "group_size": len(members),
                    "neighbors": ",".join(members),
                    "severity_act_sum_excl_max": severity,
                    "min_act": acts_sorted[0][1],
                    "max_act": acts_sorted[-1][1],
                })

        df_result = pd.DataFrame(rows)
        if not df_result.empty:
            # Normalize severity to 0-1 using percentile-based scaling
            # Use 95th percentile as max to avoid outlier distortion
            severity_col = df_result['severity_act_sum_excl_max']
            severity_max = severity_col.quantile(0.95) if len(severity_col) > 1 else severity_col.max()
            severity_max = max(severity_max, 1e-9)  # Prevent division by zero

            df_result['severity_score'] = (severity_col / severity_max).clip(0, 1)

            # Add group_size bonus: larger groups are more severe
            # Each additional member above 2 adds 0.1 to severity (capped)
            group_bonus = ((df_result['group_size'] - 2) * 0.1).clip(0, 0.3)
            df_result['severity_score'] = (df_result['severity_score'] + group_bonus).clip(0, 1)

            # Categorize severity using standard thresholds
            conditions = [
                df_result['severity_score'] >= self.params.severity_threshold_critical,
                df_result['severity_score'] >= self.params.severity_threshold_high,
                df_result['severity_score'] >= self.params.severity_threshold_medium,
                df_result['severity_score'] >= self.params.severity_threshold_low,
            ]
            choices = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            df_result['severity_category'] = np.select(conditions, choices, default='MINIMAL')

            df_result = df_result.sort_values(["severity_score"], ascending=False)

        logger.info("pci_confusions_detected", count=len(df_result))
        return df_result

    def _extract_site_id(self, cell_name: str) -> str:
        """
        Extract site ID from cell name using common naming conventions.

        Common patterns:
        - CK186K2 -> CK186 (site = prefix before sector/band indicator)
        - SITE_CELL1 -> SITE
        - ABC123L1 -> ABC123

        This extracts the site portion by removing the last 2 characters
        (typically sector number + band indicator or just sector).
        """
        if not cell_name or len(cell_name) < 3:
            return cell_name
        # Remove last 2 chars (sector + band indicator like K2, L1, H3)
        return cell_name[:-2]

    def _is_intra_site_pair(self, cell_a: str, cell_b: str) -> bool:
        """
        Determine if two cells are on the same site (intra-site).

        Uses multiple methods:
        1. Check if there's a direct intra_site edge between them
        2. Check if there's a reverse intra_site edge
        3. Fall back to site ID extraction from cell names
        """
        # Method 1: Check direct edge intra_site flag
        if self.intra_site_edge.get((cell_a, cell_b), False):
            return True
        if self.intra_site_edge.get((cell_b, cell_a), False):
            return True

        # Method 2: Extract site IDs from cell names and compare
        site_a = self._extract_site_id(cell_a)
        site_b = self._extract_site_id(cell_b)
        return site_a == site_b

    def detect_collisions(self) -> pd.DataFrame:
        """
        Detect PCI collisions and interference: relevant cell pairs with PCI conflicts.

        Detects three types of PCI issues per 3GPP TS 36.211:
        1. Exact collision: Same PCI on same band (most severe)
        2. Mod 3 conflict: Same PCI mod 3 causes PSS (Primary Sync Signal) interference
        3. Mod 30 conflict: Same PCI mod 30 causes RS (Reference Signal) interference

        Severity scoring considers:
        - Conflict type: exact > mod3 > mod30
        - Hop type: 1-hop > 2-hop (direct neighbors more severe)
        - Site relationship: inter-site > intra-site (operators can manage intra-site internally)

        Returns:
            DataFrame with columns: cell_a, cell_b, pci_a, pci_b, band, conflict_type,
            hop_type, pair_weight, severity, site_a, site_b, intra_site
        """
        rows = []
        check_mod3 = self.params.detect_mod3_conflicts
        check_mod30 = self.params.detect_mod30_conflicts

        for (a, b), w in self.pair_w.items():
            pa = self.cell_pci.get(a, -1)
            pb = self.cell_pci.get(b, -2)
            band_a = self.band_of.get(a, "")
            band_b = self.band_of.get(b, "")

            # Skip invalid PCIs or different bands
            if pa < 0 or pb < 0:
                continue
            if not band_a or not band_b or band_a != band_b:
                continue

            # Determine hop type (1-hop or 2-hop)
            is_direct = b in self.nbrs_out.get(a, set()) or a in self.nbrs_out.get(b, set())
            hop_type = "1-hop" if is_direct else "2-hop"

            # Determine if intra-site (same site) - important for severity
            is_intra_site = self._is_intra_site_pair(a, b)
            site_a = self._extract_site_id(a)
            site_b = self._extract_site_id(b)

            conflict_type = None
            severity_factor = 1.0

            # Check for exact collision (highest priority)
            if pa == pb:
                conflict_type = "exact"
                severity_factor = 1.0
            # Check for mod 3 conflict (PSS interference) per TS 36.211 Section 6.11
            # NOTE: Mod 3 only applies to INTRA-SITE cells (co-located cells cause PSS interference)
            elif check_mod3 and is_intra_site and (pa % 3 == pb % 3):
                conflict_type = "mod3"
                severity_factor = self.params.mod3_severity_factor
            # Check for mod 30 conflict (RS interference) per TS 36.211 Section 6.10
            # NOTE: Mod 30 only applies to INTRA-SITE cells (co-located cells cause RS interference)
            elif check_mod30 and is_intra_site and (pa % 30 == pb % 30):
                conflict_type = "mod30"
                severity_factor = self.params.mod30_severity_factor

            if conflict_type:
                # Calculate severity: weight * factor, with 2-hop having lower severity
                hop_factor = 1.0 if is_direct else TWO_HOP_SEVERITY_FACTOR
                severity = w * severity_factor * hop_factor

                rows.append({
                    "cell_a": a,
                    "cell_b": b,
                    "pci_a": pa,
                    "pci_b": pb,
                    "band": band_a,
                    "conflict_type": conflict_type,
                    "hop_type": hop_type,
                    "pair_weight": w,
                    "severity": severity,
                    "site_a": site_a,
                    "site_b": site_b,
                    "intra_site": is_intra_site,
                })

        df_result = pd.DataFrame(rows)
        if not df_result.empty:
            # === SEVERITY SCORING LOGIC ===
            # Priority order for severity (highest to lowest):
            # 1. EXACT inter-site 1-hop = CRITICAL (most severe)
            # 2. EXACT intra-site 1-hop = CRITICAL (same PCI is always critical)
            # 3. MOD3 intra-site 1-hop = HIGH/CRITICAL (cells co-located, UEs see both with strong signals)
            # 4. MOD3 inter-site 1-hop = MEDIUM (cells are distant, less interference impact)
            # 5. 2-hop collisions = lower than 1-hop equivalents
            # 6. MOD30 = LOW/MINIMAL (least severe)

            # Base conflict type bonus (before site adjustment)
            # Exact collisions are fundamentally more severe than mod conflicts
            conflict_base_bonus = df_result['conflict_type'].map({
                'exact': 0.35,   # Exact collision base bonus (increased)
                'mod3': 0.05,    # Mod 3 lower base - site adjustment will differentiate
                'mod30': 0.0,    # Mod 30 no bonus
            }).fillna(0.0)

            # Apply hop bonus based on conflict type
            # 1-hop is more severe, but 2-hop EXACT collisions still need reasonable priority
            # 2-hop MOD conflicts can be lower priority (already reflected in base severity)
            def calculate_hop_bonus(row):
                if row['hop_type'] == '1-hop':
                    return 0.15
                elif row['hop_type'] == '2-hop' and row['conflict_type'] == 'exact':
                    # 2-hop EXACT still significant - give partial bonus
                    return 0.10
                else:
                    # 2-hop MOD conflicts - no bonus
                    return 0.0

            hop_bonus = df_result.apply(calculate_hop_bonus, axis=1)

            # Apply intra-site adjustment: can be BONUS (positive) or PENALTY (negative)
            # MOD3/MOD30 intra-site: BONUS because cells are co-located, causing severe interference
            # EXACT intra-site: Small PENALTY because operators can manage (but still critical)
            intra_site_adjustment = df_result.apply(
                lambda row: self._calculate_intra_site_adjustment(
                    row['conflict_type'],
                    row['intra_site']
                ),
                axis=1
            )

            # Calculate structural score with bonuses and adjustments
            # This determines the baseline severity based on conflict characteristics
            structural_score = conflict_base_bonus + hop_bonus + intra_site_adjustment

            # Now normalize the base severity (traffic-weighted) to 0-1 scale
            # Use 95th percentile as max to avoid outlier distortion
            severity_col = df_result['severity']
            severity_max = severity_col.quantile(0.95) if len(severity_col) > 1 else severity_col.max()
            severity_max = max(severity_max, 1e-9)  # Prevent division by zero

            # Normalized traffic weight contribution (0 to 0.35 scale)
            # Traffic is less important than structural factors for severity category
            traffic_score = (severity_col / severity_max).clip(0, 1) * 0.35

            # Final score = structural component (determines category) + traffic (fine-tuning)
            # Structural scores by scenario:
            # - EXACT inter-site 1-hop: 0.35 + 0.15 + 0.0 = 0.50 base (+ up to 0.35 traffic = 0.85 CRITICAL)
            # - EXACT intra-site 1-hop: 0.35 + 0.15 - 0.10 = 0.40 base (+ up to 0.35 traffic = 0.75 CRITICAL)
            # - MOD3 intra-site 1-hop: 0.05 + 0.15 + 0.25 = 0.45 base (+ up to 0.35 traffic = 0.80 CRITICAL/HIGH)
            # - MOD3 inter-site 1-hop: 0.05 + 0.15 + 0.0 = 0.20 base (+ up to 0.35 traffic = 0.55 MEDIUM)
            # - MOD3 inter-site 2-hop: 0.05 + 0.0 + 0.0 = 0.05 base (+ up to 0.35 traffic = 0.40 MEDIUM/LOW)
            df_result['severity_score'] = (structural_score + traffic_score).clip(0, 1)

            # Categorize severity using standard thresholds
            conditions = [
                df_result['severity_score'] >= self.params.severity_threshold_critical,
                df_result['severity_score'] >= self.params.severity_threshold_high,
                df_result['severity_score'] >= self.params.severity_threshold_medium,
                df_result['severity_score'] >= self.params.severity_threshold_low,
            ]
            choices = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            df_result['severity_category'] = np.select(conditions, choices, default='MINIMAL')

            df_result = df_result.sort_values(["severity_score"], ascending=False)

        # Log counts by conflict type and site relationship
        if not df_result.empty:
            counts = df_result["conflict_type"].value_counts().to_dict()
            intra_counts = df_result[df_result["intra_site"]]["conflict_type"].value_counts().to_dict()
            inter_counts = df_result[~df_result["intra_site"]]["conflict_type"].value_counts().to_dict()
        else:
            counts = {}
            intra_counts = {}
            inter_counts = {}

        logger.info(
            "pci_collisions_detected",
            total=len(df_result),
            exact=counts.get("exact", 0),
            mod3=counts.get("mod3", 0),
            mod30=counts.get("mod30", 0),
            intra_site_mod3=intra_counts.get("mod3", 0),
            inter_site_mod3=inter_counts.get("mod3", 0),
        )
        return df_result

    def _calculate_intra_site_adjustment(self, conflict_type: str, is_intra_site: bool) -> float:
        """
        Calculate severity adjustment for intra-site vs inter-site collisions.

        Returns positive value (BONUS) or negative value (PENALTY):
        - MOD3/MOD30 intra-site: BONUS (positive) because cells are co-located,
          UEs see both with strong signals causing severe interference
        - EXACT intra-site: Small PENALTY (negative) because operators can
          manage their own site, but still critical
        - Inter-site: No adjustment (0.0)

        Adjustment amounts (configurable via PCIPlannerParams):
        - MOD3 intra-site: default +0.25 bonus (HIGH/CRITICAL severity)
        - MOD30 intra-site: default +0.15 bonus (MEDIUM severity)
        - EXACT intra-site: default -0.10 penalty (still CRITICAL, slightly lower)
        - Inter-site: 0.0 (no adjustment)
        """
        if not is_intra_site:
            return 0.0

        # Intra-site adjustments by conflict type (from configurable params)
        # Positive = bonus (more severe), Negative = penalty (less severe)
        adjustments = {
            'mod3': self.params.intra_site_bonus_mod3,    # +0.25 BONUS
            'mod30': self.params.intra_site_bonus_mod30,  # +0.15 BONUS
            'exact': -self.params.intra_site_penalty_exact,  # -0.10 PENALTY
        }
        return adjustments.get(conflict_type, 0.0)

    def suggest_blacklists(self) -> Tuple[pd.DataFrame, Set[Tuple[str, str]]]:
        """
        Suggest blacklisting for weak/dead neighbor relations in confusion groups.

        Conservative approach:
        - AUTO: Only if dead both directions (out=0 AND in=0)
        - SUGGEST: Low activity for manual review
        - REJECT: If intra-site or would leave too few neighbors

        Returns:
            Tuple of (decisions DataFrame, auto_apply set of edges)
        """
        decisions: List[BlacklistDecision] = []
        auto_apply: Set[Tuple[str, str]] = set()

        for S, nbrs in self.nbrs_out.items():
            # Group neighbors by (PCI, band) - same as confusion detection
            groups: Dict[Tuple[int, str], List[str]] = {}
            for N in nbrs:
                if (S, N) in self.blacklisted:
                    continue
                p = self.cell_pci.get(N, -1)
                band_N = self.band_of.get(N, "")
                if p >= 0 and band_N:
                    groups.setdefault((p, band_N), []).append(N)

            # Process confusions only (2+ neighbors with same PCI on same band)
            for (p, band), members in groups.items():
                if len(members) < 2:
                    continue

                # Sort by activity (weakest first)
                members_sorted = sorted(members, key=lambda n: self.act_ho.get((S, n), 0.0))
                remaining = members_sorted[:]

                def active_neighbor_count_after_remove(rem_n: str) -> int:
                    """Count active neighbors after removing rem_n."""
                    cnt = 0
                    for X in nbrs:
                        if (S, X) in self.blacklisted:
                            continue
                        if X == rem_n:
                            continue
                        if self.act_ho.get((S, X), 0.0) > 0:
                            cnt += 1
                    return cnt

                # Consider blacklisting weakest members
                for N in members_sorted:
                    if len(remaining) <= 1:  # Keep at least one
                        break

                    # Check constraints
                    out = self.out_ho.get((S, N), 0.0)
                    inn = self.in_ho.get((S, N), 0.0)
                    act = out + inn
                    sh = self.share.get((S, N), 0.0)
                    remain_active = active_neighbor_count_after_remove(N)

                    # REJECT: Intra-site (never blacklist)
                    if self.intra_site_edge.get((S, N), False):
                        decisions.append(BlacklistDecision(
                            serving=S, neighbor=N, reason="REJECT_INTRA_SITE",
                            out_ho=out, in_ho=inn, act_ho=act, share=sh,
                            confusion_pci=p, confusion_group_size_before=len(members),
                            remaining_active_neighbors_after=remain_active
                        ))
                        continue

                    # REJECT: Would leave too few neighbors
                    if remain_active < self.K:
                        decisions.append(BlacklistDecision(
                            serving=S, neighbor=N, reason="REJECT_MIN_NEIGHBORS",
                            out_ho=out, in_ho=inn, act_ho=act, share=sh,
                            confusion_pci=p, confusion_group_size_before=len(members),
                            remaining_active_neighbors_after=remain_active
                        ))
                        continue

                    # AUTO: Dead both directions
                    if out == 0 and inn == 0:
                        decisions.append(BlacklistDecision(
                            serving=S, neighbor=N, reason="AUTO_DEAD_RELATION",
                            out_ho=out, in_ho=inn, act_ho=act, share=sh,
                            confusion_pci=p, confusion_group_size_before=len(members),
                            remaining_active_neighbors_after=remain_active
                        ))
                        auto_apply.add((S, N))
                        if N in remaining:
                            remaining.remove(N)
                        continue

                    # SUGGEST: Low activity for review
                    cfg = self.params
                    if (act <= cfg.low_activity_max_act and
                        sh <= cfg.low_activity_max_share and
                        max(out, inn) <= cfg.low_activity_max_single_dir):
                        decisions.append(BlacklistDecision(
                            serving=S, neighbor=N, reason="SUGGEST_LOW_ACTIVITY_REVIEW",
                            out_ho=out, in_ho=inn, act_ho=act, share=sh,
                            confusion_pci=p, confusion_group_size_before=len(members),
                            remaining_active_neighbors_after=remain_active
                        ))
                        continue

                    # REJECT: Active relation (don't blacklist)
                    decisions.append(BlacklistDecision(
                        serving=S, neighbor=N, reason="REJECT_ACTIVE_RELATION",
                        out_ho=out, in_ho=inn, act_ho=act, share=sh,
                        confusion_pci=p, confusion_group_size_before=len(members),
                        remaining_active_neighbors_after=remain_active
                    ))

        out_df = pd.DataFrame([d.__dict__ for d in decisions])
        if not out_df.empty:
            out_df = out_df.sort_values(["reason", "act_ho"], ascending=[True, True])

        logger.info(
            "pci_blacklist_analysis",
            decisions=len(decisions),
            auto=len(auto_apply),
            suggest=len([d for d in decisions if d.reason == 'SUGGEST_LOW_ACTIVITY_REVIEW']),
        )

        return out_df, auto_apply

    def apply_blacklists(self, edges: Set[Tuple[str, str]]) -> None:
        """
        Apply blacklist to remove edges from the network.
        Recomputes shares and collision pairs after removal.
        """
        if not edges:
            return

        self.blacklisted |= set(edges)
        self._recompute_shares()
        self._build_collision_pairs()

        logger.info("blacklists_applied", count=len(edges), total_blacklisted=len(self.blacklisted))

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the network and detected issues."""
        confusions = self.detect_confusions()
        collisions = self.detect_collisions()

        # Break down collisions by conflict type
        collision_by_type = {}
        if not collisions.empty:
            collision_by_type = collisions["conflict_type"].value_counts().to_dict()

        return {
            "total_cells": len(self.cells),
            "total_relations": len(self.out_ho),
            "blacklisted_relations": len(self.blacklisted),
            "confusion_count": len(confusions),
            "collision_count": len(collisions),
            "collision_exact": collision_by_type.get("exact", 0),
            "collision_mod3": collision_by_type.get("mod3", 0),
            "collision_mod30": collision_by_type.get("mod30", 0),
            "cosector_groups": len(self.cosector_members) if self.params.couple_cosectors else 0,
            "collision_radius_km": self.max_collision_radius_m / 1000.0,
            "pci_domain": f"0-{self.pci_max}",
            "mod3_detection_enabled": self.params.detect_mod3_conflicts,
            "mod30_detection_enabled": self.params.detect_mod30_conflicts,
        }


def detect_pci_issues(
    impacts_df: pd.DataFrame,
    params: Optional[PCIPlannerParams] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to detect PCI issues using relation-based analysis.

    Args:
        impacts_df: DataFrame with cell impacts/relations
        params: Optional detection parameters

    Returns:
        Dict with keys: confusion, collision, blacklist
    """
    planner = PCIPlanner(impacts_df, params)

    # Detect issues
    confusion = planner.detect_confusions()
    collision = planner.detect_collisions()
    blacklist_df, _ = planner.suggest_blacklists()

    return {
        'confusion': confusion,
        'collision': collision,
        'blacklist': blacklist_df,
    }

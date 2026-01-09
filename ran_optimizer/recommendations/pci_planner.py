"""
PCI Confusion & Collision Planner - Handover Relation Based PCI Analysis.

This detector identifies PCI issues using directed handover relations:
1. PCI Confusion: Serving cell has 2+ neighbors sharing the same PCI (measurement ambiguity)
2. PCI Collision: Relevant cells share the same PCI (neighbors + optional 2-hop)
3. Blacklist Suggestions: Identifies dead or low-activity relations to remove

Key Features:
- Traffic-weighted severity (based on actual HO volumes)
- Co-sectored cell coupling (force same PCI across bands)
- Conservative auto-blacklisting (dead both ways only)
- Distance-based collision filtering (default: 30 km radius)
- 2-hop collision detection for comprehensive analysis
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional, Any

import pandas as pd

from ran_optimizer.utils.logging_config import get_logger

logger = get_logger(__name__)

# Constants
LTE_PCI_MAX = 503
NR_PCI_MAX = 1007


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

    Returns:
        - cell_to_gid: Maps each cell to its group ID
        - gid_to_members: Maps group ID to set of member cells
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

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

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> 'PCIPlannerParams':
        """Load parameters from config file or use defaults."""
        if config_path is None:
            config_path = "config/pci_planner_params.json"

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
                couple_cosectors=params.get('couple_cosectors', False),
                min_active_neighbors_after_blacklist=params.get('min_active_neighbors_after_blacklist', 2),
                max_collision_radius_m=params.get('max_collision_radius_m', 30000.0),
                two_hop_factor=params.get('two_hop_factor', 0.25),
                confusion_alpha=params.get('confusion_alpha', 1.0),
                collision_beta=params.get('collision_beta', 1.0),
                pci_change_cost=params.get('pci_change_cost', 5.0),
                low_activity_max_act=params.get('low_activity_max_act', 5.0),
                low_activity_max_share=params.get('low_activity_max_share', 0.001),
                low_activity_max_single_dir=params.get('low_activity_max_single_dir', 5.0),
            )
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
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
            logger.info(f"Found {len(self.cosector_members)} co-sector groups")
        else:
            self.cosector_gid, self.cosector_members = {}, {}

        # Filter out intra_cell rows from mobility graph
        df = df[df["intra_cell"] != "y"].copy()

        # Ensure numeric types
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
        df["pci"] = pd.to_numeric(df["pci"], errors="coerce").fillna(-1).astype(int)
        df["to_pci"] = pd.to_numeric(df["to_pci"], errors="coerce").fillna(-1).astype(int)

        self.df = df

        # Build main data structures using vectorized operations where possible

        # Get all unique cells
        self.cells = set(df["cell_name"].astype(str).unique()) | set(df["to_cell_name"].astype(str).unique())

        # Build cell properties from first occurrence (vectorized)
        serving_first = df.drop_duplicates("cell_name", keep="first")
        self.cell_pci = dict(zip(serving_first["cell_name"].astype(str), serving_first["pci"].astype(int)))
        self.base_pci = self.cell_pci.copy()
        self.band_of = dict(zip(serving_first["cell_name"].astype(str), serving_first["band"].astype(str)))

        # Add neighbor cell properties (only if not already present)
        neighbor_first = df.drop_duplicates("to_cell_name", keep="first")
        for _, row in neighbor_first.iterrows():
            n = str(row["to_cell_name"])
            if n not in self.cell_pci:
                self.cell_pci[n] = int(row["to_pci"])
                self.base_pci[n] = int(row["to_pci"])
            if n not in self.band_of:
                self.band_of[n] = str(row["to_band"])

        # Build out_ho using groupby (sum weights per edge)
        edge_weights = df.groupby(["cell_name", "to_cell_name"])["weight"].sum()
        self.out_ho = {(str(s), str(n)): w for (s, n), w in edge_weights.items()}

        # Build neighbor sets using groupby
        for s, group in df.groupby("cell_name"):
            self.nbrs_out[str(s)] = set(group["to_cell_name"].astype(str).unique())
        for n, group in df.groupby("to_cell_name"):
            self.rev_nbrs[str(n)] = set(group["cell_name"].astype(str).unique())

        # Build edge attributes (take first occurrence for intra_site and distance)
        edge_attrs = df.drop_duplicates(["cell_name", "to_cell_name"], keep="first")
        self.intra_site_edge = {
            (str(row["cell_name"]), str(row["to_cell_name"])): row["intra_site"] == "y"
            for _, row in edge_attrs.iterrows()
        }
        self.distance_m = {
            (str(row["cell_name"]), str(row["to_cell_name"])): float(row["distance"])
            for _, row in edge_attrs.iterrows()
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
                self.share[(S, N)] = act / (tot + 1e-9)

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

        # 2-hop pairs (neighbors of neighbors)
        w2: Dict[Tuple[str, str], float] = {}
        if self.k2 > 0:
            for i in self.cells:
                for k in self.nbrs_out.get(i, set()):
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
            df_result = df_result.sort_values(["severity_act_sum_excl_max"], ascending=False)

        logger.info(f"pci_confusions_detected", count=len(df_result))
        return df_result

    def detect_collisions(self) -> pd.DataFrame:
        """
        Detect PCI collisions: relevant cell pairs sharing the same PCI on the same band.

        Returns:
            DataFrame with columns: cell_a, cell_b, pci, band, collision_type, pair_weight
            collision_type: "1-hop" for direct neighbors, "2-hop" for indirect neighbors
        """
        rows = []
        for (a, b), w in self.pair_w.items():
            pa = self.cell_pci.get(a, -1)
            pb = self.cell_pci.get(b, -2)
            band_a = self.band_of.get(a, "")
            band_b = self.band_of.get(b, "")
            # Collision only valid when both cells on same band
            if pa >= 0 and pa == pb and band_a and band_b and band_a == band_b:
                # Determine if this is a 1-hop (direct neighbor) or 2-hop collision
                is_direct = b in self.nbrs_out.get(a, set()) or a in self.nbrs_out.get(b, set())
                collision_type = "1-hop" if is_direct else "2-hop"

                rows.append({
                    "cell_a": a,
                    "cell_b": b,
                    "pci": pa,
                    "band": band_a,
                    "collision_type": collision_type,
                    "pair_weight": w
                })

        df_result = pd.DataFrame(rows)
        if not df_result.empty:
            df_result = df_result.sort_values(["pair_weight"], ascending=False)

        logger.info(f"pci_collisions_detected", count=len(df_result))
        return df_result

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

        logger.info(f"Applied {len(edges)} blacklists, total blacklisted: {len(self.blacklisted)}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the network and detected issues."""
        confusions = self.detect_confusions()
        collisions = self.detect_collisions()

        return {
            "total_cells": len(self.cells),
            "total_relations": len(self.out_ho),
            "blacklisted_relations": len(self.blacklisted),
            "confusion_count": len(confusions),
            "collision_count": len(collisions),
            "cosector_groups": len(self.cosector_members) if self.params.couple_cosectors else 0,
            "collision_radius_km": self.max_collision_radius_m / 1000.0,
            "pci_domain": f"0-{self.pci_max}",
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

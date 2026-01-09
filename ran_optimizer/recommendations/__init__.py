"""
Recommendation modules for RAN optimization.

Provides algorithms for generating network optimization recommendations
such as tilt adjustments, power changes, and coverage improvements.

Detector Algorithms:
    Base 4:
    - Overshooters: Cells reaching too far, causing interference
    - Undershooters: Cells with insufficient coverage reach
    - Low Coverage: Areas with weak signal (RSRP < threshold)
    - No Coverage: Areas with no detected signal

    Additional 4:
    - PCI Conflict: Cells sharing same PCI with overlapping coverage
    - Crossed Feeder: Sector/feeder swaps detected via out-of-beam neighbor relations
    - CA Imbalance: Capacity band insufficient vs coverage band (needs 70% overlap)
    - High Interference: Excessive overlap causing poor SINR
"""
from ran_optimizer.recommendations.overshooters import (
    OvershooterDetector,
    OvershooterParams,
    detect_overshooting_cells,
)
from ran_optimizer.recommendations.undershooters import (
    UndershooterDetector,
    UndershooterParams,
    detect_undershooting_cells,
    detect_undershooting_with_environment_awareness,
    compare_undershooting_detection_approaches,
)
from ran_optimizer.recommendations.pci_conflict import (
    PCIConflictDetector,
    PCIConflictParams,
    detect_pci_conflicts,
)
from ran_optimizer.recommendations.crossed_feeder import (
    CrossedFeederDetector,
    CrossedFeederParams,
    detect_crossed_feeders,
)
from ran_optimizer.recommendations.ca_imbalance import (
    CAImbalanceDetector,
    CAImbalanceParams,
    CAPairConfig,
    detect_ca_imbalance,
)
from ran_optimizer.recommendations.interference import (
    InterferenceDetector,
    InterferenceParams,
    detect_interference,
)

__all__ = [
    # Overshooters
    'OvershooterDetector',
    'OvershooterParams',
    'detect_overshooting_cells',
    # Undershooters
    'UndershooterDetector',
    'UndershooterParams',
    'detect_undershooting_cells',
    'detect_undershooting_with_environment_awareness',
    'compare_undershooting_detection_approaches',
    # PCI Conflict
    'PCIConflictDetector',
    'PCIConflictParams',
    'detect_pci_conflicts',
    # Crossed Feeder
    'CrossedFeederDetector',
    'CrossedFeederParams',
    'detect_crossed_feeders',
    # CA Imbalance
    'CAImbalanceDetector',
    'CAImbalanceParams',
    'CAPairConfig',
    'detect_ca_imbalance',
    # Interference
    'InterferenceDetector',
    'InterferenceParams',
    'detect_interference',
]

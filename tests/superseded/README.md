# Superseded Test Scripts

This folder contains old test scripts that have been replaced by newer, more comprehensive versions.

These files are kept for reference but are no longer actively used in the project.

## Superseded Scripts

### Old Overshooting Tests
- **test_cell_coverage_overshooters.py** - Replaced by `test_full_dataset_overshooters.py`
- **test_vf_overshooting.py** - Old VF-specific test, replaced by full dataset version
- **test_visualize_overshooters.py** - Replaced by `test_visualize_unified_full.py`

### Old Undershooting Tests
- **test_undershooting_detection.py** - Replaced by `test_undershooting_full_dataset.py`
- **test_undershooting_environment_aware.py** - Old environment-aware version

### Old Visualization
- **test_visualize_unified.py** - Replaced by `test_visualize_unified_full.py`
- **test_visualize_with_enhanced_metrics.py** - Old enhanced metrics visualization

### Old Combined Tests
- **test_environment_aware_detection.py** - Old combined environment-aware test
- **test_enhanced_metrics.py** - Experimental enhanced metrics

## Current Active Scripts

See `../integration/` for the current active integration tests:
- `test_full_dataset_overshooters.py` - Main overshooting detection
- `test_undershooting_full_dataset.py` - Main undershooting detection
- `test_visualize_unified_full.py` - Unified visualization for both

## Safe to Delete?

These files can be safely deleted once you're confident the new scripts meet all requirements.
They are kept temporarily for reference purposes only.

---
*Superseded on: 2024-11-24*

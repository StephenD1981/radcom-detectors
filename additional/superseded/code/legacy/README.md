# Legacy Scripts

This folder contains archived grid enrichment scripts that have been replaced by the `ran_optimizer` package.

## Archived Files

- `create-bin-cell-enrichment-vf-ie.py` - Vodafone Ireland grid enrichment
- `create-bin-cell-enrichment-dish.py` - DISH network grid enrichment
- `create-bin-cell-enrichment-vf-ie-pm.py` - VF-IE performance management variant

## Status

⚠️ **DO NOT USE IN PRODUCTION**

These scripts are kept for reference only. All functionality has been migrated to:
- `ran_optimizer.pipeline.enrichment` - Grid enrichment pipeline
- `ran_optimizer.data.loaders` - Data loading with validation

## Replacement

Instead of:
```python
# OLD: Direct script execution
python code/create-bin-cell-enrichment-dish.py
```

Use:
```python
# NEW: Package-based approach
from ran_optimizer.pipeline import enrich_grids
from ran_optimizer.utils.config import load_config

config = load_config("config/operators/dish_denver.yaml")
result = enrich_grids(config)
```

## Migration Date

Archived: January 20, 2025
Replaced by: `ran_optimizer` v0.1.0

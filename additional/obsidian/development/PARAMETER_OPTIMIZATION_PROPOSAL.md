# Parameter Optimization Proposal

## Overview
Two strategic improvements to make the overshooting detection algorithm more flexible and accurate:
1. **Environment-aware parameters** (Urban/Suburban/Rural classification)
2. **Configurable parameters via JSON** (Operator customization)
3. **Machine learning for automatic parameter tuning**

---

## 1. Environment-Aware Parameters (Urban/Suburban/Rural)

### Rationale
Network density fundamentally changes what constitutes "overshooting":
- **Urban**: Dense sites, cells should serve locally (~500m-2km)
- **Suburban**: Medium density, moderate range (~2-6km)
- **Rural**: Sparse coverage, longer range acceptable (~5-15km)

Using the same parameters across all environments leads to:
- **Urban**: False negatives (missing actual overshooters)
- **Rural**: False positives (flagging normal long-range coverage)

### Classification Method: Intersite Distance

```python
# Calculate nearest neighbor distance for each cell
intersite_distance = distance_to_nearest_cell_same_band

# Classify environment
if intersite_distance < 1.5km:
    environment = "URBAN"
elif intersite_distance < 4.0km:
    environment = "SUBURBAN"
else:
    environment = "RURAL"
```

### Proposed Parameter Sets

| Parameter | Urban | Suburban (Default) | Rural |
|-----------|-------|-------------------|-------|
| `min_cell_distance` | **2 km** | 4 km | **6 km** |
| `min_cell_count_in_grid` | **6** | 4 | **3** |
| `max_percentage_grid_events` | 0.25 | 0.25 | **0.30** |
| `edge_traffic_percent` | **0.10** | 0.15 | **0.20** |
| `min_relative_reach` | 0.70 | 0.70 | **0.75** |
| `min_overshooting_grids` | 30 | 30 | **20** |
| `percentage_overshooting_grids` | 0.10 | 0.10 | 0.10 |

**Rationale:**
- **Urban**: Stricter edge definition (90th percentile), more competition required (6 cells)
- **Suburban**: Baseline parameters work well
- **Rural**: Relaxed distance thresholds (6km), fewer competitors expected (3 cells), slightly more lenient reach (0.75)

### Implementation Impact
- More accurate urban overshooting detection
- Fewer false positives in rural areas
- Environment-specific severity scoring

---

## 2. Configurable Parameters (JSON Export/Import)

### Problem
- Different operators have different network characteristics
- Parameters tuned for one network may not work for another
- Hard-coded parameters require code changes to adjust

### Solution: JSON Configuration

#### File Structure: `config/overshooting_params.json`

```json
{
  "version": "2.0",
  "description": "Overshooting detection parameters - VF Ireland",
  "last_updated": "2025-11-21",
  "default": {
    "edge_traffic_percent": 0.15,
    "min_cell_distance": 4000,
    "min_cell_count_in_grid": 4,
    "max_percentage_grid_events": 0.25,
    "min_relative_reach": 0.70,
    "min_overshooting_grids": 30,
    "percentage_overshooting_grids": 0.10,
    "rsrp_offset": 0.80,
    "percent_max_distance": 0.70
  },
  "environment_specific": {
    "urban": {
      "min_cell_distance": 2000,
      "min_cell_count_in_grid": 6,
      "edge_traffic_percent": 0.10,
      "enable": true
    },
    "suburban": {
      "enable": true
    },
    "rural": {
      "min_cell_distance": 6000,
      "min_cell_count_in_grid": 3,
      "edge_traffic_percent": 0.20,
      "min_relative_reach": 0.75,
      "min_overshooting_grids": 20,
      "enable": true
    }
  },
  "environment_classification": {
    "method": "intersite_distance",
    "urban_threshold_km": 1.5,
    "suburban_threshold_km": 4.0,
    "enable": true
  },
  "severity_thresholds": {
    "action_threshold": 0.50,
    "critical": 0.80,
    "high": 0.60,
    "medium": 0.40,
    "low": 0.20
  }
}
```

#### Implementation

```python
# Load configuration
from ran_optimizer.utils.config import load_overshooting_config

config = load_overshooting_config("config/overshooting_params.json")

# Create params for specific environment
params = OvershooterParams.from_config(config, environment="urban")

# Or use default
params = OvershooterParams.from_config(config)
```

#### Benefits
- **Operator Customization**: Each operator can tune parameters
- **Version Control**: JSON can be tracked in git
- **A/B Testing**: Easy to compare parameter sets
- **Documentation**: Parameters are self-documenting
- **Rapid Iteration**: No code changes needed

---

## 3. Automatic Parameter Learning (ML Approach)

### Problem
Manually tuning 7+ parameters across 3 environments is time-consuming and subjective.

### Solution Options

#### Option A: Supervised Learning (Requires Ground Truth)

**Prerequisites:**
- Labeled dataset of "true overshooters" (e.g., from field engineers)
- Ideally 100+ labeled cells

**Approach:**
```python
# 1. Feature engineering
features = [
    'max_distance_m',
    'overshooting_grids_count',
    'percentage_overshooting',
    'avg_edge_rsrp',
    'cell_competition_index',
    'relative_reach_mean',
    'environment_type',  # urban/suburban/rural
]

# 2. Grid search or Bayesian optimization
from sklearn.model_selection import GridSearchCV

param_grid = {
    'min_cell_distance': [2000, 3000, 4000, 5000, 6000],
    'min_cell_count_in_grid': [3, 4, 5, 6],
    'min_relative_reach': [0.60, 0.65, 0.70, 0.75, 0.80],
    # ... etc
}

# 3. Optimize for F1-score or precision/recall balance
best_params = optimize_params(param_grid, labeled_data)
```

**Pros:**
- Most accurate if ground truth available
- Can learn operator-specific patterns

**Cons:**
- Requires labeled data (expensive to collect)
- Risk of overfitting to specific network snapshot

#### Option B: Unsupervised Learning (No Labels Required)

**Approach 1: Anomaly Detection**
```python
# Identify statistical outliers in serving distance, competition, etc.
# Cells in top 5% of "abnormality score" → likely overshooters

from sklearn.ensemble import IsolationForest

features = df[['max_distance_m', 'competition_score', 'rsrp_degradation']]
model = IsolationForest(contamination=0.05)
anomalies = model.fit_predict(features)
```

**Approach 2: Clustering-Based**
```python
# Cluster cells by serving characteristics
# Cells in "long-range, high-competition, poor-RSRP" cluster → overshooters

from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.3, min_samples=10)
clusters = model.fit_predict(normalized_features)

# Identify "overshooting cluster" by centroid characteristics
```

**Pros:**
- No labeled data required
- Can discover patterns we didn't anticipate

**Cons:**
- Less interpretable
- May not align with RF engineering expectations

#### Option C: Optimization-Based (KPI-Driven)

**Approach:**
```python
# Optimize parameters to maximize network KPIs after changes applied

def objective_function(params):
    """
    Simulate network performance with given parameters.
    Returns: weighted_score based on KPIs
    """
    overshooters = detect_overshooters(params)

    # Apply hypothetical tilt changes
    predicted_improvements = simulate_tilt_changes(overshooters)

    # Score based on:
    # - Reduced overshooting %
    # - Improved edge RSRP
    # - Maintained coverage
    # - Limited cell count (operational cost)

    score = (
        0.3 * rsrp_improvement +
        0.3 * overshooting_reduction +
        0.2 * coverage_maintained +
        0.2 * (1 - num_cells_flagged / total_cells)
    )

    return score

# Use Bayesian optimization or genetic algorithm
from scipy.optimize import differential_evolution

result = differential_evolution(
    objective_function,
    bounds=[(0.1, 0.2), (2000, 6000), ...]  # param bounds
)

best_params = result.x
```

**Pros:**
- Directly optimizes business goals
- Can incorporate operational constraints

**Cons:**
- Requires simulation/modeling capability
- Complex to implement

#### **Recommendation: Hybrid Approach**

1. **Start with expert rules** (current approach) ✅
2. **Add configurable JSON** (easy win) → **Implement first**
3. **Collect validation data** (field engineer feedback on flagged cells)
4. **Use supervised learning** to refine thresholds once data available
5. **Long-term**: KPI-driven optimization with A/B testing

---

## Implementation Plan

### Phase 1: JSON Configuration (Week 1)
- [ ] Create `config/overshooting_params.json` schema
- [ ] Add `load_config()` and `save_config()` utilities
- [ ] Update `OvershooterParams` with `from_config()` class method
- [ ] Add environment classification logic
- [ ] Test with current parameters to ensure no regression

### Phase 2: Environment-Aware Detection (Week 2)
- [ ] Calculate intersite distance for each cell (use site_id, not sector)
- [ ] Classify cells as Urban/Suburban/Rural
- [ ] Load environment-specific parameters from JSON
- [ ] Re-run detection with new parameters
- [ ] Compare results and document improvements

### Phase 3: Validation & Tuning (Week 3-4)
- [ ] Generate reports with environment breakdown
- [ ] Share with field engineers for validation
- [ ] Collect feedback on flagged cells (true/false positives)
- [ ] Adjust parameters based on feedback
- [ ] Document tuning rationale

### Phase 4: ML Preparation (Future)
- [ ] Build labeled dataset from engineer feedback
- [ ] Set up experiment tracking (MLflow or similar)
- [ ] Implement grid search for parameter optimization
- [ ] A/B test parameter sets on different regions

---

## Expected Benefits

### Environment-Aware Parameters
- **+20-30% accuracy** in urban areas
- **-40-50% false positives** in rural areas
- Better alignment with RF engineering expectations

### JSON Configuration
- **10x faster** parameter iteration (no code changes)
- **Operator-specific tuning** without forking code
- **Version control** for parameter evolution
- **Easier A/B testing** across regions/operators

### ML-Based Optimization (Future)
- **Continuous improvement** as more data collected
- **Data-driven decisions** vs. manual tuning
- **Objective optimization** of business KPIs

---

## Questions for Discussion

1. **Environment Classification**: Should we use site-level (tower) or sector-level intersite distance?
2. **Parameter Validation**: Do you have field engineer feedback on current results we can use for validation?
3. **Operator Variability**: Are you planning to deploy this across multiple operators? If so, JSON config is critical.
4. **ML Readiness**: Do you have historical data on cells that were successfully optimized via tilt changes?
5. **Deployment**: Should environment classification run automatically, or be a separate preprocessing step?

---

## Recommended Next Steps

1. ✅ **Immediate**: Implement JSON configuration (2-3 hours)
2. ✅ **This Week**: Add environment classification and test on VF Ireland data
3. 📊 **Next Week**: Generate comparison report (current vs. environment-aware)
4. 👥 **Validation**: Share results with field team for feedback
5. 🔬 **Future**: Begin collecting labeled data for ML optimization

---

*Generated: 2025-11-21*
*Status: Proposal - Pending Approval*

# RAN Optimization System

Radio Access Network optimization tool for automated antenna tilt recommendations.

## Quick Start

```bash
# Clone repository
git clone <repo-url>
cd ran-optimizer

# Setup virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Run tests
pytest tests/
```

## Documentation

See `./obsidian/` folder for comprehensive documentation:
- [README](./obsidian/README.md) - Documentation index
- [PROJECT_OVERVIEW](./obsidian/PROJECT_OVERVIEW.md) - System overview
- [PRODUCTION_READINESS_PLAN](./obsidian/PRODUCTION_READINESS_PLAN.md) - Implementation roadmap
- [PHASE_1_IMPLEMENTATION_PLAN](./obsidian/PHASE_1_IMPLEMENTATION_PLAN.md) - Current phase details

## Project Structure

```
ran-optimizer/
├── ran_optimizer/          # Main package (NEW)
│   ├── core/              # Core algorithms (geometry, RF models)
│   ├── data/              # Data loading and validation
│   ├── recommendations/   # Recommendation features
│   ├── pipeline/          # Data pipeline orchestration
│   └── utils/             # Utilities (config, logging)
├── code-opt-data-sources/ # Data source generation (KEEP)
├── legacy/                # Archived scripts (MOVED)
├── explore/               # Jupyter notebooks (KEEP)
├── data/                  # Data files (gitignored)
├── tests/                 # Test suite
├── config/                # Configuration files
└── obsidian/              # Documentation
```

## Development

- Python 3.11+
- See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines
- Run tests: `pytest tests/`
- Format code: `black ran_optimizer/`
- Type check: `mypy ran_optimizer/`

## Features

### Production-Ready
- ✅ Overshooting detection (85% precision)
- ✅ Crossed feeder detection (67% precision)

### In Development
- ⚠️ Undershooting detection (needs validation)
- ⚠️ Interference detection (needs optimization)

### Experimental
- 🔧 Low coverage detection
- 🔧 PCI optimization

## Current Status

**Phase 1: Foundation** (In Progress)
- Week 1: Version control & package structure ← **YOU ARE HERE**
- Week 2: Configuration management
- Week 3: Data foundation
- Week 4: Documentation

See [PHASE_1_IMPLEMENTATION_PLAN](./obsidian/PHASE_1_IMPLEMENTATION_PLAN.md) for details.

## License

Internal use only.

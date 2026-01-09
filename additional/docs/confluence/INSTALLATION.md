# Installation Guide

## Before You Start

### What You Need

| Requirement | Version | How to Check |
|-------------|---------|--------------|
| Python | 3.11 or higher | `python --version` |
| pip | Any recent version | `pip --version` |
| Git | Any version | `git --version` |

### Do I Have Python 3.11+?

{code:bash}
python --version
# Output should be: Python 3.11.x or higher
{code}

If you see Python 3.10 or earlier, you need to upgrade Python first.

---

## Step-by-Step Installation

### Step 1: Get the Code

{code:bash}
# Clone the repository
git clone <repository-url>

# Go into the project folder
cd ran-optimizer
{code}

### Step 2: Create a Virtual Environment

A virtual environment keeps this project's packages separate from your other Python projects.

{code:bash}
# Create the virtual environment
python -m venv venv

# Activate it
# On macOS or Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
{code}

**How do I know it worked?**

Your command prompt should now show `(venv)` at the beginning:
{code:none}
(venv) username@computer:~/ran-optimizer$
{code}

### Step 3: Install System Dependencies

Some packages need system libraries for geographic processing.

**On macOS:**
{code:bash}
brew install gdal geos proj
{code}

**On Ubuntu/Debian:**
{code:bash}
sudo apt-get update
sudo apt-get install -y libgdal-dev libgeos-dev libproj-dev
{code}

**On Windows:**
The easiest approach is to use Anaconda instead of pip. See "Alternative: Anaconda Installation" below.

### Step 4: Install the RAN Optimizer

{code:bash}
# Install in development mode (recommended)
pip install -e .
{code}

**What does `-e .` mean?**
- `-e` means "editable" - changes you make to the code take effect immediately
- `.` means "this folder" - install from the current directory

### Step 5: Verify Installation

{code:bash}
# Check if the command-line tool is available
ran-optimize --help
{code}

You should see output like:
{code:none}
Usage: ran-optimize [OPTIONS]

  RAN Optimizer - Analyze network data for optimization recommendations.

Options:
  --input-dir PATH     Input data directory
  --output-dir PATH    Output directory
  ...
{code}

**If you see this, installation was successful!**

---

## Alternative: Anaconda Installation

If you're using Anaconda (common on Windows or for data science work):

{code:bash}
# Create a conda environment
conda create -n ran-optimizer python=3.11

# Activate it
conda activate ran-optimizer

# Install geospatial dependencies (easier than pip on Windows)
conda install -c conda-forge geopandas shapely pyproj

# Install the RAN Optimizer
pip install -e .
{code}

---

## Alternative: Docker Installation

If you prefer containers (for isolation or production):

{code:bash}
# Build the image
docker build -t ran-optimizer .

# Run it
docker run -v /path/to/your/data:/app/data ran-optimizer \
  --input-dir /app/data/input \
  --output-dir /app/data/output
{code}

---

## What Gets Installed?

### Core Packages

| Package | What It Does |
|---------|--------------|
| pandas | Data manipulation and analysis |
| numpy | Numerical calculations |
| geopandas | Geographic data handling |
| shapely | Geometric operations (polygons, etc.) |
| pyproj | Coordinate system transformations |
| geohash2 | Location encoding/decoding |

### Data Processing

| Package | What It Does |
|---------|--------------|
| scipy | Scientific computing utilities |
| scikit-learn | Machine learning helpers |
| hdbscan | Clustering algorithm for coverage gaps |
| alphashape | Creates polygon boundaries |

### Visualization

| Package | What It Does |
|---------|--------------|
| folium | Interactive map generation |
| matplotlib | Static charts and plots |

### Configuration

| Package | What It Does |
|---------|--------------|
| pydantic | Data validation |
| pyyaml | YAML file reading |
| structlog | Logging |

---

## Testing Your Installation

### Quick Test

Run a quick sanity check:

{code:bash}
# Run unit tests
pytest tests/unit/ -v
{code}

Expected output:
{code:none}
tests/unit/test_config.py::test_load_config PASSED
tests/unit/test_overshooting.py::test_basic PASSED
...
X passed in Y.YYs
{code}

### Full Test with Sample Data

If you have sample data available:

{code:bash}
ran-optimize \
  --input-dir data/vf-ie/input-data \
  --output-dir /tmp/test-output \
  --algorithms overshooting

# Check the output was created
ls /tmp/test-output/
{code}

---

## Troubleshooting

### Problem: "command not found: ran-optimize"

**Cause:** The package wasn't installed correctly, or your virtual environment isn't activated.

**Solution:**
{code:bash}
# Make sure your virtual environment is active
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall
pip install -e .

# Try again
ran-optimize --help
{code}

### Problem: "ImportError: cannot import geopandas"

**Cause:** The geospatial system libraries aren't installed.

**Solution (macOS):**
{code:bash}
brew install gdal geos proj
pip uninstall geopandas shapely
pip install geopandas shapely
{code}

**Solution (Linux):**
{code:bash}
sudo apt-get install libgdal-dev libgeos-dev libproj-dev
pip uninstall geopandas shapely
pip install geopandas shapely
{code}

### Problem: "MemoryError" when running

**Cause:** Your dataset is too large for available RAM.

**Solution:**
1. Close other applications to free memory
2. Use smaller chunks:
   {code:bash}
   # In your config, reduce chunk_size
   processing:
     chunk_size: 50000  # Instead of 100000
   {code}
3. Run algorithms one at a time:
   {code:bash}
   ran-optimize --algorithms overshooting
   ran-optimize --algorithms undershooting
   {code}

### Problem: "Permission denied" when saving output

**Cause:** You don't have write access to the output directory.

**Solution:**
{code:bash}
# Create the output directory with correct permissions
mkdir -p data/output
chmod 755 data/output
{code}

### Problem: Installation takes forever / hangs

**Cause:** Compiling from source (common with GDAL-related packages).

**Solution:** Use pre-built wheels:
{code:bash}
pip install --only-binary :all: geopandas shapely
{code}

Or use Anaconda which provides pre-built binaries.

---

## Development Installation

If you want to contribute to the project or run tests:

{code:bash}
# Install with development dependencies
pip install -e ".[dev]"
{code}

This adds:
| Package | What It Does |
|---------|--------------|
| pytest | Test runner |
| pytest-cov | Code coverage |
| black | Code formatting |
| flake8 | Code linting |
| mypy | Type checking |

### Running Development Tools

{code:bash}
# Format code
black ran_optimizer/

# Check code style
flake8 ran_optimizer/

# Check types
mypy ran_optimizer/

# Run all tests with coverage
pytest --cov=ran_optimizer tests/
{code}

---

## Next Steps

Now that you've installed the RAN Optimizer:

1. **Prepare your data** - See [DATA_FORMATS] for what files you need
2. **Configure for your network** - See [CONFIGURATION] for parameter tuning
3. **Run your first analysis** - See [README] for quick start

---

## Getting Help

If you're still stuck:

1. Check the GitHub issues for similar problems
2. Open a new issue with:
   - Your operating system
   - Python version (`python --version`)
   - Full error message
   - Steps you tried


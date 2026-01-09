FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for geopandas
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ran_optimizer/ ran_optimizer/
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install -e .

# Default data directories (mount these as volumes)
VOLUME ["/app/data/input", "/app/data/output", "/app/config"]

ENTRYPOINT ["ran-optimize"]
CMD ["--input-dir", "/app/data/input", "--output-dir", "/app/data/output"]

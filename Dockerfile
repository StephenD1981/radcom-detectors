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
COPY config/ config/
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install -e .

# Default data directory (mount as volume)
# Structure: /app/data/<operator>/input-data and /app/data/<operator>/output-data
VOLUME ["/app/data", "/app/config"]

# Environment variables for configuration
ENV OPERATOR=vf-ie
ENV LOG_LEVEL=INFO
ENV ALGORITHMS=""

# Entrypoint script to handle operator selection
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]

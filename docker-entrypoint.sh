#!/bin/bash
set -e

# RAN Optimizer Docker Entrypoint
# Supports multi-operator deployments via environment variables

# Configuration from environment
OPERATOR="${OPERATOR:-vf-ie}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
ALGORITHMS="${ALGORITHMS:-}"
CONFIG_DIR="${CONFIG_DIR:-/app/config}"
DATA_DIR="${DATA_DIR:-/app/data}"

# Build the data directory path for this operator
OPERATOR_DATA_DIR="${DATA_DIR}/${OPERATOR}"

echo "========================================"
echo "RAN Optimizer - Docker Container"
echo "========================================"
echo "Operator:    ${OPERATOR}"
echo "Data Dir:    ${OPERATOR_DATA_DIR}"
echo "Config Dir:  ${CONFIG_DIR}"
echo "Log Level:   ${LOG_LEVEL}"
echo "Algorithms:  ${ALGORITHMS:-all}"
echo "========================================"

# Validate operator data directory exists
if [ ! -d "${OPERATOR_DATA_DIR}/input-data" ]; then
    echo "ERROR: Input directory not found: ${OPERATOR_DATA_DIR}/input-data"
    echo "Please mount your data volume with the correct structure:"
    echo "  /app/data/${OPERATOR}/input-data/"
    echo "  /app/data/${OPERATOR}/output-data/"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "${OPERATOR_DATA_DIR}/output-data"
mkdir -p "${OPERATOR_DATA_DIR}/output-data/maps"
mkdir -p "${OPERATOR_DATA_DIR}/output-data/pg_tables"

# Build command arguments
CMD_ARGS="--data-dir ${OPERATOR_DATA_DIR}"
CMD_ARGS="${CMD_ARGS} --config-dir ${CONFIG_DIR}"

# Add specific algorithms if specified
if [ -n "${ALGORITHMS}" ]; then
    CMD_ARGS="${CMD_ARGS} --algorithms ${ALGORITHMS}"
fi

# Set log level
export LOG_LEVEL

# Run the optimizer
echo "Running: ran-optimize ${CMD_ARGS}"
echo "========================================"

exec ran-optimize ${CMD_ARGS}

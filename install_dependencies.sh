#!/bin/bash

# Exit on error
set -e

echo "Starting installation of all dependencies..."

# 1. Install uv
echo "Installing uv..."
pip install uv

# 2. Install project dependencies
echo "Installing project dependencies from pyproject.toml..."
cd /workspace/sae-monosemantic && uv pip install --system -e .

# 3. Install rsync
echo "Installing rsync..."
apt-get update && apt-get install -y rsync

# 4. Install git-lfs
echo "Installing git-lfs..."
apt-get install -y git-lfs

# 5. Initialize git-lfs
echo "Initializing git-lfs..."
git lfs install

echo "All installations completed successfully!"

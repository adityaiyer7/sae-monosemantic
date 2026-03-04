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

# 4. Install Homebrew
echo "Installing Homebrew..."
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 5. Configure Homebrew in bashrc
echo "Configuring Homebrew in bashrc..."
echo >> /root/.bashrc
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv bash)"' >> /root/.bashrc

# 6. Load Homebrew in current session
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv bash)"

# 7. Verify brew installation
echo "Verifying Homebrew installation..."
brew --version

# 8. Install git-lfs
echo "Installing git-lfs..."
brew install git-lfs

# 9. Initialize git-lfs
echo "Initializing git-lfs..."
git lfs install

echo "All installations completed successfully!"

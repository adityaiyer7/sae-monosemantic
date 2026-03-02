#!/bin/bash
set -euo pipefail

REPO_DIR="/workspace/sae-monosemantic"
DATA_DIR="${REPO_DIR}/data"
JUPYTER_PORT=8888

# ── Step 1: Install system packages ───────────────────────────────────
echo "==> Installing system packages..."
apt-get update -qq && apt-get install -y -qq rsync

# ── Step 2: Clone repo if not already present ─────────────────────────
echo ""
echo "==> Checking for repo..."
if [ -d "$REPO_DIR" ]; then
    echo "    Repo already present at ${REPO_DIR}. Skipping clone."
else
    echo "    Cloning repo into ${REPO_DIR}..."
    git clone git@github.com:adityaiyer7/sae-monosemantic.git "$REPO_DIR"
fi

# ── Step 3: Install Python dependencies ───────────────────────────────
echo ""
echo "==> Installing uv and project dependencies..."
pip install -q uv
cd "$REPO_DIR" && uv pip install --system -e .

# ── Step 4: Check for data, prompt for transfer if missing ────────────
echo ""
echo "==> Checking for data..."
if [ -d "$DATA_DIR" ] && [ "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    FILE_COUNT=$(find "$DATA_DIR" -type f | wc -l | tr -d '[:space:]')
    echo "    Data already present (${FILE_COUNT} files). Skipping transfer."
else
    echo "    Data not found at ${DATA_DIR}."
    read -rp "    Pod IP (from RunPod dashboard): " POD_IP
    read -rp "    SSH Port: " SSH_PORT
    echo ""
    echo "    Run this on your LOCAL machine to transfer data:"
    echo ""
    echo "    rsync -avz --progress -e 'ssh -p ${SSH_PORT} -i ~/.ssh/id_ed25519' \\"
    echo "      ~/Desktop/Projects/sae-monosemantic/data/ \\"
    echo "      root@${POD_IP}:${DATA_DIR}/"
    echo ""
    read -rp "    Press Enter once the transfer is done..."
fi

# ── Step 5: Kill existing Jupyter servers ─────────────────────────────
echo ""
echo "==> Killing existing Jupyter servers..."
pkill -f jupyter 2>/dev/null || true
sleep 2

# ── Step 6: Start JupyterLab ─────────────────────────────────────────
echo ""
echo "==> Starting JupyterLab on port ${JUPYTER_PORT}..."
cd "$REPO_DIR" && jupyter lab \
    --ip=0.0.0.0 \
    --port=${JUPYTER_PORT} \
    --no-browser \
    --allow-root \
    --ServerApp.allow_origin='*' \
    --ServerApp.allow_remote_access=True \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.token='' \
    --ServerApp.password=''
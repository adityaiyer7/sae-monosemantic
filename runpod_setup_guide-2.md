# RunPod Setup Guide for SAE Training

A complete end-to-end guide for spinning up a RunPod pod, setting up the environment, and getting training running.

---

## 1. SSH into the Pod

Get your connection details from RunPod dashboard → your pod → **Connect** → **Direct TCP ports**. It will show something like:

```
154.54.102.36:12037 → :22
```

Then SSH in:

```bash
ssh root@<POD_IP> -p <PORT> -i ~/.ssh/id_ed25519
```

---

## 2. Add Pod SSH Key to GitHub

> ✅ If you've already copied your SSH key to `/workspace/.ssh/` (one-time setup), skip this section — the setup script handles it automatically.

The pod needs its own SSH key to clone private repos from GitHub.

**Generate a key on the pod:**

```bash
ssh-keygen -t ed25519 -C "adityaiyer7"
```

**Print the public key:**

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the entire output, then go to:

**GitHub → Settings → SSH and GPG Keys → New SSH Key**

Paste it in and save.

**Test it works:**

```bash
ssh -T git@github.com
```

You should see: `Hi <username>! You've successfully authenticated.`

**Persist the key to `/workspace` so you never need to do this again:**

```bash
mkdir -p /workspace/.ssh
cp ~/.ssh/id_ed25519 /workspace/.ssh/
cp ~/.ssh/id_ed25519.pub /workspace/.ssh/
```

---

## 3. Clone Your Repo

> On subsequent pods, skip this if the repo is already in `/workspace` from a previous session.

```bash
git clone git@github.com:adityaiyer7/sae-monosemantic.git /workspace/sae-monosemantic
```

---

## 4. Install Dependencies

First install uv if not present:

```bash
pip install uv
```

Then install all project dependencies from `pyproject.toml`:

```bash
cd /workspace/sae-monosemantic && uv pip install --system -e .
```

You only need to do this **once per container instance**.

---

## 5. Install rsync

rsync is not pre-installed on RunPod. Run:

```bash
apt-get update && apt-get install -y rsync
```

---

## 6. Transfer Data from Local Machine

> ✅ If data is already in `/workspace/sae-monosemantic/data/`, skip this — it persists across sessions.

Run this on your **local terminal** (not on the pod):

```bash
rsync -avz --progress -e 'ssh -p <PORT> -i ~/.ssh/id_ed25519' \
  ~/Desktop/Projects/sae-monosemantic/data/ \
  root@<POD_IP>:/workspace/sae-monosemantic/data/
```

- Use the direct IP, not `ssh.runpod.io` — the hostname can time out
- 22GB takes ~10-30 min at typical speeds
- If the transfer is interrupted, just rerun the same command — rsync will skip already-transferred files
- You only need to do this **once** — data persists in `/workspace` across pod sessions

---

## 7. Start JupyterLab

```bash
cd /workspace/sae-monosemantic && jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root \
  --ServerApp.allow_origin='*' \
  --ServerApp.allow_remote_access=True \
  --ServerApp.disable_check_xsrf=True \
  --ServerApp.token='' \
  --ServerApp.password=''
```

Access via: `https://<pod-id>-8888.proxy.runpod.net`

If port 8888 is already in use from a previous instance:

```bash
pkill -f jupyter
```

Then restart the command above.

---

## 8. Push Changes Back to GitHub

From the pod terminal:

```bash
cd /workspace/sae-monosemantic
git add .
git commit -m "your message"
git push
```

---

## 9. Running Python Scripts

To run a script directly on the pod:

```bash
python /workspace/sae-monosemantic/your_script.py
```

Or navigate first:

```bash
cd /workspace/sae-monosemantic
python your_script.py
```

**Path tip:** Always use `Path(__file__).resolve().parents[N]` to find the project root in scripts rather than `Path.cwd()`, since the root depends on where you call the script from. For a script at `src/evaluation/script.py`, use `parents[2]` to get the repo root.

---

## 10. Notebook Cell Tips

In JupyterLab notebook cells, prefix shell commands with `!`:

```python
!apt-get update && apt-get install -y rsync
```

For commands that need a specific working directory, use subprocess:

```python
import subprocess
result = subprocess.run(
    ["uv", "pip", "install", "--system", "-e", "."],
    capture_output=True, text=True, cwd="/workspace/sae-monosemantic"
)
print(result.stdout)
print(result.stderr)
```

Note: `cd` in notebook cells doesn't persist between cells — use `cwd` in subprocess or `%cd` magic instead.

---

## 11. Gotchas

- **SSH keys persist if copied to `/workspace/.ssh/`.** The setup script restores them automatically. If you haven't done this yet, see Step 2.
- **`/workspace` is persistent.** Your repo, data, model weights, and SSH keys in `/workspace` survive pod stops and restarts. Only the container disk (system packages, etc.) resets.
- **JupyterLab terminals can be buggy** on RunPod — blank terminal tabs are common. Use notebook cells with `!` prefix instead.
- **Don't paste commands with smart quotes** on Mac — they can cause `dquote>` hanging prompts. If this happens, open a new terminal window.
- **"Notebook is not trusted"** warning is harmless.
- **`Could not determine jupyterlab build status without nodejs`** is harmless, ignore it.
- **`uv` needs `--system` flag** when not in a virtual environment: `uv pip install --system -e .`
- **`chown` errors during rsync** are harmless — rsync tries to preserve Mac file ownership but the pod won't allow it. Files transfer correctly regardless.
- **Pod IP/port can change** when you edit pod settings (e.g. resizing volume disk). Always check the RunPod dashboard for the current connection details.

---

## 12. One-Time Startup Script (Recommended)

Save this as `setup.sh` in your repo. On any new pod, just run it — it handles SSH key restoration, repo clone, dependencies, data check, and JupyterLab.

```bash
#!/bin/bash
set -euo pipefail

REPO_DIR="/workspace/sae-monosemantic"
DATA_DIR="${REPO_DIR}/data"
JUPYTER_PORT=8888

# ── Step 1: Install system packages ───────────────────────────────────
echo "==> Installing system packages..."
apt-get update -qq && apt-get install -y -qq rsync

# ── Step 2: Restore GitHub SSH key from /workspace ────────────────────
echo ""
echo "==> Setting up GitHub SSH key..."
if [ -f "/workspace/.ssh/id_ed25519" ]; then
    mkdir -p ~/.ssh
    ln -sf /workspace/.ssh/id_ed25519 ~/.ssh/id_ed25519
    ln -sf /workspace/.ssh/id_ed25519.pub ~/.ssh/id_ed25519.pub
    chmod 600 ~/.ssh/id_ed25519
    echo "    SSH key restored from /workspace."
else
    echo "    No SSH key found in /workspace."
    echo "    Generate one with: ssh-keygen -t ed25519 -C 'adityaiyer7'"
    echo "    Then add the public key to GitHub and run:"
    echo "    mkdir -p /workspace/.ssh && cp ~/.ssh/id_ed25519* /workspace/.ssh/"
    read -rp "    Press Enter once done..."
fi

# ── Step 3: Clone repo if not already present ─────────────────────────
echo ""
echo "==> Checking for repo..."
if [ -d "$REPO_DIR" ]; then
    echo "    Repo already present at ${REPO_DIR}. Skipping clone."
else
    echo "    Cloning repo into ${REPO_DIR}..."
    git clone git@github.com:adityaiyer7/sae-monosemantic.git "$REPO_DIR"
fi

# ── Step 4: Install Python dependencies ───────────────────────────────
echo ""
echo "==> Installing uv and project dependencies..."
pip install -q uv
cd "$REPO_DIR" && uv pip install --system -e .

# ── Step 5: Check for data, prompt for transfer if missing ────────────
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

# ── Step 6: Kill existing Jupyter servers ─────────────────────────────
echo ""
echo "==> Killing existing Jupyter servers..."
pkill -f jupyter 2>/dev/null || true
sleep 2

# ── Step 7: Start JupyterLab ─────────────────────────────────────────
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
```

On any new pod, just run:

```bash
bash /workspace/sae-monosemantic/setup.sh
```
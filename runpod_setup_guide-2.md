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

The pod needs its own SSH key to clone private repos from GitHub.

**Generate a key on the pod** (skip if already exists):

```bash
ssh-keygen -t ed25519 -C "your_github_username"
```

**Print the public key:**

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the entire output (including the username at the end — that's normal), then go to:

**GitHub → Settings → SSH and GPG Keys → New SSH Key**

Paste it in and save.

**Test it works:**

```bash
ssh -T git@github.com
```

You should see: `Hi <username>! You've successfully authenticated.`

---

## 3. Clone Your Repo

```bash
cd / && git clone git@github.com:adityaiyer7/sae-monosemantic.git
```

---

## 4. Install Dependencies

First install uv if not present:

```bash
pip install uv
```

Then install all project dependencies from `pyproject.toml`:

```bash
cd /sae-monosemantic && uv pip install --system -e .
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

Run this on your **local terminal** (not on the pod):

```bash
rsync -avz --progress -e 'ssh -p <PORT> -i ~/.ssh/id_ed25519' \
  /Users/adityaiyer/Desktop/Projects/sae-monosemantic/data/ \
  root@<POD_IP>:/sae-monosemantic/data/
```

- Use the direct IP, not `ssh.runpod.io` — the hostname can time out
- 22GB takes ~10-30 min at typical speeds
- Order of file transfer doesn't matter

---

## 7. Start JupyterLab

```bash
cd /sae-monosemantic && jupyter lab \
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
cd /sae-monosemantic
git add .
git commit -m "your message"
git push
```

---

## 9. Notebook Cell Tips

In JupyterLab notebook cells, prefix shell commands with `!`:

```python
!apt-get update && apt-get install -y rsync
```

For commands that need a specific working directory, use subprocess:

```python
import subprocess
result = subprocess.run(
    ["uv", "pip", "install", "--system", "-e", "."],
    capture_output=True, text=True, cwd="/sae-monosemantic"
)
print(result.stdout)
print(result.stderr)
```

Note: `cd` in notebook cells doesn't persist between cells — use `cwd` in subprocess or `%cd` magic instead.

---

## 10. Gotchas

- **Pods are not persistent.** Every new pod starts fresh — you'll need to redo steps 2-7 each time. Data also disappears unless you use a Network Volume (see below).
- **JupyterLab terminals can be buggy** on RunPod — blank terminal tabs are common. Use notebook cells with `!` prefix instead.
- **Don't paste commands with smart quotes** on Mac — they can cause `dquote>` hanging prompts. If this happens, open a new terminal window.
- **"Notebook is not trusted"** warning is harmless.
- **`Could not determine jupyterlab build status without nodejs`** is harmless, ignore it.
- **`uv` needs `--system` flag** when not in a virtual environment: `uv pip install --system -e .`

---

## 11. One-Time Startup Script (Recommended)

To avoid repeating steps 4-7 every time, save this as `start.sh` in your repo:

```bash
#!/bin/bash
set -e

echo "==> Installing system packages..."
apt-get update && apt-get install -y rsync

echo "==> Installing Python dependencies..."
pip install uv
uv pip install --system -e .

echo "==> Starting JupyterLab..."
cd /sae-monosemantic && jupyter lab \
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

On any new pod, after cloning your repo just run:

```bash
bash /sae-monosemantic/start.sh
```

---

## 12. Making Data Persistent (Recommended)

To avoid re-uploading 22GB every time:

1. Go to RunPod dashboard → **Storage → New Network Volume**
2. Create a volume and attach it to your pod at `/sae-monosemantic/data`
3. Transfer your data once — it persists across pod restarts and terminations
4. Future pods just need to attach the same volume

This is the single biggest quality of life improvement for repeated pod usage.

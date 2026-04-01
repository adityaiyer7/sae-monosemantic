# HuggingFace Space Deployment Guide

## Prerequisites

`huggingface_hub` is already installed as a project dependency — all commands below use `uv run` to invoke it from the project venv.

---

## 1. Login to HuggingFace

```bash
uv run python -c "from huggingface_hub import login; login()"
```

Or pass your token directly (reads from `.env`):

```bash
uv run python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from huggingface_hub import login
login(token=os.environ['HF_TOKEN'])
"
```

---

## 2. Create the Space

```bash
uv run python -c "
from huggingface_hub import HfApi
HfApi().create_repo('sae-feature-explorer', repo_type='space', space_sdk='docker')
"
```

Or with token from `.env`:

```bash
uv run python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from huggingface_hub import HfApi
HfApi().create_repo('sae-feature-explorer', repo_type='space', space_sdk='docker', token=os.environ['HF_TOKEN'])
"
```

---

## 3. Add the Space as a git remote

```bash
git remote add space https://huggingface.co/spaces/thedarkknight7/sae-feature-explorer
```

---

## 4. Push to the Space

HF rejects files >10 MiB — use an orphan branch to push a clean snapshot with no git history (parquets, model weights, and notebooks are excluded via `.gitignore`):

```bash
git checkout --orphan hf-space-clean
git rm -rf --cached .
git add .
git rm -rf --cached notebooks/
git commit -m "Initial HF Space deployment"
git push space hf-space-clean:main --force
git checkout -f HF-space-setup
git branch -D hf-space-clean
```

### Future deployments

Repeat the orphan branch steps above (without `--force`):

```bash
git checkout --orphan hf-space-clean
git rm -rf --cached .
git add .
git rm -rf --cached notebooks/
git commit -m "Update HF Space"
git push space hf-space-clean:main
git checkout -f HF-space-setup
git branch -D hf-space-clean
```

---

## 5. Set Space secrets

```bash
uv run python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from huggingface_hub import HfApi
api = HfApi()
api.add_space_secret('thedarkknight7/sae-feature-explorer', 'HF_TOKEN', os.environ['HF_TOKEN'])
api.add_space_secret('thedarkknight7/sae-feature-explorer', 'GROQ_API_KEY', os.environ['GROQ_API_KEY'])
"
```

| Secret | Required | Purpose |
|--------|----------|---------|
| `HF_TOKEN` | **Yes** | Read feature datasets from HuggingFace |
| `GROQ_API_KEY` | No | Enable LLM feature labelling via Groq |

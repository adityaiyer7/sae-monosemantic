# HF Space Deployment

All commands use `uv run` — no separate installs needed.

## First-time setup

**1. Create the Space**
```bash
uv run python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from huggingface_hub import HfApi
HfApi().create_repo('sae-feature-explorer', repo_type='space', space_sdk='docker', token=os.environ['HF_TOKEN'])
"
```

**2. Add git remote**
```bash
git remote add space https://huggingface.co/spaces/thedarkknight7/sae-feature-explorer
```

**3. Set secrets**
```bash
uv run python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from huggingface_hub import HfApi
api = HfApi()
api.add_space_secret('thedarkknight7/sae-feature-explorer', 'HF_TOKEN', os.environ['HF_TOKEN'])
api.add_space_secret('thedarkknight7/sae-feature-explorer', 'GROQ_API_KEY', os.environ['GROQ_API_KEY'])
"
```

**4. Push README (Space config)**
```bash
uv run python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from huggingface_hub import HfApi
content = '''---
title: SAE Feature Explorer
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---
'''
HfApi().upload_file(path_or_fileobj=content.encode(), path_in_repo='README.md',
    repo_id='thedarkknight7/sae-feature-explorer', repo_type='space', token=os.environ['HF_TOKEN'])
"
```

**5. Push code** (orphan branch avoids pushing large files stuck in git history)
```bash
git checkout --orphan hf-space-clean
git rm -rf --cached .
git add .
git rm -rf --cached notebooks/
git commit -m "Deploy to HF Space"
git push space hf-space-clean:main --force
git checkout -f HF-space-setup
git branch -D hf-space-clean
```

---

## Subsequent deployments

Repeat step 5 without `--force`.

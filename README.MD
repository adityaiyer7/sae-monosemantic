# SAE Monosemantic (WIP)

## Reproducability

1) Install uv with Homebrew:
```
brew install uv
```

2) (Recommended, optional) Create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate
```

3) Install dependencies with uv:
```
uv sync
```

4) Download the dataset:
```
python scripts/data_downloader.py
```
Note: `num_samples` in `scripts/data_downloader.py` is changeable if you want a bigger/smaller subset.

5) Form SAE training data (activations):
```
python scripts/extract_activations.py
```
This script builds the residual-stream activation dataset used to train the SAE.

## AI usage

I believe AI can help boost productivity, but the goal of this project is learning and to that end, I believe it's important to monitor the usage of AI. In this project, AI has been used for the following (not autonomously):
1. Generating docstrings and documentation (including this readme.)
2. Writing Regex Patterns.
3. Visualizations (loss curves, etc.)
4. Logging, creating output files, and logging metrics, metadata, artifacts to wandb.
5. Creating the runpod_setup.sh script.
6. Time Tracking. 
7. Refactoring code.
8. Brainstorming data pipeline architecture and debugging SQL/API integrations.
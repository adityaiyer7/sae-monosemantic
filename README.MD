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

I believe AI can help boost productivity, but the goal of this project is learning and to that end, AI has only been used to generate docstrings and documentation.

# Sparse Autoencoders for Monosemanticity

## SAE Feature Explorer (Docker)

The easiest way to explore the trained features is via the Streamlit app, which runs entirely in Docker — no GPU or local model weights needed.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A [HuggingFace](https://huggingface.co/settings/tokens) access token (read access is sufficient)

### Setup

**1. Clone the repo**
```bash
git clone https://github.com/thedarkknight7/sae-monosemantic.git
cd sae-monosemantic
```

**2. Configure environment variables**
```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | HuggingFace read token — used to fetch feature datasets |
| `GROQ_API_KEY` | Optional (Recommended)| Enables LLM feature labelling via Groq (`openai/gpt-oss-120b`) |
| `OPENAI_API_KEY` | Optional (Recommended) | Enables LLM feature labelling via OpenAI (`gpt-5.4`) |
| `WANDB_API_KEY` | Optional | Only needed if running training notebooks |

**3. Build and run**
```bash
docker compose up --build
```

The first build takes 5–10 minutes (downloading dependencies and caching the GPT-2 tokenizer). Subsequent starts are much faster.

**4. Open the app**

Navigate to [http://localhost:8501](http://localhost:8501).

### Subsequent runs

```bash
docker compose up        # start without rebuilding
docker compose down      # stop
docker compose down -v   # stop and clear the HuggingFace cache volume
```

### LLM Feature Analysis (optional)

If `GROQ_API_KEY` or `OPENAI_API_KEY` is set in `.env`, a provider selector appears in the sidebar. Selecting a provider adds an LLM-generated label and reasoning to each feature's detail view. Without either key the rest of the app works normally.

---

## Reproducability

1) Install uv with Homebrew:
```
brew install uv
```

2) Install dependencies:
```
uv sync
```

3) Download the dataset:
```
python scripts/data_downloader.py
```
Note: `num_samples` in `scripts/data_downloader.py` is changeable if you want a bigger/smaller subset.

4) Form SAE training data (activations):
```
python scripts/extract_activations.py
```
This script builds the residual-stream activation dataset used to train the SAE.

## Model Weights

All weights are stored in a common HF bucket [here](https://huggingface.co/buckets/thedarkknight7/sae-for-monosemanticity-model-weights). The naming convention is

```
model_weights_{expansion_factor}x_{lambda}_{sampling_strategy}.pth
```

**No Sampling**

| Lambda/Expansion Factor | 4x                                                                                                         | 8x                                                                                                                  | 16x                                                                                                                 | 32x                                                                                                                 |
| ----------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 1e-2                    | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/g9f0jap1?nw=nwuseradityaiyerm) | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/npvky0df/overview?nw=nwuseradityaiyerm) | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/2hm8ixm7/overview?nw=nwuseradityaiyerm) | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/ywuwb77c/overview?nw=nwuseradityaiyerm) |
| 1e-4                    | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/gxkkkdrw?nw=nwuseradityaiyerm) | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/l6h4mlaq?nw=nwuseradityaiyerm)          | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/fxuib6q0?nw=nwuseradityaiyerm)          | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/tdfx2nti?nw=nwuseradityaiyerm)          |


**Sampling**

| Lambda/Expansion Factor | 4x                                                                                                         | 8x                                                                                                         | 16x                                                                                                        | 32x                                                                                                        |
| ----------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 1e-2                    | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/0unh5odz?nw=nwuseradityaiyerm) | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/aa9zwpso?nw=nwuseradityaiyerm) | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/zqtca1j0?nw=nwuseradityaiyerm) | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/odm2giys?nw=nwuseradityaiyerm) |
| 1e-4                    | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/3qy1nj9o?nw=nwuseradityaiyerm) | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/bhzlyrvn?nw=nwuseradityaiyerm) | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/9xfpzaw6?nw=nwuseradityaiyerm) | [WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/xxmutsoo?nw=nwuseradityaiyerm) |


## Feature Extraction (No Sampling)

| Lambda/Expansion Factor | 4x                                                                                                                                                                                                             | 8x                                                                                                                                                                                                             | 16x                                                                                                                                                                                                             | 32x                                                                                                                                                                                                             |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1e-2                    | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_4x_0.01)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/53bdbmmc?nw=nwuseradityaiyerm)   | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_8x_0.01)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/ajp0vfb7?nw=nwuseradityaiyerm)   | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_16x_0.01)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/utouczvf?nw=nwuseradityaiyerm)   | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_32x_0.01)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/v0kvrur0?nw=nwuseradityaiyerm)   |
| 1e-4                    | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_4x_0.0001)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/adajyv7i?nw=nwuseradityaiyerm) | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_8x_0.0001)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/zs7gdue3?nw=nwuseradityaiyerm) | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_16x_0.0001)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/qhylxv8a?nw=nwuseradityaiyerm) | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_32x_0.0001)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/rv5w2w8s?nw=nwuseradityaiyerm) |


## Feature Extraction (Sampling)

| Lambda/Expansion Factor | 4x                                                                                                                                                                                                                      | 8x                                                                                                                                                                                                                      | 16x                                                                                                                                                                                                                      | 32x                                                                                                                                                                                                                      |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1e-2                    | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_4x_0.01_sampling)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/0unh5odz?nw=nwuseradityaiyerm)   | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_8x_0.01_sampling)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/8m2jdy4j?nw=nwuseradityaiyerm)   | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_16x_0.01_sampling)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/yi34we3f?nw=nwuseradityaiyerm)   | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_32x_0.01_sampling)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/ruvlwru7?nw=nwuseradityaiyerm)   |
| 1e-4                    | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_4x_0.0001_sampling)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/vg9axgyt?nw=nwuseradityaiyerm) | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_8x_0.0001_sampling)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/kxbluv91?nw=nwuseradityaiyerm) | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_16x_0.0001_sampling)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/tylzekkv?nw=nwuseradityaiyerm) | [HF Repo](https://huggingface.co/datasets/thedarkknight7/SAE_monosemanticity_features_32x_0.0001_sampling)<br>[WandB Run](https://wandb.ai/adityaiyer-m-self/sae-for-monosemanticity/runs/aez10421?nw=nwuseradityaiyerm) |


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
9. Feature Visualization notebook. 
10. Dockerizing the SAE Feature Explorer and building UI. 
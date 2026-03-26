# %% [markdown]
# ## 0. Setup

# %%
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path (works for both local and Colab)
project_root = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

print(f"WANDB_API_KEY: {str(os.getenv('WANDB_API_KEY'))[:4]}...")
print(f"HF_TOKEN: {str(os.getenv('HF_TOKEN'))[:4]}...")

# %%
import torch
from itertools import islice
from torch.utils.data import Dataset, DataLoader
from src.models.sparse_autoencoder import SparseAutoEncoder
from src.training.dataset_creator import ChunkIterableGenerator, split_files
from src.training.losses import compute_loss
from src.training.trainer import SAETrainer
from src.training.utils import set_seed
from typing import Optional, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb
import subprocess

# %%
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA build:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# %%
# Sweep configuration
SWEEP_CONFIGS = [
    {"expansion_factor": ef, "_lambda": lam}
    for ef in [8, 16, 32]
    for lam in [1e-2, 1e-4]
]


# %%
def compute_loss_stats(losses, split_name):
    arr = np.asarray(losses, dtype=float)
    if arr.size == 0:
        return {
            "split": split_name,
            "num_batches": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
        }

    return {
        "split": split_name,
        "num_batches": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


# %%
def run_training(expansion_factor: int, _lambda: float):
    print(f"\n{'='*60}")
    print(f"Starting run: expansion_factor={expansion_factor}, _lambda={_lambda}")
    print(f"{'='*60}\n")

    hyperparameters = {
        "seed": 42,
        "num_epochs": 20,
        "expansion_factor": expansion_factor,
        "batch_size": 32,
        "num_workers": 0,
        "lr": 1e-4,
        "log_frequency": 5000,
        "_lambda": _lambda,
    }

    set_seed(hyperparameters["seed"])

    # W&B
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    run = wandb.init(
        entity="adityaiyer-m-self",
        project="sae-for-monosemanticity",
        config=hyperparameters,
        name=f"sae_training_run_{expansion_factor}x_{_lambda}_sampling",
    )

    MODEL_SAVE_PATH = project_root / "model_weights" / f'model_weights_{expansion_factor}x_{_lambda}_sampling.pth'
    print(f"MODEL_SAVE_PATH = {MODEL_SAVE_PATH}")

    # 1. Load data
    activation_chunk_dir = str(project_root / 'data' / 'gpt2_activation_chunks')
    train_files, val_files, test_files = split_files(activation_chunk_dir, seed=hyperparameters["seed"])

    print(f"Number of Training Files: {len(train_files)}")
    print(f"Number of Validation Files: {len(val_files)}")
    print(f"Number of Testing Files: {len(test_files)}")

    train_chunk_generator = ChunkIterableGenerator(train_files)
    val_chunk_generator = ChunkIterableGenerator(val_files)
    test_chunk_generator = ChunkIterableGenerator(test_files)

    training_dataloader = torch.utils.data.DataLoader(train_chunk_generator, batch_size=hyperparameters["batch_size"], num_workers=hyperparameters["num_workers"])
    val_dataloader = torch.utils.data.DataLoader(val_chunk_generator, batch_size=hyperparameters["batch_size"], num_workers=hyperparameters["num_workers"])
    test_dataloader = torch.utils.data.DataLoader(test_chunk_generator, batch_size=hyperparameters["batch_size"], num_workers=hyperparameters["num_workers"])

    # 2. Define Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SparseAutoEncoder(d_model=768, expansion_factor=expansion_factor)
    model = model.to(device)

    # 3. Training
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"])
    loss_fn = compute_loss

    trainer = SAETrainer(
        model=model,
        dataloader=training_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        _lambda=_lambda,
    )

    trainer.train(
        num_epochs=hyperparameters["num_epochs"],
        log_every=hyperparameters["log_frequency"],
    )

    run.log({
        "training_loss_curve": wandb.plot.line_series(
            xs=list(range(len(trainer.training_losses))),
            ys=[trainer.training_losses],
            keys=["Training Loss"],
            title="Training Loss",
            xname="Step"
        )
    })

    # intermediate save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f" model weights saved to {MODEL_SAVE_PATH}")

    # 5. Validation
    validation_loss, validation_num_batches = trainer.evaluate(val_dataloader)

    run.log({
        "validation_loss_curve": wandb.plot.line_series(
            xs=list(range(len(validation_loss))),
            ys=[validation_loss],
            keys=["Validation Loss"],
            title="Validation Loss",
            xname="Step"
        )
    })

    # 7. Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f" model weights saved to {MODEL_SAVE_PATH}")

    HF_BUCKET = "hf://buckets/thedarkknight7/sae-for-monosemanticity-model-weights"
    weights_dir = project_root / "model_weights"

    result = subprocess.run(
        ["hf", "sync", str(weights_dir), HF_BUCKET],
        capture_output=True,
        text=True,
        env={**os.environ, "HF_TOKEN": os.getenv("HF_TOKEN")},
    )

    if result.returncode == 0:
        print(f"Synced {weights_dir} to {HF_BUCKET}")
    else:
        print(f"Upload failed:\n{result.stderr}")

    # 8. Test Model
    test_loss, test_num_batches = trainer.evaluate(test_dataloader)

    run.log({
        "test_loss_curve": wandb.plot.line_series(
            xs=list(range(len(test_loss))),
            ys=[test_loss],
            keys=["Test Loss"],
            title="Test Loss",
            xname="Step"
        )
    })

    # 10. Compute Loss Statistics
    stats_df = (
        pd.DataFrame([
            compute_loss_stats(trainer.training_losses, "train"),
            compute_loss_stats(validation_loss, "val"),
            compute_loss_stats(test_loss, "test"),
        ])
        .set_index("split")
    )

    stats_df_display = stats_df.copy()
    stats_df_display["num_batches"] = stats_df_display["num_batches"].astype("int64")
    for c in ["mean", "std", "min", "max", "median"]:
        stats_df_display[c] = stats_df_display[c].round(6)
    print(stats_df_display)

    wandb.log({
        "train_loss_mean": stats_df.loc["train", "mean"],
        "train_loss_std": stats_df.loc["train", "std"],
        "val_loss_mean": stats_df.loc["val", "mean"],
        "val_loss_std": stats_df.loc["val", "std"],
        "test_loss_mean": stats_df.loc["test", "mean"],
        "test_loss_std": stats_df.loc["test", "std"],
    })

    wandb.finish()
    print(f"\nFinished run: expansion_factor={expansion_factor}, _lambda={_lambda}\n")


# %%
if __name__ == "__main__":
    for config in SWEEP_CONFIGS:
        run_training(**config)

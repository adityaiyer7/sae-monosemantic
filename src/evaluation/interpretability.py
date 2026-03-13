import sys
import re
import glob
import os
import subprocess
import torch
from collections import defaultdict
from typing import DefaultDict, Tuple, Optional
from pathlib import Path
from src.models.sparse_autoencoder import SparseAutoEncoder
from src.training.dataset_creator import natural_sort_key
from transformers import GPT2Tokenizer
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import wandb
from huggingface_hub import HfApi
from dotenv import load_dotenv


class ScalableFeatureExtractor:
    def __init__(self, model, device, expansion_factor: int, _lambda: float, batch_size:int = 512, output_buffer_size:int = 10000, threshold:float = 0.07) -> None:
        self.model = model
        self.batch_size = batch_size
        self.output_buffer_size = output_buffer_size
        self.device = device
        self.expansion_factor = expansion_factor
        self._lambda = _lambda
        self.feature_mapper = defaultdict(list)
        self.feature_mapper_size = 0
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.threshold = threshold

        # Setup output directory for parquet files
        self.output_dir = Path(f"features/features_{expansion_factor}x_{_lambda}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        

    
    def get_active_features(self, x: torch.Tensor) -> list[DefaultDict[int, float]]:
        """Return per-sample dicts mapping feature index to activation value for all features above threshold."""
        batch_index, active_index = (x > self.threshold).nonzero(as_tuple=True)
        result = [{} for _ in range(x.shape[0])]
        for batch, feature, value in zip(batch_index.tolist(), active_index.tolist(), x[batch_index, active_index].tolist()):
            result[batch][feature] = value
        return result

    
    def get_max_activating_features(self, active_features: list[DefaultDict[int, float]], top_k: Optional[int], sort_order: str = 'descending') -> list[DefaultDict[int, float]]:
        """Trim each sample's active features to the top_k by activation value.

        Args:
            active_features: Per-sample feature activation dicts.
            top_k: Maximum features to keep per sample. If None, returns unchanged.
            sort_order: 'descending' keeps highest activations; 'ascending' keeps lowest.
        """
        if top_k is None:
            return active_features
        res = []
        for j in range(len(active_features)):
            feature_map = active_features[j]
            if sort_order not in('descending','ascending'):
                raise ValueError("sort_order must be either descending or ascending")
            is_descending = (sort_order == 'descending')
            sorted_items = sorted(feature_map.items(), key = lambda x:x[1], reverse = is_descending)
            k = min(top_k, len(sorted_items))
            sorted_items = sorted_items[:k]
            res.append(dict(sorted_items))
        return res

    def fill_feature_mapper(self, max_active_features: list[dict[int, float]], token_id: torch.Tensor, context_token_ids: torch.Tensor) -> int:
        """Append activation records to the in-memory feature_mapper buffer.

        Each record stores (activation_value, token_id, context_token_ids) keyed by feature dimension.
        Returns the updated total number of buffered records.
        """
        context_token_ids_cpu = context_token_ids.cpu().tolist()
        for j in range(len(max_active_features)):
            for dimension_number, activation_value in max_active_features[j].items():
                self.feature_mapper[dimension_number].append((activation_value, token_id[j].item(), context_token_ids_cpu[j]))
                self.feature_mapper_size += 1
        return self.feature_mapper_size

    def flush_to_parquet(self, writer: pq.ParquetWriter, chunk_idx: int):
        """Flush current feature_mapper as a row group into the open ParquetWriter."""
        feature_ids, activation_values, token_ids, context_token_ids_list = [], [], [], []

        for feature_id, activations in self.feature_mapper.items():
            for activation_value, token_id, context_token_ids in activations:
                feature_ids.append(feature_id)
                activation_values.append(activation_value)
                token_ids.append(token_id)
                context_token_ids_list.append(context_token_ids)

        if feature_ids:
            table = pa.table({
                'feature_id': pa.array(feature_ids, type=pa.int32()),
                'activation_value': pa.array(activation_values, type=pa.float32()),
                'token_id': pa.array(token_ids, type=pa.int32()),
                'context_token_ids': pa.array(context_token_ids_list, type=pa.list_(pa.int32())),
                'chunk_id': pa.array([chunk_idx] * len(feature_ids), type=pa.int32()),
            })
            writer.write_table(table)
            print(f"  Flushed {len(feature_ids)} records")
            self.feature_mapper.clear()
            self.feature_mapper_size = 0


    def process_chunk_batched(self, chunk_file, chunk_idx: int, context_window: int = 10):
        """Run the SAE over a single activation chunk file and write results to a parquet file.

        Loads token IDs and residual activations from chunk_file, processes them in batches,
        and streams feature activation records to features/features_{N}x/chunk_{idx}.parquet.
        Context tokens within context_window positions of each token are stored alongside each record.
        """
        import time
        print(f"Loading chunk file: {chunk_file}")

        t_load = time.time()
        chunk_data = torch.load(chunk_file, map_location='cpu')
        print(f"  torch.load took: {time.time() - t_load:.2f}s")

        token_ids = chunk_data["token_ids"]
        activations = chunk_data["filtered_residual_activations"]

        print("activations and token_ids extracted")

        if token_ids.shape[0] != activations.shape[0]:
            raise ValueError(
            f"Mismatched lengths in {chunk_file}, "
            f"token_id shape = {token_ids.shape[0]} vs "
            f"activations shape = {activations.shape[0]}"
        )

        num_samples = token_ids.shape[0]
        print(f"Processing {num_samples} samples in batches of {self.batch_size}")

        schema = pa.schema([
            ('feature_id', pa.int32()),
            ('activation_value', pa.float32()),
            ('token_id', pa.int32()),
            ('context_token_ids', pa.list_(pa.int32())),
            ('chunk_id', pa.int32()),
        ])
        output_file = self.output_dir / f'chunk_{chunk_idx:04d}.parquet'

        with pq.ParquetWriter(output_file, schema) as writer:
            for i in range(0, num_samples, self.batch_size):
                t_batch_start = time.time()

                token_id_batch = token_ids[i: i + self.batch_size].to(self.device)
                activations_batch = activations[i: i + self.batch_size].to(self.device)

                # the idea here is that when building the context, we need positions relative to the global tensor, and not just the batch (since we can go out of bounds)
                # we're now computing the indices and then storing the token_ids (this will be decoded later during analysis) 
                # global position of each element in this batch
                positions = torch.arange(i, i + len(token_id_batch)) # shape: [batch_size]

                # for each position, gather context tokens
                # shape: [batch_size, 2*context_window+1]
                context_start_idx = -context_window
                context_end_index = context_window + 1
                offsets = torch.arange(context_start_idx, context_end_index, device='cpu')

                context_indices = positions.unsqueeze(1) + offsets
                context_indices = context_indices.clamp(0, num_samples - 1)
                context_token_ids = token_ids[context_indices]
 

                t_gpu_start = time.time()
                with torch.no_grad():
                    SAE_encoded_rep, _ = self.model.forward(activations_batch)

                    SAE_encoded_cpu = SAE_encoded_rep.cpu()
                    token_ids_cpu = token_id_batch.cpu()

                t_loop_start = time.time()
                gpu_time = t_loop_start - t_gpu_start



                active_feature_mapping = self.get_active_features(SAE_encoded_cpu)
                all_active_features = self.get_max_activating_features(active_feature_mapping, top_k = 25)

                length_counter = self.fill_feature_mapper(all_active_features, token_ids_cpu, context_token_ids)
                if length_counter > self.output_buffer_size:
                    self.flush_to_parquet(writer, chunk_idx)

                loop_time = time.time() - t_loop_start
                total_time = time.time() - t_batch_start

                if i % (5 * self.batch_size) == 0:  # Print every 5 batches
                    print(f"  Batch {i}/{num_samples}: GPU={gpu_time:.3f}s, Loop={loop_time:.3f}s, Total={total_time:.3f}s")

                del SAE_encoded_rep, SAE_encoded_cpu, activations_batch, token_id_batch, token_ids_cpu, context_token_ids
                if torch.cuda.is_available() and i % (10 * self.batch_size) == 0:
                    torch.cuda.empty_cache()

            # Final flush for any remaining records
            print("Flushing remaining records to parquet...")
            t_save = time.time()
            self.flush_to_parquet(writer, chunk_idx)
            print(f"  Final flush took: {time.time() - t_save:.2f}s")

        print(f"Saved {output_file}")
        return output_file




def main(expansion_factor: int, _lambda: float):
    # project_root = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
    project_root = Path("/workspace/sae-monosemantic")

    load_dotenv(project_root / ".env")

    print(f"WANDB_API_KEY: {str(os.environ.get('WANDB_API_KEY'))[:4]}...")
    print(f"HF_TOKEN: {str(os.environ.get('HF_TOKEN'))[:4]}...")
    activation_chunk_dir = str(project_root / 'data' / 'gpt2_activation_chunks')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    HF_DATASET_REPO = f"thedarkknight7/SAE_monosemanticity_features_{expansion_factor}x_{_lambda}"

    files = sorted(glob.glob(f"{activation_chunk_dir}/*.pt"), key = natural_sort_key)

    # Download model weights from HF bucket
    HF_BUCKET = "hf://buckets/thedarkknight7/sae-for-monosemanticity-model-weights"
    weight_filename = f"model_weights_{expansion_factor}x_{_lambda}.pth"
    local_weights_dir = project_root / "model_weights"
    local_weights_path = local_weights_dir / weight_filename

    if not local_weights_path.exists():
        print(f"Downloading {weight_filename} from HF bucket...")
        local_weights_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["hf", "sync", HF_BUCKET, str(local_weights_dir)],
            capture_output=True, text=True,
            env={**os.environ, "HF_TOKEN": os.environ.get("HF_TOKEN", "")},
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download weights: {result.stderr}")
    else:
        print(f"Using cached weights at {local_weights_path}")

    # Initialize wandb
    batch_size = 2048
    wandb_config = {
        "expansion_factor": expansion_factor,
        "_lambda": _lambda,
        "batch_size": batch_size,
        "threshold": 0.07,
        "top_k": 25,
        "num_chunks": len(files)
    }

    run = wandb.init(
        entity="adityaiyer-m-self",
        project="sae-for-monosemanticity",
        job_type="feature-extraction",
        config=wandb_config,
        name=f"feature-extraction-{expansion_factor}x_{_lambda}"
    )

    state_dict = torch.load(local_weights_path, weights_only=True, map_location=device)

    model = SparseAutoEncoder(d_model=768, expansion_factor=expansion_factor)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    feature_extractor = ScalableFeatureExtractor(
        model = model,
        device = device,
        expansion_factor = expansion_factor,
        _lambda = _lambda,
        batch_size = batch_size,
        output_buffer_size = 500000
    )

    # Process chunks and upload to wandb
    for chunk_idx, file in enumerate(files):
        expected_parquet = feature_extractor.output_dir / f'chunk_{chunk_idx:04d}.parquet'
        if expected_parquet.exists():
            print(f"Skipping chunk {chunk_idx} — {expected_parquet.name} already exists")
            continue
        parquet_file = feature_extractor.process_chunk_batched(file, chunk_idx)

    print(f"\nCompleted processing {len(files)} chunks.")
    print(f"Parquet files saved to: {feature_extractor.output_dir}")

    # Log HF dataset link to wandb
    hf_dataset_url = f"https://huggingface.co/datasets/{HF_DATASET_REPO}"
    run.config.update({"hf_dataset_url": hf_dataset_url})
    run.notes = f"Features dataset: {hf_dataset_url}"
    print(f"Logged HF dataset link to wandb: {hf_dataset_url}")

    wandb.finish()

    # Upload parquet files to HuggingFace dataset
    hf_token = os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)

    # Create the dataset repo if it doesn't exist
    api.create_repo(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        exist_ok=True,
    )
    print(f"Ensured HF dataset repo exists: {HF_DATASET_REPO}")

    parquet_files = sorted(feature_extractor.output_dir.glob("*.parquet"))
    print(f"\nPushing {len(parquet_files)} parquet files to HuggingFace...")
    api.upload_folder(
        folder_path=str(feature_extractor.output_dir),
        path_in_repo="data/",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Add feature activations for {expansion_factor}x lambda={_lambda}",
    )
    print(f"Pushed to {hf_dataset_url}")


if __name__ == "__main__":
    main(expansion_factor=4, _lambda=1e-4)

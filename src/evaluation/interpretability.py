import sys
import re
import glob
import os
import torch
from collections import defaultdict
from typing import DefaultDict, Tuple, Optional
from pathlib import Path
from src.models.spare_autoencoder import SparseAutoEncoder
from src.training.dataset_creator import natural_sort_key
from transformers import GPT2Tokenizer
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import wandb

class ScalableFeatureExtractor:
    def __init__(self, model, device, expansion_factor: int, batch_size:int = 512, output_buffer_size:int = 10000, threshold:float = 0.07) -> None:
        self.model = model
        self.batch_size = batch_size
        self.output_buffer_size = output_buffer_size
        self.device = device
        self.expansion_factor = expansion_factor
        self.feature_mapper = defaultdict(list)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.threshold = threshold

        # Setup output directory for parquet files
        self.output_dir = Path(f"features/features_{expansion_factor}x")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_counter = 0
        

    
    def get_active_features(self,x: torch.Tensor)-> list[DefaultDict[int, float]]:
        batch_index, active_index = (x > self.threshold).nonzero(as_tuple=True)
        result = [{} for _ in range(x.shape[0])]
        for batch, feature, value in zip(batch_index.tolist(), active_index.tolist(), x[batch_index, active_index].tolist()):
            result[batch][feature] = value
        return result

    
    def get_max_activating_features(self,active_features:list[DefaultDict[int, float]], top_k:Optional[int], sort_order:str='descending')->list[DefaultDict[int, float]]:
        if not top_k:
            return active_features
        res = []
        for j in range(len(active_features)):
            feature_map = active_features[j]
            assert(sort_order == 'descending' or sort_order == 'ascending')
            is_descending = (sort_order == 'descending')
            sorted_items = sorted(feature_map.items(), key = lambda x:x[1], reverse = is_descending)
            k = min(top_k, len(sorted_items))
            sorted_items = sorted_items[:k]
            res.append(dict(sorted_items))
        return res

    def fill_feature_mapper(self, max_active_features, token_id, context_token_ids):
        context_token_ids_cpu = context_token_ids.cpu().tolist()
        length_counter = 0
        for j in range(len(max_active_features)):
            for dimension_number, activation_value in max_active_features[j].items():
                self.feature_mapper[dimension_number].append((activation_value, token_id[j].item(), context_token_ids_cpu[j]))
                # this tells us how many records are being stored in the dictionary (how many activation values across dimensions)
                length_counter += 1
        return length_counter

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

   
    def process_chunk_batched(self, chunk_file, chunk_idx: int, context_window = 10):
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
        total_records = 0

        with pq.ParquetWriter(output_file, schema) as writer:
            for i in range(0, num_samples, self.batch_size):
                t_batch_start = time.time()

                token_id_batch = token_ids[i: i + self.batch_size].to(self.device)
                activations_batch = activations[i: i + self.batch_size].to(self.device)

                t_gpu_start = time.time()
                with torch.no_grad():
                    SAE_encoded_rep, results = self.model.forward(activations_batch)

                    SAE_encoded_cpu = SAE_encoded_rep.cpu()
                    token_ids_cpu = token_id_batch.cpu()

                t_loop_start = time.time()
                gpu_time = t_loop_start - t_gpu_start

                context_start_idx = -context_window
                context_end_index = context_window + 1
                offsets = torch.arange(context_start_idx, context_end_index, device=token_id_batch.device)

                context_token_ids = token_id_batch.unsqueeze(1) + offsets

                active_feature_mapping = self.get_active_features(SAE_encoded_cpu)
                all_active_features = self.get_max_activating_features(active_feature_mapping, top_k = None)

                length_counter = self.fill_feature_mapper(all_active_features, token_ids_cpu, context_token_ids)
                if length_counter > self.output_buffer_size:
                    self.flush_to_parquet(writer, chunk_idx)
                    total_records += length_counter

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




def main():
    # project_root = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
    project_root = Path("/workspace/sae-monosemantic")
    expansion_factor = 4
    MODEL_SAVE_PATH = project_root / f'model_weights_{expansion_factor}x.pth'
    activation_chunk_dir = str(project_root / 'data' / 'gpt2_activation_chunks')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    files = sorted(glob.glob(f"{activation_chunk_dir}/*.pt"), key = natural_sort_key)[:1]

    # Initialize wandb
    batch_size = 2048
    wandb_config = {
        "expansion_factor": expansion_factor,
        "batch_size": batch_size,
        "threshold": 0.07,
        "top_k": None,  # Using all active features above threshold
        "num_chunks": len(files)
    }

    run = wandb.init(
        entity="adityaiyer-m-self",
        project="sae-for-monosemanticity",
        job_type="feature-extraction",
        config=wandb_config,
        name=f"feature-extraction-{expansion_factor}x"
    )

    state_dict = torch.load(MODEL_SAVE_PATH, weights_only=True, map_location = device)

    model = SparseAutoEncoder(d_model=768, expansion_factor=expansion_factor)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    feature_extractor = ScalableFeatureExtractor(
        model = model,
        device = device,
        expansion_factor = expansion_factor,
        batch_size = batch_size
    )

    # Process chunks and upload to wandb
    for chunk_idx, file in enumerate(files):
        parquet_file = feature_extractor.process_chunk_batched(file, chunk_idx)

        # Upload parquet file to wandb as artifact
        if parquet_file:
            artifact = wandb.Artifact(
                name=f"feature-chunk-{expansion_factor}x",
                type="feature-data",
                description=f"Feature activations for {expansion_factor}x SAE model"
            )
            artifact.add_file(str(parquet_file))
            run.log_artifact(artifact)
            print(f"Uploaded {parquet_file.name} to wandb")

    print(f"\nCompleted processing {len(files)} chunks.")
    print(f"Parquet files saved to: {feature_extractor.output_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()

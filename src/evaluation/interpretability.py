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
        

    
    def get_active_features(self,x: torch.Tensor)-> DefaultDict[int, float]:
        active_idx = (x > self.threshold).nonzero(as_tuple = True)[0]
        return dict(zip(active_idx.tolist(), x[active_idx].tolist()))

    
    def get_max_activating_features(self,feature_map:DefaultDict[int, float], top_k:Optional[int], sort_order:str='descending')->dict[int, float]:
        assert(sort_order == 'descending' or sort_order == 'ascending')
        is_descending = (sort_order == 'descending')
        sorted_items = sorted(feature_map.items(), key = lambda x:x[1], reverse = is_descending)
        if top_k:
            top_k = min(top_k, len(sorted_items))
            sorted_items = sorted_items[:top_k]
        return dict[int, float](sorted_items)

    def fill_feature_mapper(self, max_active_features, token_id, context_token_ids):
        for dimension_number, activation_value in max_active_features.items():
            self.feature_mapper[dimension_number].append((activation_value, token_id, context_token_ids))


    def save_chunk_to_parquet(self):
        """Save current feature_mapper to parquet file and return the file path"""
        records = []
        for feature_id, activations in self.feature_mapper.items():
            for activation_value, token_id, context_token_ids in activations:
                records.append({
                    'feature_id': feature_id,  # feature_id is the SAE feature dimension index
                    'activation_value': activation_value,
                    'token_id': token_id,
                    'context_token_ids': context_token_ids,
                    'chunk_id': self.chunk_counter
                })

        if records:  # Only save if we have data
            # Convert feature records to DataFrame, save as Parquet file, clear memory, and return path for wandb upload
            df = pd.DataFrame(records)
            output_file = self.output_dir / f'chunk_{self.chunk_counter:04d}.parquet'
            df.to_parquet(output_file, index=False)
            print(f"Saved {len(records)} records to {output_file}")

            # Clear memory for next chunk
            self.feature_mapper.clear()
            self.chunk_counter += 1
            return output_file
        else:
            self.chunk_counter += 1
            return None

    def process_chunk_batched(self, chunk_file, context_window = 10):
        print(f"Loading chunk file: {chunk_file}")

        chunk_data = torch.load(chunk_file, map_location='cpu')
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


        for i in range(0, num_samples, self.batch_size):
            token_id_batch = token_ids[i: i + self.batch_size].to(self.device)
            activations_batch = activations[i: i + self.batch_size].to(self.device)

            with torch.no_grad():
                SAE_encoded_rep, results = self.model.forward(activations_batch)
            
                SAE_encoded_cpu = SAE_encoded_rep.cpu()
                token_ids_cpu = token_id_batch.cpu()

            


            for j in range(len(token_id_batch)):
                start_idx = max(0, i + j - context_window)
                end_idx = min(num_samples, i + j + context_window + 1)
                context_token_ids = token_ids[start_idx:end_idx].tolist()
                
                active_feature_mapping = self.get_active_features(SAE_encoded_cpu[j])
                # Pass all active features (no top_k filtering)
                all_active_features = self.get_max_activating_features(active_feature_mapping, top_k = None)

                self.fill_feature_mapper(all_active_features, token_ids_cpu[j].item(), context_token_ids = context_token_ids)
            
            del SAE_encoded_rep, SAE_encoded_cpu, activations_batch, token_id_batch, token_ids_cpu
            if torch.cuda.is_available() and i % (10 * self.batch_size) == 0:  # Every 10 batches
                # apparently empty cache is expensive, so reducing frequency to reduce overhead.
                torch.cuda.empty_cache()

        # Save chunk to parquet and return file path
        return self.save_chunk_to_parquet()



def main():
    # project_root = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
    project_root = Path("/workspace/sae-monosemantic")
    expansion_factor = 4
    MODEL_SAVE_PATH = project_root / f'model_weights_{expansion_factor}x.pth'
    activation_chunk_dir = str(project_root / 'data' / 'gpt2_activation_chunks')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    files = sorted(glob.glob(f"{activation_chunk_dir}/*.pt"), key = natural_sort_key)

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
    for file in files:
        parquet_file = feature_extractor.process_chunk_batched(file)

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

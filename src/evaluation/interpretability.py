import sys
import re
import glob
import os
import torch
from collections import defaultdict
from typing import DefaultDict, Tuple, Optional
from pathlib import Path
from src.models.spare_autoencoder import SparseAutoEncoder
from transformers import GPT2Tokenizer
import json


project_root = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
MODEL_SAVE_PATH = project_root / f'model_weights_4x.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
activation_chunk_dir = str(project_root / 'data' / 'gpt2_activation_chunks')

# Structure: {feature_id: [(activation_value, token_text, token_id, context), ...]}
feature_mapper = defaultdict(list)



def natural_sort_key(path):
    """Extract numbers from filename for proper numeric sorting."""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', path)]


files = sorted(glob.glob(f"{activation_chunk_dir}/*.pt"), key = natural_sort_key)[:2]


state_dict = torch.load(MODEL_SAVE_PATH, weights_only=True, map_location = 'cpu')


model = SparseAutoEncoder(d_model=768, expansion_factor=4)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def get_active_features(x: torch.Tensor, threshold:float = 0.07)-> DefaultDict[int, float]:
    '''
    x is going to be a two dimensional tensor
    '''
    assert x.ndim == 1
    active_idx = (x > threshold).nonzero(as_tuple = True)[0]
    return dict(zip(active_idx.tolist(), x[active_idx].tolist()))


def get_max_activating_features(feature_map:DefaultDict[int, float], top_k:Optional[int], sort_order:str='descending')->dict[int, float]:
    assert(sort_order == 'descending' or sort_order == 'ascending')
    is_descending = (sort_order == 'descending')
    sorted_items = sorted(feature_map.items(), key = lambda x:x[1], reverse = is_descending)
    if top_k:
        top_k = min(top_k, len(sorted_items))
        sorted_items = sorted_items[:top_k]
    return dict[int, float](sorted_items)

def fill_feature_mapper(max_active_features, token_text, token_id, context="NA"):
    for dimension_number, activation_value in max_active_features.items():
        feature_mapper[dimension_number].append((activation_value, token_text, token_id))


for chunk_file in files:
    print(f"Loading chunk file: {chunk_file}")
    chunk_data = torch.load(chunk_file, map_location=device)
    token_ids = chunk_data["token_ids"]
    activations = chunk_data["filtered_residual_activations"]

    print("activations and token_ids extracted")

    if token_ids.shape[0] != activations.shape[0]:
        raise ValueError(
            f"Mismatched lengths in {chunk_file}, "
            f"token_id shape = {token_ids.shape[0]} vs "
            f"activations shape = {activations.shape[0]}"
        )
    
    for i in range(token_ids.shape[0]):
        sample = {
                    "token_ids": token_ids[i],
                    "activations": activations[i]
        }
        
        SAE_encoded_rep, results = model.forward(sample["activations"])
    
        active_feature_mapping = get_active_features(SAE_encoded_rep)

        max_active_features = get_max_activating_features(active_feature_mapping, top_k = 10)
        token_text = tokenizer.decode([sample["token_ids"].item()])

        fill_feature_mapper(max_active_features, sample["token_ids"].item(), token_text)
    
    
    


feature_mapper = dict(sorted(feature_mapper.items(), key=lambda item: len(item[1]), reverse=True))
with open("feature_mapper.json", "w", encoding="utf-8") as f:
    json.dump(feature_mapper, f, ensure_ascii=False, indent=2)
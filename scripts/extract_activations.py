import os
from src.models.gpt2_wrapper import GPT2Wrapper
from datasets import load_dataset, Dataset, load_from_disk
import numpy as np
import torch
from torch.utils.data import DataLoader
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATASET_PATH = os.path.join("data", "openwebtext-10k")
CHUNK_DATA_PATH = os.path.join("data", "gpt2_activation_chunks")
# TRAINING_DATASET_PATH = os.path.join("data","sae_training_data_full.pt")
LOG_EVERY = 50
SAVE_EVERY = 100
model = GPT2Wrapper()

os.makedirs(CHUNK_DATA_PATH, exist_ok=True)

ds = load_from_disk(DATASET_PATH)

print(f"Loaded dataset from {DATASET_PATH}")
print(f"Successfully loaded {len(ds)} rows.")
print(f"Using device: {device}")

batch_size = 2
total_batches = (len(ds) + batch_size - 1) // batch_size
dataloader = DataLoader(ds, batch_size=batch_size)
print(f"DataLoader configured with batch_size={batch_size} ({total_batches} batches).")

buffer_token_ids = []
buffer_attention_masks = []
buffer_filtered_residual_activations = []

chunk_counter = 0

def save_buffer_to_disk(token_ids, masks, activations, chunk_id):
    if not token_ids:
        return [],[],[]
    print(f"Saving chunk_{chunk_id} to disk...")
    chunk_data = {
        'token_ids': torch.cat(token_ids, dim = 0),
        'attention_mask': torch.cat(masks, dim=0),
        'filtered_residual_activations': torch.cat(activations, dim = 0)
    }
    save_path = os.path.join(CHUNK_DATA_PATH, f"chunk_{chunk_id}.pt")
    torch.save(chunk_data, save_path)
    print(f"Saved chunk_{chunk_id} to disk...")
    return [], [], []
    


with torch.no_grad(): 
    start_time = time.time()
    running_examples = 0
    running_tokens = 0
    for batch_idx, batch in enumerate(dataloader, start=1):

        inputs = batch["text"]
        running_examples += len(inputs)

        encoded_results = model.encode(inputs)
        residual_state = model.forward(encoded_results) 


        attention_mask = encoded_results["attention_mask"].bool()
        running_tokens += int(attention_mask.sum().item())

        token_ids = encoded_results["input_ids"][attention_mask]
        filtered_residual_activations = residual_state[attention_mask]
        
        buffer_token_ids.append(token_ids.to(device))
        buffer_attention_masks.append(attention_mask.to(device))
        buffer_filtered_residual_activations.append(filtered_residual_activations.to(device))

        # periodic saving
        if batch_idx % SAVE_EVERY == 0:
            buffer_token_ids, buffer_attention_masks, buffer_filtered_residual_activations = save_buffer_to_disk(
                buffer_token_ids, buffer_attention_masks, buffer_filtered_residual_activations, chunk_counter
            )
            chunk_counter += 1

        # logging
        if batch_idx == 1 or batch_idx % LOG_EVERY == 0 or batch_idx == total_batches:
            elapsed = time.time() - start_time
            print(
                f"Progress {batch_idx}/{total_batches} | "
                f"examples={running_examples} | "
                f"tokens={running_tokens} | "
                f"elapsed={elapsed:.1f}s"
            )

# save any remaining batches 
if buffer_token_ids:
    save_buffer_to_disk(
            buffer_token_ids, buffer_attention_masks, buffer_filtered_residual_activations, chunk_counter
        )
            

# concatenate all batches
# print("concatenating all batches")
# token_ids_full = torch.cat(all_token_ids, dim = 0)
# attention_mask_full = torch.cat(all_attention_masks, dim = 0)
# filtered_residual_activations_full = torch.cat(all_filtered_residual_activations, dim = 0)

# print(
#     "Final shapes: "
#     f"token_ids={tuple(token_ids_full.shape)} | "
#     f"attention_mask={tuple(attention_mask_full.shape)} | "
#     f"activations={tuple(filtered_residual_activations_full.shape)}"
# )

# print("saving to disk")                                                                                    
# # save to disk 
# torch.save({
#     'token_ids': token_ids_full,
#     'attention_mask':attention_mask_full,
#     'filtered_residual_activations':filtered_residual_activations_full
# }, TRAINING_DATASET_PATH)

import os
from src.models.gpt2_wrapper import GPT2Wrapper
from datasets import load_dataset, Dataset, load_from_disk
import numpy as np
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATASET_PATH = os.path.join("data", "openwebtext-10k")
TRAINING_DATASET_PATH = os.path.join("data","sae_training_data.pt")
model = GPT2Wrapper()


ds = load_from_disk(DATASET_PATH).select(range(10))

print(f"Successfully loaded {len(ds)} rows.")

dataloader = DataLoader(ds, batch_size=2)

all_token_ids = []
all_attention_masks = []
all_filtered_residual_activations = []


with torch.no_grad(): 
    for batch in dataloader:

        inputs = batch["text"]

        encoded_results = model.encode(inputs)
        residual_state = model.forward(encoded_results) 


        attention_mask = encoded_results["attention_mask"].bool()

        token_ids = encoded_results["input_ids"][attention_mask]
        filtered_residual_activations = residual_state[attention_mask]
        
        all_token_ids.append(token_ids.to(device))
        all_attention_masks.append(attention_mask.to(device))
        all_filtered_residual_activations.append(filtered_residual_activations.to(device))

# concatenate all batches
print("concatenating all batches")
token_ids_full = torch.cat(all_token_ids, dim = 0)
attention_mask_full = torch.cat(all_attention_masks, dim = 0)
filtered_residual_activations_full = torch.cat(all_filtered_residual_activations, dim = 0)

print("saving to disk")                                                                                    
# save to disk 
torch.save({
    'token_ids': token_ids_full,
    'attention_mask':attention_mask_full,
    'filtered_residual_activations':filtered_residual_activations_full
}, TRAINING_DATASET_PATH)




# Old code - no longer needed.

# def get_residual_activations(input_text, layer_num = 6): 
#     encoded = model.encode(input_text)
#     residual_state = model.forward(encoded, layer_num) 

#     # print(residual_state)
#     # print(type(residual_state))
#     # print(residual_state.size())

#     mask = encoded["attention_mask"].bool()
#     activations_filtered = residual_state[mask]
#     return activations_filtered
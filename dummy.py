# dummy file used for scratch padding. 
import torch

# Load the saved data
data = torch.load('sae_training_data.pt')

token_ids = data['token_ids']
attention_masks = data['attention_mask']
filtered_residuals = data['filtered_residual_activations']


print(filtered_residuals)
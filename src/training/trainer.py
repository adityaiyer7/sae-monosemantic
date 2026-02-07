import torch
from torch.utils.data import Dataset
from src.models.spare_autoencoder import SparseAutoEncoder
from dataset_creator import ChunkIterableGenerator


training_chunk_dir = 'data/sae_training_chunks'
iterable_chunk_generator = ChunkIterableGenerator(training_chunk_dir)

dataloader = torch.utils.data.DataLoader(iterable_chunk_generator, batch_size = 32, num_workers=0)
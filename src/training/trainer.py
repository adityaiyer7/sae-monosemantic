import torch
from itertools import islice
from torch.utils.data import Dataset, DataLoader
from src.models.spare_autoencoder import SparseAutoEncoder
from dataset_creator import ChunkIterableGenerator
from losses import compute_loss
from typing import Optional, Callable


class SAETrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None
    ):

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
    
    def train(self, num_epochs: int, log_every: int):
        self.model.train()
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate((self.dataloader)):
                token_ids = batch["token_ids"].to(self.device)
                activations = batch["activations"].to(self.device)

                features, results = self.model.forward(activations)

                loss = self.loss_fn(results, activations, features)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")


    def evaluate(self, dataloader : Optional[DataLoader]):
        eval_data = dataloader or self.dataloader
        self.model.eval()
    
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_data):
                token_ids = batch["token_ids"].to(self.device)
                activations = batch["activations"].to(self.device)

                features, results = self.model.forward(activations)
                loss = self.loss_fn(results, activations, features)

                total_loss += loss.item()
                num_batches += 1
        return total_loss/num_batches if num_batches > 0 else 0.0


training_chunk_dir = 'data/gpt2_activation_chunks'
num_epochs = 1
iterable_chunk_generator = ChunkIterableGenerator(training_chunk_dir)


model = SparseAutoEncoder(d_model=768, expansion_factor=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

dataloader = torch.utils.data.DataLoader(iterable_chunk_generator, batch_size=32, num_workers=0)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

loss_fn = compute_loss

trainer = SAETrainer(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device
)

trainer.train(num_epochs=1, log_every=1)




# model = SparseAutoEncoder(d_model = 768, expansion_factor = 4)
# dataloader = torch.utils.data.DataLoader(iterable_chunk_generator, batch_size = 32, num_workers=0)
# optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

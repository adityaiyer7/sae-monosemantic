import torch
from itertools import islice
from torch.utils.data import Dataset, DataLoader
from src.models.sparse_autoencoder import SparseAutoEncoder
from src.training.dataset_creator import ChunkIterableGenerator
from src.training.losses import compute_loss
from typing import Optional, Callable
import torch.nn.functional as F


class SAETrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        _lambda: float = 0.01,
    ):

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self._lambda = _lambda
        self.training_losses: list[float] = []
        

    def train(self, num_epochs: int, log_every: int):
        self.model.train()
        self.training_losses = []
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate((self.dataloader)):
                token_ids = batch["token_ids"].to(self.device)
                activations = batch["activations"].to(self.device)

                features, results = self.model.forward(activations)

                loss = self.loss_fn(self._lambda, results, activations, features)
                self.training_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()

                if not self.model.tie_weights:
                    with torch.no_grad():
                        W = self.model.W_dec
                        W_norm = F.normalize(W, dim = 1)
                        parallel = (self.model.W_dec.grad * W_norm).sum(dim = 1, keepdim = True) * W_norm
                        self.model.W_dec.grad -= parallel
            

                self.optimizer.step()

                if not self.model.tie_weights:
                    with torch.no_grad():
                        self.model.W_dec.data = F.normalize(self.model.W_dec.data, dim = 1)

                if batch_idx % log_every == 0 or batch_idx == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")


    def evaluate(self, dataloader : Optional[DataLoader]):
        eval_data = dataloader if dataloader is not None else self.dataloader
        self.model.eval()
    
        total_loss = 0.0
        num_batches = 0

        evaluation_loss = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_data):
                token_ids = batch["token_ids"].to(self.device)
                activations = batch["activations"].to(self.device)

                features, results = self.model.forward(activations)
                loss = self.loss_fn(self._lambda, results, activations, features)
                evaluation_loss.append(loss.item())

                total_loss += loss.item()
                num_batches += 1
        return evaluation_loss, num_batches


# training_chunk_dir = 'data/gpt2_activation_chunks'
# num_epochs = 1
# iterable_chunk_generator = ChunkIterableGenerator(training_chunk_dir)


# model = SparseAutoEncoder(d_model=768, expansion_factor=4)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

# dataloader = torch.utils.data.DataLoader(iterable_chunk_generator, batch_size=32, num_workers=0)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# loss_fn = compute_loss

# trainer = SAETrainer(
#     model=model,
#     dataloader=dataloader,
#     optimizer=optimizer,
#     loss_fn=loss_fn,
#     device=device
# )

# trainer.train(num_epochs=1, log_every=1)




# model = SparseAutoEncoder(d_model = 768, expansion_factor = 4)
# dataloader = torch.utils.data.DataLoader(iterable_chunk_generator, batch_size = 32, num_workers=0)
# optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

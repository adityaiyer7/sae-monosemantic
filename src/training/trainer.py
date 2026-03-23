import torch
from itertools import islice
from torch.utils.data import Dataset, DataLoader
from src.models.sparse_autoencoder import SparseAutoEncoder
from src.training.dataset_creator import ChunkIterableGenerator
from src.training.losses import compute_loss
from src.training.utils import set_seed
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
        seed: int = 42,
    ):
        set_seed(seed)

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self._lambda = _lambda
        self.training_losses: list[float] = []
        self.steps = 0
        self.neuron_fire_counts = torch.zeros(model.d_hidden, dtype=torch.long).to(self.device)
        

    def train(self, num_epochs: int, log_every: int):
        self.model.train()
        self.training_losses = []
        self.steps = 0
        resample_every = 25000
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
                self.steps += 1
                fired = (features > 0).any(dim=0)  # shape [d_hidden], bool
                self.neuron_fire_counts += fired.long()

                # resampling dead neurons, see comments below (EOF) for explanation
                if self.steps % resample_every == 0:
                    print(f"resampling, steps = {self.steps}")
                    dead_neurons = (self.neuron_fire_counts == 0).nonzero(as_tuple = True)[0] # this gives us the idxs of dead neurons

                    # Reset Nueron fire Counts to 0
                    self.neuron_fire_counts.zero_()

                    if len(dead_neurons) > 0:
                            
                        print(f"Batch = {batch_idx}, detected {len(dead_neurons)} dead neurons")

                        # pull batch from data loader and run forward pass
                        # We're going ahead with the existing batch because this is as good as any random batch - no reason to expect it to be any different. 
                        # next(iter(self.dataloader)) - creates a new instance, so this would always train on the same first sample. 
                        # there is a way to maintain the state to get a more random sample, but it doesn't seem to justify the effort. 
                        resample_batch = batch # [batch_size, d_model]
                        resample_activations = activations

                        
                        resample_features, resample_results = features, results

                        # compute loss on that sample (computing L2 loss)
                        per_sample_loss = ((resample_activations - resample_results) ** 2).sum(dim = 1)     # [batch_size]
                        
                        # normalize the loss to build distribution
                        sampling_prob = (per_sample_loss)/(per_sample_loss).sum()

                        # use loss computation to build sampling distribution
                        sampled_indices = torch.multinomial(sampling_prob, num_samples = len(dead_neurons), replacement = True) # 1d tensor of indices, of length len(dead_neurons), eg tensor([42, 7, 42])

                        # Sample one input per dead neuron from that distribution
                        # sampled_indices[i] gives us the input to use for dead neuron i 
                    
                        # Reinitialize weights for each dead neuron using its sampled input
                        for idx, sample_idx in enumerate(sampled_indices):
                            vec = F.normalize(resample_activations[sample_idx], dim = 0)
                            self.model.W_enc[:,dead_neurons[idx]] = vec
                            self.model.W_dec[dead_neurons[idx], :] = vec
                        
                        self.model.b_enc.data[dead_neurons] = -torch.rand(len(dead_neurons), device=self.device) * 0.01

                        # Reset Adam state for those specific parameters 
                        if self.model.W_enc in self.optimizer.state:
                            self.optimizer.state[self.model.W_enc]['exp_avg'][:, dead_neurons] = 0
                            self.optimizer.state[self.model.W_enc]['exp_avg_sq'][:, dead_neurons] = 0
                        
                        if self.model.W_dec in self.optimizer.state:
                            self.optimizer.state[self.model.W_dec]['exp_avg'][dead_neurons, :] = 0
                            self.optimizer.state[self.model.W_dec]['exp_avg_sq'][dead_neurons, :] = 0
                        
                        if self.model.b_enc in self.optimizer.state:
                            self.optimizer.state[self.model.b_enc]['exp_avg'][dead_neurons] = 0
                            self.optimizer.state[self.model.b_enc]['exp_avg_sq'][dead_neurons] = 0

                    
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


# -- Sampling Strategy -- # 

# 1. Find Dead Neurons at Checkpoints
# 2. Take a random sample of input, compute loss on each data point in that sample. 
# 3. Assign each data point in the above set a sampling probability proportional to its loss squared (so datapoints that have high loss are likely to be sampled again - this helps the SAE overall)
# (we make this proportional to L2 loss, since it includes square term anyway)
# 4. Take a high loss input vector (input with reference to SAE, basically the activation vector) and normalize it. 
# 5. Set W_enc[:, dead_neuron] to that normalized vector 
# 6. Set W_dec[dead_neuron, :] to that normalized vector 
# 7. Reset the encoder bias to a small negative value or 0 (to control initial activation threshold )
# 8. Reset the Adam Optimizer State
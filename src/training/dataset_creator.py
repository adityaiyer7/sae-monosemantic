import torch
import glob
import re 

def natural_sort_key(path):
    """Extract numbers from filename for proper numeric sorting."""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', path)]

class ChunkIterableGenerator(torch.utils.data.IterableDataset):
    """An iterable dataset that loads and yields data chunks from PyTorch files.
    
    This class extends PyTorch's IterableDataset to efficiently load pre-processed
    activation data stored in chunk files. It supports multi-worker data loading
    by distributing chunk files across workers, and optionally applies transforms
    to each sample.
    
    Attributes:
        chunk_dir (str): Directory path containing the chunk files (.pt files).
        transform (callable, optional): Optional transform function to apply to each sample.
        chunk_files (list): Sorted list of chunk file paths.
    """
    def __init__(self,chunk_dir, transform = None):
        """Initialize the ChunkIterableGenerator.
        
        Args:
            chunk_dir (str): Path to the directory containing chunk files (.pt files).
            transform (callable, optional): Optional transform function to apply to each sample.
                The transform should accept a dictionary with 'token_ids' and 'activations' keys
                and return a transformed dictionary. Defaults to None.
        """
        super().__init__()
        self.chunk_dir = chunk_dir
        self.transform = transform
        self.chunk_files = sorted(glob.glob(f"{self.chunk_dir}/*.pt"), key = natural_sort_key)
        

    def __iter__(self):
        """Iterate over chunk files and yield individual samples.
        
        This method handles multi-worker data loading by distributing chunk files
        across workers. Each chunk file is loaded and split into individual samples
        containing token_ids and activations. If a transform is specified, it is
        applied to each sample before yielding.
        
        Yields:
            dict: A dictionary containing:
                - 'token_ids' (torch.Tensor): Token IDs for the sample.
                - 'activations' (torch.Tensor): Filtered residual activations for the sample.
                The dictionary may be transformed if self.transform is not None.
        
        Raises:
            ValueError: If token_ids and activations have mismatched lengths in a chunk file.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:  
            files = self.chunk_files[worker_info.id::worker_info.num_workers]
        else:  
            files = self.chunk_files
        
        for chunk_file in files:
            chunk_data = torch.load(chunk_file, map_location="cpu")
            token_ids = chunk_data["token_ids"]
            activations = chunk_data["filtered_residual_activations"]

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
                if self.transform:
                    sample = self.transform(sample)
                yield sample
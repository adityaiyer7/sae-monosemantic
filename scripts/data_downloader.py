import os
from datasets import load_dataset, Dataset, load_from_disk

SAVE_PATH = os.path.join("data", "openwebtext-10k")

def create_and_save_dataset(num_samples):
    print("Loading the official OpenWebText in streaming mode to avoid massive download")
    ds_stream = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    print(f"Taking the first {num_samples} entries")
    dataset_head = list(ds_stream.take(num_samples))

    ds = Dataset.from_list(dataset_head)

    print(f"Saving dataset to {SAVE_PATH}...")
    ds.save_to_disk(SAVE_PATH)
    print("Dataset saved.")
    return ds

def print_sample(save_path):
    ds = load_from_disk(save_path)
    print(f"Successfully loaded {len(ds)} rows.")
    df_head = ds.select(range(10)).to_pandas()
    print(df_head)


if __name__ == "__main__":
    # Check if we already have it saved to save time
    if os.path.exists(SAVE_PATH):
        print(f"Found existing dataset at {SAVE_PATH}. Loading from disk...")
        ds = load_from_disk(SAVE_PATH)
        print("Here is a sample from the dataset")

    else:
        ds = create_and_save_dataset(10000)

    print(f"\nDataset Info:\n{ds}")
    print(f"Example 0 text length: {len(ds[0]['text'])}")
    print("Here is a sample from the dataset")
    print(print_sample(SAVE_PATH))

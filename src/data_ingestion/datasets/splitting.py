import random

from datasets import DatasetDict

from src.core.names import COLS


def split_dataset(dataset, num_splits, split_ratio, seed):
    """
    Split the dataset into multiple train splits and a test set.

    Args:
        dataset
        num_splits (int): Number of train splits.
        split_ratio (float): Proportion of data for each train split.
        seed (int): Seed for shuffling to ensure reproducibility.

    Returns:
        DatasetDict: A dictionary containing the train splits and test set.
    """

    # Shuffle the dataset
    dataset = dataset.shuffle(seed=seed)
    # Keep original indices while shuffling
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    dataset = dataset.select(indices)
    # Add original indices as a feature
    dataset = dataset.map(lambda example, idx: {COLS.ORIGINAL_IDX: indices[idx]}, with_indices=True)
    # Calculate sizes
    num_examples = len(dataset)
    split_size = int(split_ratio * num_examples)

    # Create splits
    splits = {}
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = start_idx + split_size
        split_name = f"train{i + 1}"
        splits[split_name] = dataset.select(range(start_idx, end_idx)).map(lambda x: {"split": split_name})

    # Remaining data for test split
    remaining_start_idx = num_splits * split_size
    splits["test"] = dataset.select(range(remaining_start_idx, num_examples)).map(lambda x: {"split": "test"})

    return DatasetDict(splits)

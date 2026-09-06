"""Training pair sampling and batch collation."""

import random

import torch
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    """Build training samples from positive edges and sampled unknown pairs."""

    def __init__(self, positive_edges, negative_pool, negatives_per_positive=3, rng=None):
        if not positive_edges:
            raise ValueError("The training domain contains no positive edges.")
        if not negative_pool:
            raise ValueError("The training domain contains no unknown pairs for negative sampling.")
        self.positive_edges = positive_edges
        self.negative_pool = negative_pool
        self.negatives_per_positive = negatives_per_positive
        self.rng = random.Random(0) if rng is None else rng

    def __len__(self):
        return len(self.positive_edges)

    def __getitem__(self, index):
        drug, disease = self.positive_edges[index]
        negatives = [
            (*self.rng.choice(self.negative_pool), 0)
            for _ in range(self.negatives_per_positive)
        ]
        return (drug, disease, 1), negatives


def collate_training_pairs(batch):
    """Flatten positive-centered samples into model input tensors."""
    drug_indices, disease_indices, labels = [], [], []
    for positive, negatives in batch:
        for drug, disease, label in (positive, *negatives):
            drug_indices.append(drug)
            disease_indices.append(disease)
            labels.append(label)
    return (
        torch.tensor(drug_indices, dtype=torch.long),
        torch.tensor(disease_indices, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )

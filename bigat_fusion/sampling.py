import random

import torch
from torch.utils.data import Dataset


class GlobalUnknownTrainingDataset(Dataset):


    def __init__(self, positive_edges, n_drugs, n_diseases, positive_set,
                 negatives_per_positive=3, rng=None):
        self.positive_edges = positive_edges
        self.n_drugs = n_drugs
        self.n_diseases = n_diseases
        self.positive_set = set(positive_set)
        self.negatives_per_positive = negatives_per_positive
        self.rng = random.Random(0) if rng is None else rng

    def __len__(self):
        return len(self.positive_edges)

    def __getitem__(self, index):
        negatives = []
        while len(negatives) < self.negatives_per_positive:
            pair = (self.rng.randrange(self.n_drugs), self.rng.randrange(self.n_diseases))
            if pair not in self.positive_set:
                negatives.append((*pair, 0))
        return (*self.positive_edges[index], 1), negatives


def collate_training_pairs(batch):

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

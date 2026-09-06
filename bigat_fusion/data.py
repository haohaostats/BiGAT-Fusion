"""Dataset loading and similarity graph construction."""

from pathlib import Path

import numpy as np
import scipy.io as sio


def knn_adjacency(similarity: np.ndarray, k: int) -> list[list[int]]:
    """Build a directed k-nearest-neighbor list with self connections."""
    size = similarity.shape[0]
    adjacency = [[] for _ in range(size)]
    for index in range(size):
        neighbors = np.argsort(-similarity[index])
        neighbors = neighbors[neighbors != index][:k]
        adjacency[index] = [index] + neighbors.tolist()
    return adjacency


def load_dataset(path="data/Gdataset/Gdataset.mat", *, k=4, **_):
    """Load association labels, similarities, and entity names from a MAT file."""
    mat_path = Path(path)
    if not mat_path.is_file() or mat_path.suffix.lower() != ".mat":
        raise FileNotFoundError(f"Dataset .mat file not found: {mat_path}")

    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    association = mat["didr"].astype(np.int8).T
    disease_similarity = mat["disease"].astype(np.float32)
    drug_similarity = mat["drug"].astype(np.float32)

    n_drugs, n_diseases = association.shape
    return {
        "n_drugs": n_drugs,
        "n_diseases": n_diseases,
        "drug_feat_neighbors": knn_adjacency(drug_similarity, k),
        "disease_feat_neighbors": knn_adjacency(disease_similarity, k),
        "drug_names": [str(value) for value in mat["Wrname"].flatten()],
        "disease_names": [str(value) for value in mat["Wdname"].flatten()],
        "assoc_pos_set": set(map(tuple, np.argwhere(association == 1))),
    }

#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import scipy.io as sio


def _knn_adj(sim: np.ndarray, k: int):
    n = sim.shape[0]
    adj = [[] for _ in range(n)]
    for i in range(n):
        idx = np.argsort(-sim[i])
        idx = idx[idx != i][:k]
        adj[i] = [i] + idx.tolist()
    return adj


def load_dataset(path="data/Gdataset/Gdataset.mat", *, k=4, **_):
    mat_path = Path(path)
    if not mat_path.is_file() or mat_path.suffix.lower() != ".mat":
        raise FileNotFoundError(f"Gdataset .mat file not found: {mat_path}")

    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    assoc = mat["didr"].astype(np.int8).T
    disease_sim = mat["disease"].astype(np.float32)
    drug_sim = mat["drug"].astype(np.float32)

    n_drugs, n_diseases = assoc.shape
    return {
        "n_drugs": n_drugs,
        "n_diseases": n_diseases,
        "drug_feat_neighbors": _knn_adj(drug_sim, k),
        "disease_feat_neighbors": _knn_adj(disease_sim, k),
        "drug_names": [str(x) for x in mat["Wrname"].flatten()],
        "disease_names": [str(x) for x in mat["Wdname"].flatten()],
        "assoc_pos_set": set(map(tuple, np.argwhere(assoc == 1))),
    }

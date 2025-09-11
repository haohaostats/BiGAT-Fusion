
#!/usr/bin/env python3
# data_loader.py 

import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path

def _knn_adj(sim: np.ndarray, k: int):

    n = sim.shape[0]
    adj = [[] for _ in range(n)]
    for i in range(n):
        idx = np.argsort(-sim[i])               
        idx = idx[idx != i][:k]                  
        adj[i] = [i] + idx.tolist()             
    return adj

def _split_data(assoc: np.ndarray, val_r: float, test_r: float, seed: int):

    n_d, n_p = assoc.shape
    pos = np.argwhere(assoc == 1)
    rng = np.random.default_rng(seed); rng.shuffle(pos)
    n = len(pos); n_t = int(n * test_r); n_v = int(n * val_r)
    test  = [tuple(e) for e in pos[:n_t]]
    val   = [tuple(e) for e in pos[n_t:n_t + n_v]]
    train = [tuple(e) for e in pos[n_t + n_v:]]

    adj_d = {i: [] for i in range(n_d)}   # disease neighbors for each drug (drug <- disease edges)
    adj_p = {j: [] for j in range(n_p)}   # drug neighbors for each disease (disease <- drug edges)
    for d, p in train:
        adj_d[d].append(p)
        adj_p[p].append(d)

    all_pairs = {(i, j) for i in range(n_d) for j in range(n_p)}
    pos_set = set(train) | set(val) | set(test)
    neg = list(all_pairs - pos_set); rng.shuffle(neg)
    trn_neg = neg[:len(train)]
    val_neg = neg[len(train):len(train) + len(val)]
    tes_neg = neg[len(train) + len(val):len(train) + len(val) + len(test)]
    return train, val, test, trn_neg, val_neg, tes_neg, adj_d, adj_p

def _load_cg_mat(mat_path: Path, k: int, v: float, t: float, seed: int):
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    didr        = mat["didr"].astype(np.int8)       # (disease, drug) in some distributions
    disease_sim = mat["disease"].astype(np.float32)
    drug_sim    = mat["drug"].astype(np.float32)

    assoc = didr.T                                  # (drug, disease)
    n_d, n_p = assoc.shape

    drug_names    = [str(x) for x in mat["Wrname"].flatten()]   # columns → drugs
    disease_names = [str(x) for x in mat["Wdname"].flatten()]   # rows    → diseases

    d_knn = _knn_adj(drug_sim, k)
    p_knn = _knn_adj(disease_sim, k)

    tr, va, te, trn, van, ten, adj_d, adj_p = _split_data(assoc, val_r=v, test_r=t, seed=seed)

    return dict(
        n_drugs=n_d, n_diseases=n_p,
        drug_feat_neighbors=d_knn,
        disease_feat_neighbors=p_knn,
        drug_neighbors=adj_d, disease_neighbors=adj_p,
        train_edges=tr, val_edges=va, test_edges=te,
        train_neg_edges=trn, val_neg_edges=van, test_neg_edges=ten,
        drug_names=drug_names, disease_names=disease_names,
        assoc_pos_set=set(map(tuple, np.argwhere(assoc == 1))),
    )

def _load_ld_csv(dir_path: Path, k: int, v: float, t: float, seed: int):
    dis = pd.read_csv(dir_path / "dis_sim.csv",  header=None).values.astype(np.float32)
    dru = pd.read_csv(dir_path / "drug_sim.csv", header=None).values.astype(np.float32)
    didr= pd.read_csv(dir_path / "drug_dis.csv", header=None).values.astype(np.int8)

    assoc = didr                               # (drug, disease)
    n_d, n_p = assoc.shape
    drug_names    = [f"drug_{i}"    for i in range(n_d)]
    disease_names = [f"disease_{i}" for i in range(n_p)]

    d_knn = _knn_adj(dru, k); p_knn = _knn_adj(dis, k)
    tr, va, te, trn, van, ten, adj_d, adj_p = _split_data(assoc, v, t, seed)

    return dict(
        n_drugs=n_d, n_diseases=n_p,
        drug_feat_neighbors=d_knn,
        disease_feat_neighbors=p_knn,
        drug_neighbors=adj_d, disease_neighbors=adj_p,
        train_edges=tr, val_edges=va, test_edges=te,
        train_neg_edges=trn, val_neg_edges=van, test_neg_edges=ten,
        drug_names=drug_names, disease_names=disease_names,
        assoc_pos_set=set(map(tuple, np.argwhere(assoc == 1))),
    )

def _load_lrssl_txt(dir_path: Path, k: int, v: float, t: float, seed: int):
    dis_df = pd.read_csv(dir_path / "dis_sim.txt",  sep="\t", header=0, index_col=0)
    dru_df = pd.read_csv(dir_path / "drug_sim.txt", sep="\t", header=0, index_col=0)
    didr_df= pd.read_csv(dir_path / "drug_dis.txt", sep="\t", header=0, index_col=0)

    dis = dis_df.astype(float).values
    dru = dru_df.astype(float).values
    didr= didr_df.astype(int).values             # (drug, disease)

    assoc = didr
    n_d, n_p = assoc.shape
    drug_names    = list(dru_df.index)
    disease_names = list(dis_df.index)

    d_knn = _knn_adj(dru, k); p_knn = _knn_adj(dis, k)
    tr, va, te, trn, van, ten, adj_d, adj_p = _split_data(assoc, v, t, seed)

    return dict(
        n_drugs=n_d, n_diseases=n_p,
        drug_feat_neighbors=d_knn,
        disease_feat_neighbors=p_knn,
        drug_neighbors=adj_d, disease_neighbors=adj_p,
        train_edges=tr, val_edges=va, test_edges=te,
        train_neg_edges=trn, val_neg_edges=van, test_neg_edges=ten,
        drug_names=drug_names, disease_names=disease_names,
        assoc_pos_set=set(map(tuple, np.argwhere(assoc == 1))),
    )

def load_dataset(path: str, *, k: int = 4, val_ratio: float = 0.1, test_ratio: float = 0.2, seed: int = 42):

    p = Path(path)
    if p.suffix == ".mat":
        return _load_cg_mat(p, k, val_ratio, test_ratio, seed)
    elif p.name.lower() == "ldataset":
        return _load_ld_csv(p, k, val_ratio, test_ratio, seed)
    elif p.name.lower() == "lrssl":
        return _load_lrssl_txt(p, k, val_ratio, test_ratio, seed)
    else:
        raise FileNotFoundError(f"Unsupported dataset path: {p}")

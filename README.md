
# BiGAT-Fusion · Bidirectional Graph Attention with Node-wise Gated Fusion for Drug–Disease Association Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#-license)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)
![OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-555.svg)

> 🚀 Official PyTorch implementation of our paper: "BiGAT-Fusion: Bidirectional Graph Attention with Node-wise Gated Fusion for Drug–Disease Association Prediction".
---

## ✨ Highlights

- 🧠 **Two-view encoders**: feature-view GAT (drug/drug & disease/disease kNN graphs) + topology-view **Bi-GAT** (drug⇄disease).
- 🔀 **Node-wise gated fusion**: type-specific gates adaptively mix feature and topology embeddings per node.
- 🧩 **Residual-MoE head**: MLP main path + **low-rank bilinear** residual with a bounded pairwise gate.
- 🧪 **Rigorous CV**: 10×10 cross-validation; validation-driven LR scheduling; BCE with negative sampling.
- 📊 **Artifacts for analysis**: per-fold metrics, checkpoints, attentions (both directions), and fusion gates.

---

## 🗂️ Repository Structure

```
.
├── data_loader.py    # Dataset loading + kNN feature graphs + CV-friendly splitting
├── bigat_model.py    # BiGAT-Fusion model (encoders + gated fusion + Residual-MoE)
├── run_fullcv.py     # 10×10 CV training/evaluation
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

```bash
# (optional) create a fresh environment
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```
✅ **Minimal requirements**: `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch` (≥1.12)

### 🧰 Data Formats

The loader supports four common DDA benchmark formats:

- **C/G .mat** (`Gdataset.mat`, `Cdataset.mat`)  
  Required variables:
  - `didr` (association matrix; some distributions store it as [disease, drug])
  - `disease` (disease–disease similarity)
  - `drug` (drug–drug similarity)
  - `Wrname` (drug names)
  - `Wdname` (disease names)

- **Ldataset** (CSV directory)  
  Files:
  - `dis_sim.csv`, `drug_sim.csv`, `drug_dis.csv`

- **LRSSL** (TSV directory)  
  Files (tab-separated):
  - `dis_sim.txt`, `drug_sim.txt`, `drug_dis.txt`

📁 Place datasets under `datasets/` and reference them via `--mat_path`.

---

## ▶️ Reproduce 10×10 CV

```bash
python run_fullcv.py --mat_path data/Gdataset/Gdataset.mat --folds 10 --repeats 10 --device cuda
```

### 🔧 Key Defaults
- 🔁 **Negative sampling**: `neg_k=3` (training uses unweighted BCE; the batch prior comes from sampling)
- 🎯 **Attention normalization**: destination-wise stabilized softmax (subtract a batchwise constant)
- 🪫 **Weight decay**: backbone `1e-4`, gates `1e-3`
- 📉 **LR scheduling**: `ReduceLROnPlateau` on validation AUROC
- ✂️ **Gradient clipping**: `5.0`; **Dropout**: `0.2`

### 📤 Outputs
- `results/<dataset>/cv_log_full.csv` — per-fold AUROC/AUPRC
- `train/full_<dataset>_r<R>_f<F>.pt` — best checkpoints
- `interpret/folds/<dataset>/full/r<R>_f<F>.npz` — attentions (drug→disease & disease→drug) and node-wise gates

---

---

## 🧪 Reproducibility

We set seeds and:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
Minor numerical differences can occur across hardware/BLAS stacks; metrics are robust to small jitter.

---

## 🧱 Model Components (at a glance)
- **Feature-view encoders**: GAT over kNN graphs built from similarity matrices
- **Topology-view encoder**: Bi-GAT, with direction-specific attentions (drug→disease, disease→drug)
- **Fusion**: type-specific node-wise gates for adaptive mixing of views
- **Decoder**: Residual-MoE → MLP(main) + γ · (low-rank bilinear) + node biases

---

## 📜 License
Released under the MIT License — see `LICENSE`.

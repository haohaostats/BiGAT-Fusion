#!/usr/bin/env python3
"""Leakage-safe pair-level and entity cold-start evaluation for BiGAT-Fusion."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch as th
from sklearn import metrics
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from bigat_model import BiGATFusionModel
from data_loader import load_dataset


class TrainingDataset(Dataset):
    """Positive training edges with negatives drawn only from the training universe."""

    def __init__(self, pos_edges, neg_pool, neg_k=3, rng=None):
        if not pos_edges:
            raise ValueError("The training domain contains no positive edges.")
        if not neg_pool:
            raise ValueError("The training domain contains no unknown pairs for negative sampling.")
        self.pos = pos_edges
        self.neg_pool = neg_pool
        self.neg_k = neg_k
        self.rng = random.Random(0) if rng is None else rng

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        d, p = self.pos[idx]
        negatives = [(*self.rng.choice(self.neg_pool), 0) for _ in range(self.neg_k)]
        return (d, p, 1), negatives


def collate(batch):
    d_idx, p_idx, labels = [], [], []
    for positive, negatives in batch:
        for d, p, y in (positive, *negatives):
            d_idx.append(d)
            p_idx.append(p)
            labels.append(y)
    return (
        th.tensor(d_idx, dtype=th.long),
        th.tensor(p_idx, dtype=th.long),
        th.tensor(labels, dtype=th.float32),
    )


@th.no_grad()
def evaluate(model, edges, pos_set, device, batch_size, group_axis=None):
    """Use full-fold pair metrics or per-entity macro metrics for cold-start."""
    model.eval()
    drug_z, disease_z = model.get_fused_embeddings()
    scores = []
    for start in range(0, len(edges), batch_size):
        batch = edges[start:start + batch_size]
        d_idx = th.tensor([e[0] for e in batch], device=device)
        p_idx = th.tensor([e[1] for e in batch], device=device)
        logits = model.decoder(
            drug_z[d_idx], disease_z[p_idx], d_idx, p_idx, model.bias_d, model.bias_p
        )
        scores.append(th.sigmoid(logits).cpu().numpy())

    y_score = np.concatenate(scores)
    y_true = np.fromiter((1 if e in pos_set else 0 for e in edges), dtype=np.int8)
    if np.unique(y_true).size != 2:
        raise ValueError("Evaluation domain must contain both positive and unknown pairs.")

    if group_axis is None:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        return {
            "AUROC": metrics.auc(fpr, tpr),
            "AUPRC": metrics.auc(recall, precision),
            "EligibleEntities": 0,
        }

    groups = np.fromiter((e[group_axis] for e in edges), dtype=np.int64)
    entity_aurocs, entity_aps = [], []
    for entity in np.unique(groups):
        mask = groups == entity
        entity_y = y_true[mask]
        entity_score = y_score[mask]
        if entity_y.any():
            entity_aps.append(metrics.average_precision_score(entity_y, entity_score))
        if np.unique(entity_y).size == 2:
            entity_aurocs.append(metrics.roc_auc_score(entity_y, entity_score))
    if not entity_aps or not entity_aurocs:
        raise ValueError("No eligible entities were available for macro evaluation.")
    macro_auroc = float(np.mean(entity_aurocs))
    macro_ap = float(np.mean(entity_aps))
    return {
        "AUROC": macro_auroc,
        "AUPRC": macro_ap,
        "EligibleEntities": len(entity_aps),
    }


def build_folds(items, k, seed):
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i::k] for i in range(k)]


def build_domains(protocol, n_drugs, n_diseases, folds, fold_id):
    """Return disjoint training, validation, and test candidate universes."""
    val_id = (fold_id + 1) % len(folds)

    if protocol == "pair":
        test_domain = folds[fold_id]
        val_domain = folds[val_id]
        train_domain = [
            pair for i, fold in enumerate(folds) if i not in (fold_id, val_id) for pair in fold
        ]
    elif protocol == "drug_cold":
        test_drugs = folds[fold_id]
        val_drugs = folds[val_id]
        train_drugs = [d for i, fold in enumerate(folds) if i not in (fold_id, val_id) for d in fold]
        train_domain = [(d, p) for d in train_drugs for p in range(n_diseases)]
        val_domain = [(d, p) for d in val_drugs for p in range(n_diseases)]
        test_domain = [(d, p) for d in test_drugs for p in range(n_diseases)]
    elif protocol == "disease_cold":
        test_diseases = folds[fold_id]
        val_diseases = folds[val_id]
        train_diseases = [p for i, fold in enumerate(folds) if i not in (fold_id, val_id) for p in fold]
        train_domain = [(d, p) for d in range(n_drugs) for p in train_diseases]
        val_domain = [(d, p) for d in range(n_drugs) for p in val_diseases]
        test_domain = [(d, p) for d in range(n_drugs) for p in test_diseases]
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    return train_domain, val_domain, test_domain, val_id


def select_device(requested):
    if requested == "auto":
        if hasattr(th, "xpu") and th.xpu.is_available():
            return th.device("xpu")
        if th.cuda.is_available():
            return th.device("cuda")
        return th.device("cpu")
    if requested == "xpu" and not (hasattr(th, "xpu") and th.xpu.is_available()):
        raise RuntimeError(
            "Intel GPU (XPU) was requested but is unavailable. Install the XPU build "
            "of PyTorch and update the Intel GPU driver."
        )
    if requested == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return th.device(requested)


def make_model(data, drug_adj, disease_adj, args, device, protocol=None):
    drug_topology_dropout = args.cold_topology_dropout if protocol == "drug_cold" else 0.0
    disease_topology_dropout = (
        args.cold_topology_dropout if protocol == "disease_cold" else 0.0
    )
    return BiGATFusionModel(
        n_drugs=data["n_drugs"],
        n_diseases=data["n_diseases"],
        drug_feat_neighbors=data["drug_feat_neighbors"],
        disease_feat_neighbors=data["disease_feat_neighbors"],
        drug_neighbors=drug_adj,
        disease_neighbors=disease_adj,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        embedding_init=args.embedding_init,
        fusion_gate_bias=args.fusion_gate_bias,
        drug_topology_dropout=drug_topology_dropout,
        disease_topology_dropout=disease_topology_dropout,
    ).to(device)


def save_interpretation(model, path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with th.no_grad():
        model.eval()
        drug_init = model.drug_emb.weight
        disease_init = model.disease_emb.weight
        drug_feat = model.gat_drug_feat(drug_init, model.drug_feat_src, model.drug_feat_dst)
        disease_feat = model.gat_dis_feat(
            disease_init, model.disease_feat_src, model.disease_feat_dst
        )
        drug_topo, disease_topo = model.bipartite_gat(
            drug_init,
            disease_init,
            model.dis_to_drug_src,
            model.dis_to_drug_dst,
            model.drug_to_dis_src,
            model.drug_to_dis_dst,
        )
        gate_d = th.sigmoid(model.gate_drug(th.cat([drug_feat, drug_topo], 1))).squeeze(1)
        gate_p = th.sigmoid(model.gate_dis(th.cat([disease_feat, disease_topo], 1))).squeeze(1)

        np.savez_compressed(
            path,
            g_d=gate_d.cpu().numpy(),
            g_p=gate_p.cpu().numpy(),
            alpha_d=model.bipartite_gat.last_alpha_d.cpu().numpy(),
            src_d=model.dis_to_drug_src.cpu().numpy(),
            dst_d=model.dis_to_drug_dst.cpu().numpy(),
            alpha_p=model.bipartite_gat.last_alpha_p.cpu().numpy(),
            src_p=model.drug_to_dis_src.cpu().numpy(),
            dst_p=model.drug_to_dis_dst.cpu().numpy(),
            meta=json.dumps(metadata),
        )


def run_protocol(args, protocol, data, device):
    n_drugs, n_diseases = data["n_drugs"], data["n_diseases"]
    pos_set = data["assoc_pos_set"]
    dataset_name = Path(args.mat_path).stem
    repeat_count = args.repeats if protocol == "pair" else args.cold_repeats
    logs = []
    group_axis = 0 if protocol == "drug_cold" else 1 if protocol == "disease_cold" else None

    for repeat in range(repeat_count):
        split_seed = args.seed + repeat
        if protocol == "pair":
            items = [(d, p) for d in range(n_drugs) for p in range(n_diseases)]
        elif protocol == "drug_cold":
            items = list(range(n_drugs))
        else:
            items = list(range(n_diseases))
        folds = build_folds(items, args.folds, split_seed)

        fold_ids = args.fold_ids if args.fold_ids is not None else range(args.folds)
        for fold_id in fold_ids:
            train_domain, val_domain, test_domain, val_id = build_domains(
                protocol, n_drugs, n_diseases, folds, fold_id
            )
            train_pos = [e for e in train_domain if e in pos_set]
            train_neg_pool = [e for e in train_domain if e not in pos_set]

            drug_adj = {d: [] for d in range(n_drugs)}
            disease_adj = {p: [] for p in range(n_diseases)}
            for d, p in train_pos:
                drug_adj[d].append(p)
                disease_adj[p].append(d)

            model = make_model(data, drug_adj, disease_adj, args, device, protocol)
            gate_params, backbone_params = [], []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                (gate_params if "gate_" in name else backbone_params).append(parameter)

            parameter_groups = [{"params": backbone_params, "weight_decay": args.wd_backbone}]
            if gate_params:
                parameter_groups.append({"params": gate_params, "weight_decay": args.wd_gate})
            optimizer = Adam(parameter_groups, lr=args.lr)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=args.lr_patience,
                threshold=1e-4,
                min_lr=1e-5,
            )
            criterion = nn.BCEWithLogitsLoss()
            loader = DataLoader(
                TrainingDataset(
                    train_pos,
                    train_neg_pool,
                    args.neg_k,
                    random.Random(args.seed + repeat * 1000 + fold_id),
                ),
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate,
                num_workers=0,
            )

            best_val, best_state, stale_evaluations = -1.0, None, 0
            for epoch in range(1, args.epochs + 1):
                model.train()
                for d_idx, p_idx, labels in loader:
                    d_idx, p_idx, labels = d_idx.to(device), p_idx.to(device), labels.to(device)
                    loss = criterion(model.logits_on_pairs(d_idx, p_idx), labels)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

                if epoch % args.eval_every == 0 or epoch == args.epochs:
                    val_metrics = evaluate(
                        model, val_domain, pos_set, device, args.eval_batch_size, group_axis
                    )
                    val_auc = val_metrics["AUROC"]
                    scheduler.step(val_auc)
                    if val_auc > best_val + 1e-6:
                        best_val = val_auc
                        best_state = {
                            name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()
                        }
                        stale_evaluations = 0
                    else:
                        stale_evaluations += 1
                    if args.early_stop and stale_evaluations >= args.es_patience:
                        break

            if best_state is None:
                raise RuntimeError("No validation checkpoint was produced.")
            model.load_state_dict(best_state)
            test_metrics = evaluate(
                model, test_domain, pos_set, device, args.eval_batch_size, group_axis
            )
            auroc, auprc = test_metrics["AUROC"], test_metrics["AUPRC"]
            record = {
                "Protocol": protocol,
                "Repeat": repeat,
                "Fold": fold_id,
                "ValFold": val_id,
                "TrainPairs": len(train_domain),
                "ValPairs": len(val_domain),
                "TestPairs": len(test_domain),
                "TrainPos": len(train_pos),
                "TestPos": sum(e in pos_set for e in test_domain),
                "BestValAUROC": best_val,
                "AUROC": auroc,
                "AUPRC": auprc,
                "EligibleEntities": test_metrics["EligibleEntities"],
            }
            logs.append(record)

            checkpoint_dir = Path("train")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
            stem = f"{protocol}_{dataset_name}_r{repeat}_f{fold_id}{tag_suffix}"
            th.save(model.state_dict(), checkpoint_dir / f"{stem}.pt")
            save_interpretation(
                model,
                Path("interpret")
                / "folds"
                / dataset_name
                / protocol
                / f"r{repeat}_f{fold_id}{tag_suffix}.npz",
                {
                    "dataset": dataset_name,
                    "protocol": protocol,
                    "repeat": repeat,
                    "fold": fold_id,
                    "val_fold": val_id,
                },
            )
            print(
                f"[{dataset_name}|{protocol}] R{repeat + 1}/{repeat_count} "
                f"F{fold_id + 1}/{args.folds} AUROC={auroc:.4f} AUPRC={auprc:.4f}"
            )

    result_dir = Path("results") / dataset_name
    result_dir.mkdir(parents=True, exist_ok=True)
    fold_suffix = (
        "" if args.fold_ids is None else "_folds_" + "-".join(map(str, args.fold_ids))
    )
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    csv_path = result_dir / f"cv_log_{protocol}{tag_suffix}{fold_suffix}.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=logs[0].keys())
        writer.writeheader()
        writer.writerows(logs)

    auroc_values = np.array([row["AUROC"] for row in logs])
    auprc_values = np.array([row["AUPRC"] for row in logs])
    print(
        f"[{dataset_name}|{protocol}] AUROC={auroc_values.mean():.4f}+/-{auroc_values.std():.4f} "
        f"AUPRC={auprc_values.mean():.4f}+/-{auprc_values.std():.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_path", default="data/Gdataset/Gdataset.mat")
    parser.add_argument(
        "--protocol",
        choices=["pair", "drug_cold", "disease_cold", "all"],
        default="pair",
    )
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=65536)
    parser.add_argument("--neg_k", type=int, default=3)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument(
        "--fold_ids",
        type=lambda value: [int(item) for item in value.split(",")],
        help="optional comma-separated zero-based folds, for example 0 or 0,1",
    )
    parser.add_argument("--repeats", type=int, default=10, help="pair-level repetitions")
    parser.add_argument("--cold_repeats", type=int, default=1)
    parser.add_argument("--run_tag", default="", help="label used to keep outputs separate")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--embedding_init",
        choices=["pytorch", "scaled_normal", "xavier"],
        default="pytorch",
    )
    parser.add_argument("--fusion_gate_bias", type=float, default=0.0)
    parser.add_argument("--cold_topology_dropout", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    parser.add_argument("--wd_backbone", type=float, default=1e-4)
    parser.add_argument("--wd_gate", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--lr_patience", type=int, default=20)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--es_patience", type=int, default=40)
    args = parser.parse_args()

    if args.fold_ids is not None and any(i < 0 or i >= args.folds for i in args.fold_ids):
        parser.error("every --fold_ids value must be between 0 and --folds-1")
    if not 0.0 <= args.cold_topology_dropout < 1.0:
        parser.error("--cold_topology_dropout must be in [0, 1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

    device = select_device(args.device)
    print(f"Using device: {device}")
    data = load_dataset(args.mat_path, k=args.k, val_ratio=0.0, test_ratio=0.0)
    protocols = (
        ["pair", "drug_cold", "disease_cold"] if args.protocol == "all" else [args.protocol]
    )
    for protocol in protocols:
        run_protocol(args, protocol, data, device)


if __name__ == "__main__":
    main()

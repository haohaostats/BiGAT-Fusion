import csv
import json
from pathlib import Path

import numpy as np
import torch


def run_suffix(run_tag):
    return f"_{run_tag}" if run_tag else ""


def save_checkpoint(model, dataset_name, protocol, repeat, fold, run_tag=""):

    checkpoint_dir = Path("train")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{protocol}_{dataset_name}_r{repeat}_f{fold}{run_suffix(run_tag)}"
    torch.save(model.state_dict(), checkpoint_dir / f"{stem}.pt")


def save_interpretation(model, path, metadata):

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.eval()
        drug_initial = model.drug_emb.weight
        disease_initial = model.disease_emb.weight
        drug_feature = model.gat_drug_feat(
            drug_initial, model.drug_feat_src, model.drug_feat_dst
        )
        disease_feature = model.gat_dis_feat(
            disease_initial, model.disease_feat_src, model.disease_feat_dst
        )
        drug_topology, disease_topology = model.bipartite_gat(
            drug_initial,
            disease_initial,
            model.dis_to_drug_src,
            model.dis_to_drug_dst,
            model.drug_to_dis_src,
            model.drug_to_dis_dst,
        )
        drug_gate = torch.sigmoid(
            model.gate_drug(torch.cat([drug_feature, drug_topology], dim=1))
        ).squeeze(1)
        disease_gate = torch.sigmoid(
            model.gate_dis(torch.cat([disease_feature, disease_topology], dim=1))
        ).squeeze(1)
        np.savez_compressed(
            path,
            g_d=drug_gate.cpu().numpy(),
            g_p=disease_gate.cpu().numpy(),
            alpha_d=model.bipartite_gat.last_alpha_d.cpu().numpy(),
            src_d=model.dis_to_drug_src.cpu().numpy(),
            dst_d=model.dis_to_drug_dst.cpu().numpy(),
            alpha_p=model.bipartite_gat.last_alpha_p.cpu().numpy(),
            src_p=model.drug_to_dis_src.cpu().numpy(),
            dst_p=model.drug_to_dis_dst.cpu().numpy(),
            meta=json.dumps(metadata),
        )


def save_fold_outputs(model, record, dataset_name, protocol, repeat, fold, run_tag=""):

    save_checkpoint(model, dataset_name, protocol, repeat, fold, run_tag)
    save_interpretation(
        model,
        Path("interpret")
        / "folds"
        / dataset_name
        / protocol
        / f"r{repeat}_f{fold}{run_suffix(run_tag)}.npz",
        {
            "dataset": dataset_name,
            "protocol": protocol,
            "repeat": repeat,
            "fold": fold,
            "val_fold": record["ValFold"],
        },
    )


def save_metric_log(records, dataset_name, protocol, run_tag="", fold_ids=None):

    result_dir = Path("results") / dataset_name
    result_dir.mkdir(parents=True, exist_ok=True)
    fold_suffix = "" if fold_ids is None else "_folds_" + "-".join(map(str, fold_ids))
    path = result_dir / f"cv_log_{protocol}{run_suffix(run_tag)}{fold_suffix}.csv"
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    return path

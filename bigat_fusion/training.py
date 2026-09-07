"""Model construction and cross-validation training."""

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .artifacts import save_fold_outputs, save_metric_log
from .metrics import evaluate
from .model import BiGATFusionModel
from .protocols import build_domains, build_folds, candidate_items
from .sampling import GlobalUnknownTrainingDataset, collate_training_pairs


def topology_neighbors(n_drugs, n_diseases, positive_edges):
    """Build bipartite adjacency maps from training associations."""
    drug_neighbors = {drug: [] for drug in range(n_drugs)}
    disease_neighbors = {disease: [] for disease in range(n_diseases)}
    for drug, disease in positive_edges:
        drug_neighbors[drug].append(disease)
        disease_neighbors[disease].append(drug)
    return drug_neighbors, disease_neighbors


def build_model(data, positive_edges, args, device):
    """Construct a model and its fold-specific topology graph."""
    drug_neighbors, disease_neighbors = topology_neighbors(
        data["n_drugs"], data["n_diseases"], positive_edges
    )
    return BiGATFusionModel(
        n_drugs=data["n_drugs"],
        n_diseases=data["n_diseases"],
        drug_feat_neighbors=data["drug_feat_neighbors"],
        disease_feat_neighbors=data["disease_feat_neighbors"],
        drug_neighbors=drug_neighbors,
        disease_neighbors=disease_neighbors,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)


def build_optimizer(model, args):
    """Create parameter groups and the learning-rate scheduler."""
    gate_parameters, backbone_parameters = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            target = gate_parameters if "gate_" in name else backbone_parameters
            target.append(parameter)
    parameter_groups = [
        {"params": backbone_parameters, "weight_decay": args.wd_backbone}
    ]
    if gate_parameters:
        parameter_groups.append(
            {"params": gate_parameters, "weight_decay": args.wd_gate}
        )
    optimizer = Adam(parameter_groups, lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.lr_patience,
        threshold=1e-4,
        min_lr=1e-5,
    )
    return optimizer, scheduler


def build_training_loader(positive_edges, positive_set, n_drugs, n_diseases,
                          args, repeat, fold):
    """Sample uniformly from the complete unknown-pair universe."""
    dataset = GlobalUnknownTrainingDataset(
        positive_edges, n_drugs, n_diseases, positive_set, args.neg_k,
        random.Random(args.seed + repeat * 1000 + fold),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_training_pairs,
        num_workers=0,
    )


def fit_fold(model, loader, validation_domain, positive_set, device, args):
    """Train one fold and restore the checkpoint selected on validation AUPRC."""
    optimizer, scheduler = build_optimizer(model, args)
    criterion = nn.BCEWithLogitsLoss()
    best_validation, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for drug_idx, disease_idx, labels in loader:
            drug_idx = drug_idx.to(device)
            disease_idx = disease_idx.to(device)
            labels = labels.to(device)
            loss = criterion(model.logits_on_pairs(drug_idx, disease_idx), labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            validation_metrics = evaluate(
                model,
                validation_domain,
                positive_set,
                device,
                args.eval_batch_size,
            )
            scheduler.step(validation_metrics["AUROC"])
            validation_auprc = validation_metrics["AUPRC"]
            if validation_auprc > best_validation + 1e-6:
                best_validation = validation_auprc
                selected_validation = validation_metrics.copy()
                selected_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            if epoch % 200 == 0 or epoch == args.epochs:
                print(f"Epoch {epoch}: validation AUPRC={validation_auprc:.4f}, "
                      f"AUROC={validation_metrics['AUROC']:.4f}", flush=True)

    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced.")
    model.load_state_dict(best_state)
    return {
        "metric": "AUPRC",
        "epoch": selected_epoch,
        "AUPRC": selected_validation["AUPRC"],
        "AUROC": selected_validation["AUROC"],
    }


def fold_record(
    protocol,
    repeat,
    fold,
    validation_fold,
    training_domain,
    validation_domain,
    test_domain,
    training_positive,
    positive_set,
    selection,
    test_metrics,
):
    """Assemble the metric row for one completed fold."""
    return {
        "Protocol": protocol,
        "Repeat": repeat,
        "Fold": fold,
        "ValFold": validation_fold,
        "TrainPairs": len(training_domain),
        "ValPairs": len(validation_domain),
        "TestPairs": len(test_domain),
        "TrainPos": len(training_positive),
        "TestPos": sum(edge in positive_set for edge in test_domain),
        "SelectionMetric": selection["metric"],
        "BestEpoch": selection["epoch"],
        "BestValAUPRC": selection["AUPRC"],
        "SelectedValAUROC": selection["AUROC"],
        "AUROC": test_metrics["AUROC"],
        "AUPRC": test_metrics["AUPRC"],
    }


def run_protocol(args, data, device):
    """Run every requested repetition and fold for one protocol."""
    n_drugs, n_diseases = data["n_drugs"], data["n_diseases"]
    positive_set = data["assoc_pos_set"]
    dataset_name = Path(args.mat_path).stem
    protocol = "pair"
    repeat_count = args.repeats
    records = []

    pair_split_rng = random.Random(args.seed)
    for repeat in range(repeat_count):
        items = candidate_items(n_drugs, n_diseases)
        split_seed = pair_split_rng.randint(0, 1 << 30)
        folds = build_folds(items, args.folds, split_seed)
        selected_folds = args.fold_ids if args.fold_ids is not None else range(args.folds)
        for fold in selected_folds:
            training_domain, validation_domain, test_domain, validation_fold = build_domains(
                folds, fold
            )
            training_positive = [
                edge for edge in training_domain if edge in positive_set
            ]
            model = build_model(data, training_positive, args, device)
            loader = build_training_loader(
                training_positive, positive_set, n_drugs, n_diseases,
                args, repeat, fold
            )
            selection = fit_fold(
                model,
                loader,
                validation_domain,
                positive_set,
                device,
                args,
            )
            test_metrics = evaluate(
                model,
                test_domain,
                positive_set,
                device,
                args.eval_batch_size,
            )
            record = fold_record(
                protocol,
                repeat,
                fold,
                validation_fold,
                training_domain,
                validation_domain,
                test_domain,
                training_positive,
                positive_set,
                selection,
                test_metrics,
            )
            records.append(record)
            save_fold_outputs(
                model, record, dataset_name, protocol, repeat, fold, args.run_tag
            )
            print(
                f"[{dataset_name}|{protocol}] R{repeat + 1}/{repeat_count} "
                f"F{fold + 1}/{args.folds} AUROC={record['AUROC']:.4f} "
                f"AUPRC={record['AUPRC']:.4f}"
            )

    save_metric_log(records, dataset_name, protocol, args.run_tag, args.fold_ids)
    auroc_values = np.array([record["AUROC"] for record in records])
    auprc_values = np.array([record["AUPRC"] for record in records])
    print(
        f"[{dataset_name}|{protocol}] AUROC={auroc_values.mean():.4f}+/-"
        f"{auroc_values.std():.4f} AUPRC={auprc_values.mean():.4f}+/-"
        f"{auprc_values.std():.4f}"
    )

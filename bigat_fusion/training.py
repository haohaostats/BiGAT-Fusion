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
from .protocols import build_domains, build_folds, candidate_items, metric_group_axis
from .sampling import TrainingDataset, collate_training_pairs


def topology_neighbors(n_drugs, n_diseases, positive_edges):
    """Build bipartite adjacency maps from training associations."""
    drug_neighbors = {drug: [] for drug in range(n_drugs)}
    disease_neighbors = {disease: [] for disease in range(n_diseases)}
    for drug, disease in positive_edges:
        drug_neighbors[drug].append(disease)
        disease_neighbors[disease].append(drug)
    return drug_neighbors, disease_neighbors


def build_model(data, positive_edges, args, device, protocol):
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
        embedding_init=args.embedding_init,
        fusion_gate_bias=args.fusion_gate_bias,
        drug_topology_dropout=(
            args.cold_topology_dropout if protocol == "drug_cold" else 0.0
        ),
        disease_topology_dropout=(
            args.cold_topology_dropout if protocol == "disease_cold" else 0.0
        ),
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


def build_training_loader(positive_edges, negative_pool, args, repeat, fold):
    """Create the sampled training loader for one fold."""
    dataset = TrainingDataset(
        positive_edges,
        negative_pool,
        args.neg_k,
        random.Random(args.seed + repeat * 1000 + fold),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_training_pairs,
        num_workers=0,
    )


def fit_fold(model, loader, validation_domain, positive_set, device, args, group_axis):
    """Train one fold and restore the checkpoint selected on validation AUROC."""
    optimizer, scheduler = build_optimizer(model, args)
    criterion = nn.BCEWithLogitsLoss()
    best_validation, best_state, stale_evaluations = -1.0, None, 0

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
                group_axis,
            )
            validation_auroc = validation_metrics["AUROC"]
            scheduler.step(validation_auroc)
            if validation_auroc > best_validation + 1e-6:
                best_validation = validation_auroc
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
    return best_validation


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
    best_validation,
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
        "BestValAUROC": best_validation,
        "AUROC": test_metrics["AUROC"],
        "AUPRC": test_metrics["AUPRC"],
        "EligibleEntities": test_metrics["EligibleEntities"],
    }


def run_protocol(args, protocol, data, device):
    """Run every requested repetition and fold for one protocol."""
    n_drugs, n_diseases = data["n_drugs"], data["n_diseases"]
    positive_set = data["assoc_pos_set"]
    dataset_name = Path(args.mat_path).stem
    repeat_count = args.repeats if protocol == "pair" else args.cold_repeats
    group_axis = metric_group_axis(protocol)
    records = []

    for repeat in range(repeat_count):
        items = candidate_items(protocol, n_drugs, n_diseases)
        folds = build_folds(items, args.folds, args.seed + repeat)
        selected_folds = args.fold_ids if args.fold_ids is not None else range(args.folds)
        for fold in selected_folds:
            training_domain, validation_domain, test_domain, validation_fold = build_domains(
                protocol, n_drugs, n_diseases, folds, fold
            )
            training_positive = [
                edge for edge in training_domain if edge in positive_set
            ]
            negative_pool = [
                edge for edge in training_domain if edge not in positive_set
            ]
            model = build_model(data, training_positive, args, device, protocol)
            loader = build_training_loader(
                training_positive, negative_pool, args, repeat, fold
            )
            best_validation = fit_fold(
                model,
                loader,
                validation_domain,
                positive_set,
                device,
                args,
                group_axis,
            )
            test_metrics = evaluate(
                model,
                test_domain,
                positive_set,
                device,
                args.eval_batch_size,
                group_axis,
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
                best_validation,
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

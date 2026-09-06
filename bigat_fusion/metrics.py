"""Prediction and metric calculation."""

import numpy as np
import torch
from sklearn import metrics


@torch.no_grad()
def predict_pairs(model, edges, device, batch_size):
    """Predict continuous association scores for candidate pairs."""
    model.eval()
    drug_z, disease_z = model.get_fused_embeddings()
    scores = []
    for start in range(0, len(edges), batch_size):
        batch = edges[start:start + batch_size]
        drug_idx = torch.tensor([edge[0] for edge in batch], device=device)
        disease_idx = torch.tensor([edge[1] for edge in batch], device=device)
        logits = model.decoder(
            drug_z[drug_idx],
            disease_z[disease_idx],
            drug_idx,
            disease_idx,
            model.bias_d,
            model.bias_p,
        )
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def pair_metrics(labels, scores):
    """Calculate AUROC and area under the precision-recall curve."""
    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(labels, scores)
    precision, recall, _ = metrics.precision_recall_curve(labels, scores)
    return {
        "AUROC": metrics.auc(false_positive_rate, true_positive_rate),
        "AUPRC": metrics.auc(recall, precision),
        "EligibleEntities": 0,
    }


def macro_entity_metrics(labels, scores, groups):
    """Calculate per-entity AUROC and AP, then macro-average them."""
    entity_aurocs, entity_average_precisions = [], []
    for entity in np.unique(groups):
        mask = groups == entity
        entity_labels = labels[mask]
        entity_scores = scores[mask]
        if entity_labels.any():
            entity_average_precisions.append(
                metrics.average_precision_score(entity_labels, entity_scores)
            )
        if np.unique(entity_labels).size == 2:
            entity_aurocs.append(metrics.roc_auc_score(entity_labels, entity_scores))
    if not entity_average_precisions or not entity_aurocs:
        raise ValueError("No eligible entities were available for macro evaluation.")
    return {
        "AUROC": float(np.mean(entity_aurocs)),
        "AUPRC": float(np.mean(entity_average_precisions)),
        "EligibleEntities": len(entity_average_precisions),
    }


def evaluate(model, edges, positive_set, device, batch_size, group_axis=None):
    """Evaluate a pair domain with pair-level or entity-macro metrics."""
    scores = predict_pairs(model, edges, device, batch_size)
    labels = np.fromiter(
        (1 if edge in positive_set else 0 for edge in edges), dtype=np.int8
    )
    if np.unique(labels).size != 2:
        raise ValueError("Evaluation domain must contain both positive and unknown pairs.")
    if group_axis is None:
        return pair_metrics(labels, scores)
    groups = np.fromiter((edge[group_axis] for edge in edges), dtype=np.int64)
    return macro_entity_metrics(labels, scores, groups)

import numpy as np
import torch
from sklearn import metrics


@torch.no_grad()
def predict_pairs(model, edges, device, batch_size):

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

    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(labels, scores)
    precision, recall, _ = metrics.precision_recall_curve(labels, scores)
    return {
        "AUROC": metrics.auc(false_positive_rate, true_positive_rate),
        "AUPRC": metrics.auc(recall, precision),
    }


def evaluate(model, edges, positive_set, device, batch_size):

    scores = predict_pairs(model, edges, device, batch_size)
    labels = np.fromiter(
        (1 if edge in positive_set else 0 for edge in edges), dtype=np.int8
    )
    if np.unique(labels).size != 2:
        raise ValueError("Evaluation domain must contain both positive and unknown pairs.")
    return pair_metrics(labels, scores)

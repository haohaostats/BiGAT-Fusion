"""Cross-validation partitions for pair-level and cold-start evaluation."""

import random


PROTOCOLS = ("pair", "drug_cold", "disease_cold")


def build_folds(items, fold_count, seed):
    """Shuffle items reproducibly and distribute them across folds."""
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[index::fold_count] for index in range(fold_count)]


def candidate_items(protocol, n_drugs, n_diseases):
    """Return the objects partitioned by a protocol."""
    if protocol == "pair":
        return [(drug, disease) for drug in range(n_drugs) for disease in range(n_diseases)]
    if protocol == "drug_cold":
        return list(range(n_drugs))
    if protocol == "disease_cold":
        return list(range(n_diseases))
    raise ValueError(f"Unknown protocol: {protocol}")


def build_domains(protocol, n_drugs, n_diseases, folds, test_fold):
    """Create disjoint training, validation, and test candidate domains."""
    validation_fold = (test_fold + 1) % len(folds)
    training_fold_ids = [
        index for index in range(len(folds)) if index not in (test_fold, validation_fold)
    ]

    if protocol == "pair":
        training = [pair for index in training_fold_ids for pair in folds[index]]
        validation = folds[validation_fold]
        test = folds[test_fold]
    elif protocol == "drug_cold":
        training_drugs = [drug for index in training_fold_ids for drug in folds[index]]
        validation_drugs = folds[validation_fold]
        test_drugs = folds[test_fold]
        training = [(drug, disease) for drug in training_drugs for disease in range(n_diseases)]
        validation = [
            (drug, disease) for drug in validation_drugs for disease in range(n_diseases)
        ]
        test = [(drug, disease) for drug in test_drugs for disease in range(n_diseases)]
    elif protocol == "disease_cold":
        training_diseases = [
            disease for index in training_fold_ids for disease in folds[index]
        ]
        validation_diseases = folds[validation_fold]
        test_diseases = folds[test_fold]
        training = [(drug, disease) for drug in range(n_drugs) for disease in training_diseases]
        validation = [
            (drug, disease) for drug in range(n_drugs) for disease in validation_diseases
        ]
        test = [(drug, disease) for drug in range(n_drugs) for disease in test_diseases]
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    return training, validation, test, validation_fold


def metric_group_axis(protocol):
    """Map a cold-start protocol to its held-out entity axis."""
    if protocol == "drug_cold":
        return 0
    if protocol == "disease_cold":
        return 1
    return None

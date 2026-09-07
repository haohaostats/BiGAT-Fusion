import random


def build_folds(items, fold_count, seed):

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[index::fold_count] for index in range(fold_count)]


def candidate_items(n_drugs, n_diseases):
    return [(drug, disease) for drug in range(n_drugs) for disease in range(n_diseases)]


def build_domains(folds, test_fold):

    validation_fold = (test_fold + 1) % len(folds)
    training = [
        pair for index in range(len(folds))
        if index not in (test_fold, validation_fold) for pair in folds[index]
    ]
    return training, folds[validation_fold], folds[test_fold], validation_fold

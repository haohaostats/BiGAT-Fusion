import random
import unittest

from bigat_fusion.protocols import build_domains, build_folds, candidate_items
from bigat_fusion.sampling import GlobalUnknownTrainingDataset


class ProtocolTests(unittest.TestCase):
    def test_pair_domains_partition_complete_universe(self):
        items = candidate_items(7, 5)
        folds = build_folds(items, 5, 19)
        for fold in range(5):
            train, validation, test, val_fold = build_domains(folds, fold)
            domains = [set(domain) for domain in (train, validation, test)]
            self.assertEqual(val_fold, (fold + 1) % 5)
            self.assertEqual(set.union(*domains), set(items))
            self.assertEqual(sum(map(len, domains)), len(items))
            self.assertFalse(domains[0] & domains[1])
            self.assertFalse(domains[0] & domains[2])
            self.assertFalse(domains[1] & domains[2])

    def test_pair_sampler_matches_original_rejection_rule(self):
        positives = [(0, 0), (1, 1)]
        known = {(0, 0), (1, 1), (2, 2)}
        dataset = GlobalUnknownTrainingDataset(
            positives, 3, 3, known, 3, random.Random(2025))
        reference = random.Random(2025)
        for index in range(20):
            _, negatives = dataset[index % len(positives)]
            expected = []
            while len(expected) < 3:
                pair = (reference.randrange(3), reference.randrange(3))
                if pair not in known:
                    expected.append((*pair, 0))
            self.assertEqual(negatives, expected)

if __name__ == "__main__":
    unittest.main()

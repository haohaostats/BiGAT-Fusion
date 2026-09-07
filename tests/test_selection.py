import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from bigat_fusion.training import fit_fold


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def logits_on_pairs(self, drug, disease):
        return self.weight.expand(drug.shape)


class SelectionTests(unittest.TestCase):
    def test_auprc_selects_checkpoint_while_auroc_schedules_lr(self):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = Mock()
        args = SimpleNamespace(epochs=3, eval_every=1, eval_batch_size=16)
        loader = [(torch.tensor([0]), torch.tensor([0]), torch.tensor([1.0]))]
        weights = []
        scores = iter([
            {"AUPRC": 0.2, "AUROC": 0.9},
            {"AUPRC": 0.4, "AUROC": 0.8},
            {"AUPRC": 0.3, "AUROC": 0.95},
        ])

        def evaluation(*unused):
            weights.append(model.weight.detach().clone())
            return next(scores)

        with patch("bigat_fusion.training.build_optimizer", return_value=(optimizer, scheduler)), \
             patch("bigat_fusion.training.evaluate", side_effect=evaluation):
            selected = fit_fold(model, loader, [], set(), "cpu", args)
        self.assertEqual(selected["epoch"], 2)
        self.assertEqual(selected["metric"], "AUPRC")
        self.assertTrue(torch.equal(model.weight, weights[1]))
        self.assertEqual([call.args[0] for call in scheduler.step.call_args_list],
                         [0.9, 0.8, 0.95])


if __name__ == "__main__":
    unittest.main()

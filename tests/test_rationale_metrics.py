import torch

from src.eval.comprehensiveness import comprehensiveness
from src.eval.sufficiency import sufficiency


class DummyOutput:
    def __init__(self, logits):
        self.logits = logits


class DummyModel:
    def eval(self):
        return self

    def __call__(self, **x):
        score = x["attention_mask"].float().sum(dim=1, keepdim=True)
        logits = torch.cat([torch.zeros_like(score), score], dim=1)
        return DummyOutput(logits)


def test_comprehensiveness_positive_when_removing_rationale():
    model = DummyModel()
    x = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }
    comp = comprehensiveness(model, x, rationale_tokens=[0, 1], target_class=1)
    assert comp > 0


def test_sufficiency_non_negative():
    model = DummyModel()
    x = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }
    suff = sufficiency(model, x, rationale_tokens=[0, 1], target_class=1)
    assert suff >= 0

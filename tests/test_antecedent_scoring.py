import pytest
import torch

from decofre.tasks import antecedent_scoring


@pytest.fixture
def scores():
    return torch.tensor(
        [
            [5.0, -2.0, -3.0, -1e32, -1e32],  # Correct (true new)
            [1.0, -4.1, 5.2, 2.713, 0.2],     # Correct (correct link)
            [-0.1, -3.5, -0.3, -2.0, -1e32],  # Wrong (false new)
            [0.0, 1.0, -1e32, -1e32, -1e32],  # Wrong (false link)
            [-10.0, -11.0, -12.0, 13, -14],   # Correct (correct link)
            [10.0, 11.0, 12.0, 13, 14],       # Wrong (wrong link)
        ]
    )


@pytest.fixture
def target():
    return torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=torch.bool,
    )


def test_antecedent_accuracy(scores, target):
    total, mention_new, anaphora = antecedent_scoring.antecedent_accuracy(scores, target)
    assert total.item() == pytest.approx(3 / 6)
    assert mention_new.item() == pytest.approx(1 / 2)
    assert anaphora.item() == pytest.approx(2 / 4)


def test_attributions(scores, target):
    errors = antecedent_scoring.attributions(scores, target)
    assert errors.equal(torch.tensor([1, 1, 2, 1, 1]))
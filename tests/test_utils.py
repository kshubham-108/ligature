import torch

from ligature.utils import set_seed


def test_set_seed_makes_randn_reproducible() -> None:
    set_seed(1234)
    first = torch.randn(4, 8)
    set_seed(1234)
    second = torch.randn(4, 8)
    assert torch.equal(first, second)

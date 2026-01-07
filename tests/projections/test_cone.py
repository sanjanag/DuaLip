import torch

from dualip.projections.base import project


def test_cone_I():
    x = torch.tensor([-0.2, 0.6, 0.1])
    p = project("cone", lower=None, upper=0.5)
    y = p(x)
    expected = torch.tensor([-0.2, 0.5, 0.1])
    assert torch.allclose(y, expected, atol=1e-6), f"Projection mismatch: got {y}, expected {expected}"


def test_cone_II():
    x = torch.tensor([-0.2, 0.6, 0.1])
    p = project("cone", lower=0, upper=None)
    y = p(x)
    expected = torch.tensor([0.0, 0.6, 0.1])
    assert torch.allclose(y, expected, atol=1e-6), f"Projection mismatch: got {y}, expected {expected}"

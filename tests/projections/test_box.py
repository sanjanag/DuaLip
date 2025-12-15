import torch 
from dualip.projections.base import project

def test_box():
    x = torch.tensor([0.2, 0.6, 0.1])
    p = project("box", l=0.25, u=0.3)
    y = p(x)
    assert torch.all(y >= 0.25)
    assert torch.all(y <= 0.3)
    assert torch.isclose(y.sum(), torch.tensor(0.8), atol=1e-6)


from dualip.projections.base import ProjectionOperator, register
import torch

@register("box")
class BoxProjection(ProjectionOperator):
    """
    Projection onto a box [l, u] per-coordinate of x.
    """
    def __init__(self, l: float = 0.0, u: float = 1.0):
        self.l, self.u = l, u

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.l, max=self.u)

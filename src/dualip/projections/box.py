import torch

from dualip.projections.base import ProjectionOperator, register


@register("box")
class BoxProjection(ProjectionOperator):
    """
    Projection onto a box [lower, upper] per-coordinate of x.
    """

    def __init__(self, l: float = 0.0, u: float = 1.0):
        self.lower, self.upper = l, u

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.lower, max=self.upper)

from dualip.projections.base import ProjectionOperator, register
import torch

@register("cone")
class coneProjection(ProjectionOperator):
    """
    Projection onto a cone [l, +inf] or [-inf, u] per-coordinate of x.
    Default behavior is to project onto the cone [0, +inf] or [-inf, 0].
    Only one of l or u should be specified.
    If both are None, the projection is identity.
    """
    def __init__(self, l: float | None = None, u: float | None = None):
        if l is not None and u is not None:
            raise ValueError("Only one of 'l' or 'u' should be specified, not both.")

        self.l, self.u = l, u

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.l is not None:
            return x.clamp(min=self.l)
        elif self.u is not None:
            return x.clamp(max=self.u)
        else:
            # If both l and u are None, return x unchanged
            return x    

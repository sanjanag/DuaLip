import torch

from dualip.projections.base import ProjectionOperator, register


@register("cone")
class coneProjection(ProjectionOperator):
    """
    Projection onto a cone [lower, +inf] or [-inf, upper] per-coordinate of x.
    Default behavior is to project onto the cone [0, +inf] or [-inf, 0].
    Only one of l or u should be specified.
    If both are None, the projection is identity.
    """

    def __init__(self, lower: float | None = None, upper: float | None = None):
        if lower is not None and upper is not None:
            raise ValueError("Only one of 'lower' or 'upper' should be specified, not both.")

        self.lower, self.upper = lower, upper

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.lower is not None:
            return x.clamp(min=self.lower)
        elif self.upper is not None:
            return x.clamp(max=self.upper)
        else:
            # If both lower and upper are None, return x unchanged
            return x

from abc import ABC, abstractmethod
import torch
from typing import Dict, List, Union
from dataclasses import dataclass, field

@dataclass
class ProjectionEntry:
    proj_type: str = ""
    proj_params: dict[str, float] = field(default_factory=dict)
    indices: list[int] = field(default_factory=list)

class ProjectionOperator(ABC):
    """
    Base class for all projection operators.
    Subclasses must precompute whatever they need in __init__,
    then apply projection in __call__.
    """
    @abstractmethod
    def __init__(self, **params):
        """
        Build any precomputed state (e.g. projection parameters or buckets).
        Params are operator-specific.
        """
        pass 

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project x into the desired set and return the result.
        Must not modify x in-place.
        """
        pass


# A registry to look up projections by name
_registry = {}

def register(name):
    def decorator(cls):
        _registry[name] = cls
        return cls
    return decorator

def project(name: str, **params) -> ProjectionOperator:
    """
    Instantiate a projection operator by name.
    """
    if name not in _registry:
        raise ValueError(f"Unknown projection operator '{name}'")
    return _registry[name](**params)

def create_projection_map(
    proj_type: str,
    proj_params: Dict[str, float],
    num_indices: int,
    indices: Union[List[int], None] = None,
    key_prefix: str = ""
) -> Dict[str, ProjectionEntry]:
    """
    Create a projection map with default behavior for constant projection types.
    
    Args:
        proj_type: Type of projection (e.g., "simplex_ineq", "box")
        proj_params: Parameters for the projection (e.g., {"z": 1.0})
        num_indices: Total number of indices to apply projection to
        indices: Specific indices to apply projection to. If None, applies to all indices [0, num_indices)
        key_prefix: Optional prefix for the projection map key
        
    Returns:
        Dictionary mapping projection keys to ProjectionEntry objects
        
    Example:
        # Apply simplex_ineq projection to all 5 indices
        projection_map = create_projection_map("simplex_ineq", {"z": 1.0}, 5)
        
        # Apply box projection to specific indices
        projection_map = create_projection_map("box", {"l": 0.0, "u": 1.0}, 10, indices=[0, 2, 4, 6, 8])
    """
    if indices is None:
        indices = list(range(num_indices))
    
    # Create a unique key for this projection
    param_str = "_".join(f"{k}_{v}" for k, v in sorted(proj_params.items()))
    key = f"{key_prefix}{proj_type}_{param_str}" if key_prefix else f"{proj_type}_{param_str}"
    
    projection_map = {}
    projection_map[key] = ProjectionEntry(
        proj_type=proj_type,
        proj_params=proj_params,
        indices=indices
    )
    
    return projection_map

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch


@dataclass
class SolverArgs:
    max_iter: int
    initial_step_size: float
    gamma: float
    max_step_size: float = 0.1
    initial_dual_path: Optional[str] = None
    gamma_decay_type: Optional[Literal["step"]] = None
    gamma_decay_params: Optional[dict] = None
    save_primal: bool = False


@dataclass
class ComputeArgs:
    host_device: str
    compute_device_num: Optional[int] = None


@dataclass
class ObjectiveArgs:
    objective_type: Literal["miplib2017", "matching"]
    use_jacobi_precondition: bool = False
    objective_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class ObjectiveResult:
    dual_gradient: torch.Tensor
    dual_objective: torch.Tensor
    reg_penalty: torch.Tensor = None
    primal_objective: torch.Tensor = None
    primal_var: torch.Tensor = None
    dual_val_times_grad: torch.Tensor = None
    max_pos_slack: torch.Tensor = None
    sum_pos_slack: torch.Tensor = None


@dataclass
class SolverResult:
    dual_val: torch.Tensor
    dual_objective: float
    objective_result: ObjectiveResult
    dual_objective_log: list[float]
    step_size_log: list[float]

import math

import torch

from dualip.objectives.base import BaseObjective
from dualip.optimizers.agd_utils import calculate_step_size
from dualip.types import ObjectiveResult, SolverResult
from dualip.utils.mlflow_utils import log_metrics, log_objective_result


def project_on_nn_cone(y: torch.Tensor, equality_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Projects the dual variables onto the nonnegative cone.
    """
    projected = torch.maximum(y, torch.tensor(0.0, device=y.device))
    if equality_mask is not None:
        return torch.where(equality_mask, y, projected)
    else:
        return projected


class AcceleratedGradientDescent:
    def __init__(
        self,
        max_iter: int,
        gamma: float,
        initial_step_size: float = 1e-5,
        max_step_size: float = 0.1,
        gamma_decay_type: str = None,
        gamma_decay_params: dict = {},
        save_primal: bool = False,
    ):

        self.initial_step_size = initial_step_size
        self.max_step_size = max_step_size
        self.max_iter = max_iter
        self.beta_seq = self._compute_beta_seq(self.max_iter)
        self.streams = None
        self.gamma = gamma
        self.gamma_decay_type = gamma_decay_type
        self.gamma_decay_params = gamma_decay_params
        self.save_primal = save_primal

    def _compute_beta_seq(self, max_iter: int) -> torch.Tensor:
        t_seq = torch.zeros(max_iter + 2)
        beta_seq = torch.zeros(max_iter)
        for i in range(1, max_iter + 2):
            t_seq[i] = (1 + math.sqrt(1 + 4 * (t_seq[i - 1] ** 2))) / 2
        for i in range(max_iter):
            beta_seq[i] = (1 - t_seq[i + 1]) / t_seq[i + 2]
        return beta_seq

    def _update_gamma(self, itr: int, step_size: float):
        if self.gamma_decay_type == "step":
            if itr % self.gamma_decay_params["decay_steps"] == 0:
                decay_factor = self.gamma_decay_params["decay_factor"]
                self.gamma = self.gamma * decay_factor
                self.max_step_size = step_size * decay_factor
        else:
            raise ValueError(f"Unsupported gamma decay type: {self.gamma_decay_type}")

    def maximize(self, f: BaseObjective, initial_value: torch.Tensor) -> SolverResult:
        """
        Maximizes the dual-primal objective function f.
        f must provide a method:
          - f.calculate(x) returning an object with attributes:
              * dual_gradient (torch.Tensor)
              * dual_objective (float)
              * dual_val (torch.Tensor)
        Returns a tuple: (final solution, final result, dual_obj_log, step_size_log),
        where dual_obj_log is the list of dual objective values recorded at each iteration
        and step_size_log is the list of the dynamic step size.
        """
        grad_history = []
        dual_history = []
        dual_obj_log = []  # Log of dual objective values per iteration
        step_size_log = []

        # x and y for the accelerated update.
        x = initial_value.clone()
        y = initial_value.clone()
        equality_mask = f.equality_mask

        i = 1
        while i <= self.max_iter:

            gamma_params = {"gamma": self.gamma} if self.gamma is not None else {}

            if i == self.max_iter and self.save_primal:
                objective_result: ObjectiveResult = f.calculate(
                    dual_val=x, **gamma_params, save_primal=self.save_primal
                )
            else:
                objective_result: ObjectiveResult = f.calculate(dual_val=x, **gamma_params)

            dual_obj = objective_result.dual_objective.cpu().item()
            dual_obj_log.append(dual_obj)

            step_size = calculate_step_size(
                objective_result.dual_gradient,
                y,
                grad_history,
                dual_history,
                initial_step_size=self.initial_step_size,
                max_step_size=self.max_step_size,
            )

            step_size_log.append(step_size)
            # Gradient ascent step.
            y_new = x + objective_result.dual_gradient * step_size
            y_new = project_on_nn_cone(y_new, equality_mask)
            # Accelerated update.
            x = (y_new * (1.0 - self.beta_seq[i - 1])) + (y * self.beta_seq[i - 1])
            y = y_new
            if self.gamma is not None and self.gamma_decay_type is not None:
                self._update_gamma(i, step_size)

            # Log iteration metrics (will check MLflow state internally)
            iteration_metrics = {
                "step_size": step_size,
                "dual_objective": dual_obj,
            }

            if self.gamma is not None:
                iteration_metrics["gamma"] = self.gamma

            log_metrics(iteration_metrics, step=i)

            # Log objective result details
            log_objective_result(objective_result, step=i)

            i += 1

        solver_result = SolverResult(
            dual_val=y,
            dual_objective=dual_obj,
            objective_result=objective_result,
            dual_objective_log=dual_obj_log,
            step_size_log=step_size_log,
        )
        return solver_result

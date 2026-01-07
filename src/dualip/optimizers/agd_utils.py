import torch


def norm_of_difference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate the L2 norm of the difference between two tensors.
    """
    diff = x - y
    return torch.norm(diff)


def update_dual_gradient_history(
    gradient: torch.Tensor,
    dual_val: torch.Tensor,
    grad_history: list,
    dual_history: list,
    max_history_length: int,
) -> None:
    """
    Updates the history lists for gradients and dual variables.
    If the history length reaches max_history_length, remove the oldest entries.
    """
    if len(grad_history) == max_history_length:
        # Ensure both histories are kept in sync.
        dual_history.pop(0)
        grad_history.pop(0)
    grad_history.append(gradient.clone().detach())
    dual_history.append(dual_val.clone().detach())


def estimate_lipschitz_constant(
    grad_one: torch.Tensor,
    grad_two: torch.Tensor,
    dual_one: torch.Tensor,
    dual_two: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate the Lipschitz constant using the ratio:
        L = norm(grad_two - grad_one) / norm(dual_two - dual_one)
    """
    return norm_of_difference(grad_one, grad_two) / norm_of_difference(dual_one, dual_two)


def step_size_from_lipschitz_constants(
    lipschitz_constants: list,
    max_history_length: int,
    initial_step_size: float,
    max_step_size: float,
) -> float:
    """
    Compute the candidate step size as the inverse of the maximum Lipschitz constant,
    provided that the history is full. If the history is not complete, if the maximum constant
    is NaN or infinite, or if there are too few constants computed, then return the initial step size.
    Otherwise, return the candidate step size clamped by max_step_size.
    """
    if not lipschitz_constants or len(lipschitz_constants) < max_history_length - 1:
        return initial_step_size
    L_max = max(lipschitz_constants)
    if torch.isnan(L_max) or torch.isinf(L_max):
        return initial_step_size
    candidate = 1.0 / L_max.item() if L_max != 0 else max_step_size
    return min(candidate, max_step_size)


def calculate_step_size(
    dual_grad: torch.Tensor,
    dual_val: torch.Tensor,
    grad_history: list,
    dual_history: list,
    max_history_length: int = 15,
    initial_step_size: float = 1e-5,
    max_step_size: float = 0.1,
) -> float:
    """
    Approximate step size calculation based on the history of dual gradients and dual variables.

    The function follows these steps:
      1. Update the histories for the current gradient and dual variable.
      2. For each consecutive pair in the history, estimate a Lipschitz constant.
      3. Return the candidate step size as 1 / (maximum Lipschitz constant) clamped to the specified bounds.
         If the history is incomplete or invalid, return initial_step_size.
    """
    update_dual_gradient_history(dual_grad, dual_val, grad_history, dual_history, max_history_length)

    lipschitz_constants = []
    for i in range(len(grad_history) - 1):
        L_est = estimate_lipschitz_constant(grad_history[i], grad_history[i + 1], dual_history[i], dual_history[i + 1])
        lipschitz_constants.append(L_est)
    return step_size_from_lipschitz_constants(lipschitz_constants, max_history_length, initial_step_size, max_step_size)

import torch
from dualip.projections.base import ProjectionOperator, register


def _proj_via_bisection_search(
    x: torch.Tensor,
    z: float = 1.0,
    inequality: bool = False,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> torch.Tensor:
    """
    Implementation of a bisection search algorithm.
    Goal: search for optimal nu (nu_star) given that we know w = max(x - nu_star*1, 0) (See https://see.stanford.edu/materials/lsocoee364b/hw4sol.pdf)
    Projects a batch of vectors onto the z-simplex using bisection search.
    Args:
        x (torch.Tensor): A tensor of shape (L, B) where L is the vector
                          dimension and B is the batch size.
        z (float): The radius of the simplex. Must be positive.
        inequality (bool): If True, project onto {w | sum(w) <= z, w >= 0}.
                           If False, project onto {w | sum(w) = z, w >= 0}.
        tol (float): Tolerance for bisection search convergence.
        max_iter (int): Maximum number of iterations for bisection search.

    Returns:
        torch.Tensor: The projected vectors, of the same shape as x.
    """
    L, B = x.shape
    assert z > 0, "Simplex radius z must be positive."
    device, dtype = x.device, x.dtype

    w = torch.empty_like(x)

    # mask for columns that need projection via bisection search
    to_project_mask = torch.zeros(B, dtype=torch.bool, device=device)
    # ---------- Early check for the case of inequality  ----------
    if inequality:
        is_feasible = (x.sum(dim=0) <= z + tol) & (x >= -tol).all(dim=0)
        w[:, is_feasible] = x[:, is_feasible]
        infeasible_mask = ~is_feasible
        if not infeasible_mask.any():
            return w
        to_project_mask[infeasible_mask] = True
    else:
        to_project_mask.fill_(True)

    if not to_project_mask.any():
        return w
    # ---------- Add check from modified Duchi Algorithm ----------
    if L > 1:
        # only check the columns that are candidates for projection.
        to_project_indices = to_project_mask.nonzero(as_tuple=True)[0]
        to_project = x[:, to_project_indices]

        # normalized to handle z != 1
        to_project_normalized = to_project / z

        # get top 2 values to check the condition
        vals, indices = torch.topk(to_project_normalized, 2, dim=0)

        # condition for direct vertex projection: max_val > second_max_val + 1
        shortcut_mask = (vals[0] - vals[1]) > 1.0

        if shortcut_mask.any():
            # get the columns that satisfy the shortcut
            shortcut_cols_idx = to_project_indices[shortcut_mask]

            # for these columns, the solution is a one-hot vector
            shortcut_max_indices = indices[0, shortcut_mask]
            solution = torch.zeros(L, shortcut_cols_idx.shape[0], device=device, dtype=dtype)
            solution[shortcut_max_indices, torch.arange(solution.shape[1])] = z

            # update the final output tensor
            w[:, shortcut_cols_idx] = solution

            # update mask
            to_project_mask[shortcut_cols_idx] = False

    if not to_project_mask.any():
        return w
    # ---------- Bisection search on remaining columns ----------
    to_project = x[:, to_project_mask]
    x_normalized = to_project / z
    x_max = torch.max(x_normalized, dim=0)[0]

    x_shifted = to_project - x_max.unsqueeze(0)

    nu_low = -1.0
    nu_high = torch.zeros_like(x_max)
    nu_mid_prev = None

    active_mask = torch.ones(to_project.shape[1], dtype=torch.bool, device=device)

    for iteration in range(max_iter):
        if not active_mask.any():
            break

        nu_mid = (nu_low + nu_high) / 2.0
        if nu_mid_prev is not None:
            # check convergence on nu_mid
            if torch.abs(nu_mid - nu_mid_prev).max() < tol:
                break

        S = torch.clamp(x_shifted - nu_mid.unsqueeze(0), min=0.0).sum(dim=0)

        too_high = (S > 1.0) & active_mask
        too_low = (S <= 1.0) & active_mask

        nu_low = torch.where(too_high, nu_mid, nu_low)
        nu_high = torch.where(too_low, nu_mid, nu_high)

        converged = (nu_high - nu_low) < tol
        active_mask = active_mask & ~converged

        nu_mid_prev = nu_mid.clone()

    nu_star = (nu_low + nu_high) / 2.0
    w_proj_shifted = torch.clamp(x_shifted - nu_star.unsqueeze(0), min=0.0)
    w[:, to_project_mask] = w_proj_shifted * z
    return w


def _duchi_proj(
    x: torch.Tensor, z: float, inequality: bool = False, tol: float = 1e-6, cols_per_chunk: int = 10000
) -> torch.Tensor:
    """
    Projects each column of x onto the general simplex with radius z {w >= 0, sum(w) = z}
    (or {w >= 0, sum(w) <= z} if inequality=True), using a batched implementation
    of the Duchi et al. algorithm.

    Args:
        x: Tensor of shape (L, B), B vectors of length L in the columns.
        z: Simplex radius (must be > 0).
        inequality: If True, projects onto {w >= 0, sum(w) <= z}.
        tol: Numerical tolerance for early feasibility checks.

    Returns:
        w: Tensor of shape (L, B), each column projected onto the chosen simplex.
    """
    L, B = x.shape
    assert z > 0, "Simplex radius z must be positive."
    device, dtype = x.device, x.dtype

    w = torch.empty_like(x)
    x = torch.clamp(x, min=0.0)

    # mask for columns that need projection via bisection search
    to_project_mask = torch.zeros(B, dtype=torch.bool, device=device)
    # ---------- Early check for the case of inequality  ----------
    if inequality:
        is_feasible = (x.sum(dim=0) <= z + tol) & (x >= -tol).all(dim=0)
        w[:, is_feasible] = x[:, is_feasible]
        infeasible_mask = ~is_feasible
        if not infeasible_mask.any():
            return w
        to_project_mask[infeasible_mask] = True
    else:
        to_project_mask.fill_(True)

    if not to_project_mask.any():
        return w
    # ---------- Add check from modified Duchi Algorithm ----------
    if L > 1:
        # only check the columns that are candidates for projection.
        to_project_indices = to_project_mask.nonzero(as_tuple=True)[0]
        to_project = x[:, to_project_indices]

        # normalized to handle z != 1
        to_project_normalized = to_project / z

        # get top 2 values to check the condition
        vals, indices = torch.topk(to_project_normalized, 2, dim=0)

        # condition for direct vertex projection: max_val > second_max_val + 1
        shortcut_mask = (vals[0] - vals[1]) > 1.0

        if shortcut_mask.any():
            # get the columns that satisfy the shortcut
            shortcut_cols_idx = to_project_indices[shortcut_mask]

            # for these columns, the solution is a one-hot vector
            shortcut_max_indices = indices[0, shortcut_mask]
            solution = torch.zeros(L, shortcut_cols_idx.shape[0], device=device, dtype=dtype)
            solution[shortcut_max_indices, torch.arange(solution.shape[1])] = z

            # update the final output tensor
            w[:, shortcut_cols_idx] = solution

            # update mask
            to_project_mask[shortcut_cols_idx] = False

    if not to_project_mask.any():
        return w


    to_project = x[:, to_project_mask]
    proj_cols_idx = to_project_mask.nonzero(as_tuple=True)[0]

    _, K = to_project.shape
    for c0 in range(0, K, cols_per_chunk):
        c1 = min(c0 + cols_per_chunk, K)
        to_project_sub = to_project[:, c0:c1]
        proj_cols_idx_sub = proj_cols_idx[c0:c1]

        # 1) Sort each column in descending order
        u_sorted, _ = to_project_sub.sort(dim=0, descending=True)

        # 2) Compute cumulative sums
        cumsum_u = u_sorted.cumsum(dim=0)

        # 3) Build index vectors
        idx = torch.arange(1, L + 1, device=device)  # long: [1,2,…,L]
        idx0 = idx - 1  # zero-based: [0,1,…,L-1]
        idx_f = idx.to(dtype).view(L, 1)  # float version for arithmetic

        # 4) Find the threshold condition: u_i - (cumsum_u[i] - z)/i > 0
        cond = u_sorted - (cumsum_u - z) / idx_f > 0

        # 5) For each column j, rho_j = max{i-1 | cond[i,j] == True}
        mask = cond.to(torch.long) * idx0.view(L, 1)
        rho = mask.max(dim=0).values  # shape (num_proj,), dtype long

        # 6) Compute theta_j = (cumsum_u[rho_j, j] - z) / (rho_j + 1)
        col_indices = torch.arange(rho.size(0), device=device)
        cumsum_at_rho = cumsum_u[rho, col_indices]
        theta = (cumsum_at_rho - z) / (rho.to(dtype) + 1)

        # 7) Threshold & clamp
        proj_cols = (to_project_sub - theta.unsqueeze(0)).clamp(min=0)

        # 8) Scatter projected columns back into w
        w[:, proj_cols_idx_sub] = proj_cols

    return w


@register("simplex")
class SimplexIneq(ProjectionOperator):
    # inequality: sum(x) <= z exactly
    def __init__(self, z: float = 1.0, method: str = "duchi"):
        self.z = z
        self.proj_method = method
        if self.proj_method not in ("duchi", "bisection_search"):
            raise ValueError(f"Unsupported projection method: {self.proj_method}")

    def __call__(self, x: torch.Tensor):
        if x.ndim == 1:
            x = x.unsqueeze(1)

        if self.proj_method == "bisection_search":
            return _proj_via_bisection_search(x, z=self.z, inequality=True)
        elif self.proj_method == "duchi":
            return _duchi_proj(x=x, z=self.z, inequality=True)


@register("simplex_eq")
class SimplexEq(ProjectionOperator):
    # equality: sum(x) = z exactly
    def __init__(self, z: float = 1.0, method: str = "duchi"):
        self.z = z
        self.proj_method = method
        if self.proj_method not in ("duchi", "bisection_search"):
            raise ValueError(f"Unsupported projection method: {self.proj_method}")

    def __call__(self, x: torch.Tensor):
        if x.ndim == 1:
            x = x.unsqueeze(1)

        if self.proj_method == "bisection_search":
            return _proj_via_bisection_search(x, z=self.z, inequality=False)
        elif self.proj_method == "duchi":
            return _duchi_proj(x=x, z=self.z, inequality=False)

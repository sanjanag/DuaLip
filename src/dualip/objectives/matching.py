from dataclasses import dataclass
from operator import add, mul

import torch
import torch.distributed as dist
import torch.cuda.comm as cuda_comm
from dualip.objectives.base import BaseInputArgs, BaseObjective, ObjectiveResult
from dualip.projections.base import ProjectionEntry, project
from dualip.utils.dist_utils import global_to_local_projection_map, split_tensors_to_devices
from dualip.utils.sparse_utils import apply_F_to_columns, elementwise_csc, left_multiply_sparse, row_sums_csc


@dataclass
class MatchingInputArgs(BaseInputArgs):
    """
    Input arguments specific to Matching objective function.
    """

    A: torch.Tensor
    c: torch.Tensor
    projection_map: dict[str, ProjectionEntry]
    b_vec: torch.Tensor
    equality_mask: torch.Tensor = None


def calc_grad(
    dual_grad: torch.Tensor,
    dual_obj: torch.Tensor,
    dual_val: torch.Tensor,
    b_vec: torch.Tensor,
    reg_penalty: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dual_grad = dual_grad - b_vec
    dual_obj = dual_obj + reg_penalty + torch.dot(dual_val, dual_grad)
    return dual_grad, dual_obj


class MatchingSolverDualObjectiveFunction(BaseObjective):
    """
    Computes dual gradient, objective, and regularization penalty
    for a (single-GPU) matching problem.
    """

    def __init__(
        self,
        matching_input_args: MatchingInputArgs,
        gamma: float,
        batching: bool = True,
    ):
        self.A = matching_input_args.A
        self.c = matching_input_args.c
        self.gamma = gamma
        self.b_vec = matching_input_args.b_vec
        self.projection_map = matching_input_args.projection_map
        # If b_vec is provided, then this is a total single device objecitve function, otherwise this class
        # is being used to encapsulate single-GPU computation in the distributed setting.
        self.is_distributed = self.b_vec is None
        self.equality_mask = matching_input_args.equality_mask

        device = self.A.device

        # Batching variables
        self._thresholds = []
        self._bucket_ids = None

        # Precompute c_rescaled = -c / gamma
        self.c_rescaled = -1.0 / gamma * self.c

        # Build buckets
        self.buckets = {}
        for proj_key, proj_item in self.projection_map.items():
            indices = torch.tensor(proj_item.indices, dtype=torch.int32, device=device)
            proj_type = proj_item.proj_type
            proj_params = proj_item.proj_params
            if batching:
                self.buckets[proj_key] = (self._compute_buckets(indices), proj_type, proj_params)
            else:
                self.buckets[proj_key] = ([indices], proj_type, proj_params)

        # Pre-allocate a CSC tensor to hold intermediate results
        self.intermediate = torch.sparse_csc_tensor(
            self.A.ccol_indices(),
            self.A.row_indices(),
            torch.zeros_like(self.A.values()),
            size=self.A.size(),
        )

    def _compute_buckets(self, indices: list[torch.Tensor]) -> list[list[torch.Tensor]]:
        device = self.A.device
        if not self._thresholds:
            ccol_ptr = self.A.ccol_indices()

            # build thresholds
            self._thresholds = [0]
            i = 1
            max_nnz = self.A.size(0)
            while 2**i <= max_nnz:
                self._thresholds.append(2**i)
                i += 1
            self._thresholds.append(max_nnz + 1)

            # per-column nnz counts
            lengths = ccol_ptr[1:] - ccol_ptr[:-1]
            th_tensor = torch.tensor(self._thresholds, dtype=lengths.dtype, device=device)
            self._bucket_ids = torch.bucketize(lengths.to(device), th_tensor)

        indices = torch.as_tensor(indices, dtype=torch.int32, device=device)
        proj_bucket_ids = self._bucket_ids[indices]
        # for each nnz bucket, gather columns of this projection type that fall into that bucket
        buckets = []
        for j in range(1, len(self._thresholds)):
            bucket = indices[proj_bucket_ids == j]
            if bucket.numel() > 0:
                buckets.append(bucket)
        return buckets

    def calculate(
        self,
        dual_val: torch.Tensor,
        gamma: float = None,
        save_primal: bool = False,
    ) -> ObjectiveResult:
        """
        Compute dual gradient, objective, and reg penalty.

        Args:
            dual_val: current dual variables
            gamma: regularization parameter
            save_primal: if True, save the primal variable

        Returns:
            ObjectiveResult
        """
        if gamma is not None and gamma != self.gamma:
            self.gamma = gamma
            # Recompute c_rescaled when gamma changes
            self.c_rescaled = -1.0 / gamma * self.c

        # -dual_val/gamma
        scaled = -1.0 / self.gamma * dual_val

        # intermediate = A * scaled
        left_multiply_sparse(scaled, self.A, output_tensor=self.intermediate)

        # intermediate += c_rescaled
        elementwise_csc(self.intermediate, self.c_rescaled, add, output_tensor=self.intermediate)

        # apply each projection
        for _, proj_item in self.buckets.items():
            buckets = proj_item[0]
            proj_type = proj_item[1]
            proj_params = proj_item[2]
            fn = project(proj_type, **proj_params)
            apply_F_to_columns(self.intermediate, fn, buckets, output_tensor=self.intermediate)

        # dual gradient = row sums of A * intermediate
        grad = row_sums_csc(elementwise_csc(self.A, self.intermediate, mul))

        # reg penalty = (gamma/2) * ||intermediate.values||^2
        vals = self.intermediate.values()
        reg_penalty = (self.gamma / 2) * torch.norm(vals) ** 2

        # dual objective = c * intermediate.values
        dual_obj = torch.dot(self.c.values(), vals)
        primal_obj = dual_obj.clone()
        primal_var = vals

        if not self.is_distributed and self.b_vec is not None:
            grad, dual_obj = calc_grad(grad, dual_obj, dual_val, self.b_vec, reg_penalty)

            dual_val_times_grad = torch.dot(dual_val, grad)
            max_pos_slack = max(torch.max(grad), 0)
            sum_pos_slack = torch.relu(grad).sum()

            obj_result = ObjectiveResult(
                dual_gradient=grad,
                dual_objective=dual_obj,
                reg_penalty=reg_penalty,
                dual_val_times_grad=dual_val_times_grad,
                max_pos_slack=max_pos_slack,
                sum_pos_slack=sum_pos_slack,
            )
        else:
            obj_result = ObjectiveResult(
                dual_gradient=grad,
                dual_objective=dual_obj,
                reg_penalty=reg_penalty,
            )
        if save_primal:
            obj_result.primal_var = primal_var
            obj_result.primal_objective = primal_obj
        return obj_result


class MatchingSolverDualObjectiveFunctionDistributed(BaseObjective):
    """
    Wrap the single-GPU objective across multiple devices.
    """

    def __init__(
        self,
        matching_input_args: MatchingInputArgs,
        gamma: float,
        host_device: torch.device,
        compute_devices: list[torch.device],
    ):
        self.gamma = gamma
        self.host_device = host_device
        self.compute_devices = compute_devices
        self.equality_mask = matching_input_args.equality_mask
        self.A = matching_input_args.A
        self.c = matching_input_args.c
        self.b_vec = matching_input_args.b_vec.to(host_device)
        self.projection_map = matching_input_args.projection_map

        # Split data for each GPU
        A_splits, c_splits, split_index_map = split_tensors_to_devices(self.A, self.c, compute_devices)
        self.objectives = []
        for idx, (A_part, c_part) in enumerate(zip(A_splits, c_splits)):
            pm = global_to_local_projection_map(self.projection_map, split_index_map[idx])
            part_input_args = MatchingInputArgs(A_part, c_part, pm, b_vec=None, equality_mask=self.equality_mask)
            self.objectives.append(MatchingSolverDualObjectiveFunction(part_input_args, self.gamma))

    def calculate(
        self,
        dual_val: torch.Tensor,
        gamma: float = None,
        save_primal: bool = False,
        rank: int = 0,
    ) -> ObjectiveResult:
        """Compute and reduce gradients/objectives across all GPUs."""
        if save_primal:
            raise NotImplementedError("save_primal=True is not yet supported in distributed mode")

        # Keep a copy of dual_val on host device for final calculations
        dual_val_host = dual_val.clone()

        # Each rank computes its partition
        objective = self.objectives[rank]
        dual_val_gpu = dual_val.to(f"cuda:{rank}", non_blocking=True)
        objective_result = objective.calculate(dual_val_gpu, gamma, save_primal=False)

        # Move results back to host device for reduction
        grad_host = objective_result.dual_gradient.to(self.host_device)
        obj_host = objective_result.dual_objective.to(self.host_device)
        reg_host = objective_result.reg_penalty.to(self.host_device)

        # ALL ranks participate in collective operations
        dist.all_reduce(grad_host, op=dist.ReduceOp.SUM)
        dist.all_reduce(obj_host, op=dist.ReduceOp.SUM)
        dist.all_reduce(reg_host, op=dist.ReduceOp.SUM)

        # ALL ranks synchronize
        dist.barrier()

        # Only rank 0 computes the final objective and returns meaningful result
        if rank == 0:
            grad = grad_host - self.b_vec
            dual_val_times_grad = torch.dot(dual_val_host, grad)
            obj = obj_host + reg_host + dual_val_times_grad

            max_pos_slack = max(torch.max(grad), 0)
            sum_pos_slack = torch.relu(grad).sum()

            obj_result = ObjectiveResult(
                dual_gradient=grad,
                dual_objective=obj,
                reg_penalty=reg_host,
                dual_val_times_grad=dual_val_times_grad,
                max_pos_slack=max_pos_slack,
                sum_pos_slack=sum_pos_slack,
            )

            return obj_result
        else:
            # Non-zero ranks return a dummy result (won't be used by optimizer)
            return ObjectiveResult(
                dual_gradient=grad_host,
                dual_objective=obj_host,
                reg_penalty=reg_host,
            )

from dataclasses import dataclass
from operator import add, mul
import time

import torch
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
                bucket_list = self._compute_buckets(indices)
                metadata = self._compute_bucket_metadata(bucket_list)
                self.buckets[proj_key] = (bucket_list, proj_type, proj_params, metadata)
            else:
                bucket_list = [indices]
                metadata = self._compute_bucket_metadata(bucket_list)
                self.buckets[proj_key] = (bucket_list, proj_type, proj_params, metadata)

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

    def _compute_bucket_metadata(self, buckets: list[torch.Tensor]) -> list[tuple[int, int, torch.Tensor]]:
        """
        Pre-compute metadata (total, L, lengths_cpu) for each bucket to avoid GPU sync during iterations.

        Args:
            buckets: List of column index tensors

        Returns:
            List of (total, L, lengths_cpu) tuples for each bucket
        """
        ccol = self.A.ccol_indices()
        metadata = []

        for cols in buckets:
            if cols.numel() == 0:
                metadata.append((0, 0, torch.tensor([], dtype=torch.long)))
                continue

            starts = ccol[cols]
            ends = ccol[cols + 1]
            lengths = ends - starts

            # Compute on CPU to get Python ints and store CPU tensor for repeat_interleave
            lengths_cpu = lengths.cpu()
            total = int(lengths_cpu.sum().item())
            L = int(lengths_cpu.max().item())
            metadata.append((total, L, lengths_cpu))

        return metadata

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
            metadata = proj_item[3]
            fn = project(proj_type, **proj_params)
            apply_F_to_columns(self.intermediate, fn, buckets, metadata=metadata, output_tensor=self.intermediate)

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

        self.compute_devices = [torch.device(d) for d in compute_devices]  # ["cuda:0","cuda:1"] -> devices
        self.compute_device_indices = [d.index for d in self.compute_devices]  # [0, 1]
        self.host_device = torch.device(host_device)  # should be torch.device("cuda:0")
        self.equality_mask = matching_input_args.equality_mask
        self.A = matching_input_args.A
        self.c = matching_input_args.c
        self.b_vec = matching_input_args.b_vec.to(self.host_device, non_blocking=True)
        self.projection_map = matching_input_args.projection_map

        # Split data for each GPU
        A_splits, c_splits, split_index_map = split_tensors_to_devices(self.A, self.c, compute_devices)
        self.objectives = []
        for idx, (A_part, c_part) in enumerate(zip(A_splits, c_splits)):
            pm = global_to_local_projection_map(self.projection_map, split_index_map[idx])
            part_input_args = MatchingInputArgs(A_part, c_part, pm, b_vec=None, equality_mask=self.equality_mask)
            self.objectives.append(MatchingSolverDualObjectiveFunction(part_input_args, self.gamma))
        self.streams = {dev: torch.cuda.Stream(device=dev) for dev in self.compute_devices}

    def calculate(
        self,
        dual_val: torch.Tensor,
        gamma: float = None,
        save_primal: bool = False,
    ) -> ObjectiveResult:
        """Compute and reduce gradients/objectives across all GPUs."""

        if save_primal:
            raise ValueError("Saving primal is not supported for distributed objective functions")

        if gamma is not None:
            self.gamma = gamma

        # Ensure dual_val lives on host_device (cuda:0) to use fast GPU->GPU broadcast
        if dual_val.device != self.host_device:
            dual_val = dual_val.to(self.host_device, non_blocking=True)

        # 1) Broadcast dual_val to all compute devices (includes host_device as first element if you pass it)
        # Order matters: match this list to your objectives/devices iteration below.
        dv_per_dev = cuda_comm.broadcast(dual_val, devices=self.compute_device_indices)

        grads_per_dev = []
        dual_objs_per_dev = []
        regs_per_dev = []
        res_per_dev = []

        loop_start = time.perf_counter()
        for solver, dev, dv in zip(self.objectives, self.compute_devices, dv_per_dev):
            stream = self.streams[dev]
            enqueue_start = time.perf_counter()
            with torch.cuda.stream(stream):

                res = solver.calculate(dv, gamma, save_primal=False)
                res_per_dev.append(res)
            enqueue_time = time.perf_counter() - enqueue_start
            print(f"[Parallelism] Device {dev}: enqueue took {enqueue_time:.6f}s")

        loop_time = time.perf_counter() - loop_start
        print(f"[Parallelism] Total loop time: {loop_time:.6f}s")

        sync_start = time.perf_counter()
        for dev in self.compute_devices:
            self.streams[dev].synchronize()
        sync_time = time.perf_counter() - sync_start
        print(f"[Parallelism] Synchronization time: {sync_time:.6f}s")

        for res in res_per_dev:
            grads_per_dev.append(res.dual_gradient)
            dual_objs_per_dev.append(res.dual_objective)
            regs_per_dev.append(res.reg_penalty)

        total_grad = cuda_comm.reduce_add(grads_per_dev, destination=self.host_device.index)

        total_dual_obj = cuda_comm.reduce_add(dual_objs_per_dev, destination=self.host_device.index)

        total_reg = cuda_comm.reduce_add(regs_per_dev, destination=self.host_device.index)

        # final adjustments

        grad = total_grad - self.b_vec
        dual_val_times_grad = torch.dot(dual_val, grad)
        dual_obj = total_dual_obj + total_reg + dual_val_times_grad

        max_pos_slack = max(torch.max(grad), 0)
        sum_pos_slack = torch.relu(grad).sum()

        obj_result = ObjectiveResult(
            dual_gradient=grad,
            dual_objective=dual_obj,
            reg_penalty=total_reg,
            dual_val_times_grad=dual_val_times_grad,
            max_pos_slack=max_pos_slack,
            sum_pos_slack=sum_pos_slack,
        )

        return obj_result

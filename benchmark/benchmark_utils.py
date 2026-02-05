"""
Shared utilities for benchmarking.
"""

import csv
import time

import torch
from generate_synthetic_data import generate_synthetic_matching_input_args

from dualip.objectives.matching import MatchingInputArgs
from dualip.preprocessing.precondition import jacobi_precondition


def generate_benchmark_data(
    num_sources: int,
    num_destinations: int,
    target_sparsity: float,
    device: str,
    use_preconditioning: bool = False,
    rng=None,
) -> tuple[MatchingInputArgs, float]:
    """
    Generate synthetic matching data with optional preconditioning.

    Args:
        num_sources: Number of source nodes
        num_destinations: Number of destination nodes
        target_sparsity: Target sparsity for the constraint matrix
        device: Device to generate data on
        use_preconditioning: Whether to apply Jacobi preconditioning
        rng: Random number generator (optional)

    Returns:
        Tuple of (input_args, generation_time)
    """
    print("\nGenerating data...")
    t0 = time.perf_counter()
    input_args: MatchingInputArgs = generate_synthetic_matching_input_args(
        num_sources=num_sources,
        num_destinations=num_destinations,
        target_sparsity=target_sparsity,
        device=device,
        rng=rng,
    )
    data_time = time.perf_counter() - t0
    print(f"      {data_time:.3f}s | NNZ: {input_args.A._nnz()}")

    if use_preconditioning:
        print("      Applying preconditioning...")
        jacobi_precondition(input_args.A, input_args.b_vec)

    return input_args, data_time


def get_output_filename(
    num_sources: int,
    num_destinations: int,
    target_sparsity: float,
    gamma: float,
    max_iter: int,
    use_preconditioning: bool = False,
    gamma_decay_factor: float = None,
    gamma_decay_steps: int = None,
) -> str:
    """
    Generate informative filename based on parameters.

    Args:
        num_sources: Number of source nodes
        num_destinations: Number of destination nodes
        target_sparsity: Target sparsity
        gamma: Gamma (regularization) value
        max_iter: Maximum iterations
        use_preconditioning: Whether preconditioning was used
        gamma_decay_factor: Decay factor (if using decay)
        gamma_decay_steps: Decay steps (if using decay)

    Returns:
        CSV filename string
    """
    parts = [
        f"s{num_sources//1_000_000}M",
        f"d{num_destinations//1_000}K",
        f"sp{target_sparsity}",
        f"g{gamma}",
        f"iter{max_iter}",
    ]
    if use_preconditioning:
        parts.append("precon")
    if gamma_decay_factor is not None and gamma_decay_steps is not None:
        parts.append(f"decay{gamma_decay_factor}x{gamma_decay_steps}")
    return "_".join(parts) + ".csv"


def print_config(
    num_sources: int,
    num_destinations: int,
    target_sparsity: float,
    seed: int,
    gamma: float,
    max_iter: int,
    use_preconditioning: bool,
    device: str = None,
    world_size: int = None,
    use_gamma_decay: bool = False,
    initial_gamma: float = None,
    final_gamma: float = None,
):
    """
    Print benchmark configuration in a formatted way.

    Args:
        num_sources: Number of source nodes
        num_destinations: Number of destination nodes
        target_sparsity: Target sparsity
        seed: Random seed
        gamma: Gamma value
        max_iter: Maximum iterations
        use_preconditioning: Whether preconditioning is enabled
        device: Device string (for single-GPU mode)
        world_size: Number of processes (for distributed mode)
        use_gamma_decay: Whether gamma decay is enabled
        initial_gamma: Initial gamma (if using decay)
        final_gamma: Final gamma (if using decay)
    """
    print("=" * 60)
    print("CONFIG")
    print("=" * 60)
    print(f"  Data: {num_sources} sources x {num_destinations} destinations")
    print(f"  Sparsity: {target_sparsity}")
    print(f"  Seed: {seed}")

    if device is not None:
        print(f"  Device: {device}")
    if world_size is not None:
        print(f"  World size: {world_size}")

    print(f"  Preconditioning: {use_preconditioning}")
    print(f"  Gamma decay: {use_gamma_decay}")

    if use_gamma_decay and initial_gamma is not None and final_gamma is not None:
        print(f"  Gamma: {initial_gamma} -> {final_gamma}, Max iter: {max_iter}")
    else:
        print(f"  Gamma: {gamma}, Max iter: {max_iter}")

    print("=" * 60)


def print_results(result, solve_time: float, max_iter: int):
    """
    Print benchmark results in a formatted way.

    Args:
        result: Solver result object
        solve_time: Time taken to solve (seconds)
        max_iter: Maximum iterations
    """
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Solve time: {solve_time:.3f}s ({solve_time/max_iter*1000:.2f} ms/iter)")
    print(f"  Dual objective: {result.dual_objective:.6f}")

    # Primal objective may not be available in distributed mode
    if (
        hasattr(result.objective_result, "primal_objective")
        and result.objective_result.primal_objective is not None
    ):
        print(f"  Primal objective: {result.objective_result.primal_objective.item():.6f}")

    print(f"  Reg penalty: {result.objective_result.reg_penalty.item():.6f}")

    max_slack = result.objective_result.max_pos_slack
    max_slack = max_slack.item() if hasattr(max_slack, "item") else max_slack
    print(f"  Max positive slack: {max_slack:.6e}")
    print(f"  Sum positive slack: {result.objective_result.sum_pos_slack.item():.6e}")
    print("=" * 60)


def save_dual_curve(result, filename: str):
    """
    Save dual objective curve to CSV file.

    Args:
        result: Solver result object with dual_objective_log
        filename: Output CSV filename
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "dual_objective"])
        for i, obj in enumerate(result.dual_objective_log, 1):
            writer.writerow([i, obj])
    print(f"\nDual objective curve saved to: {filename}")

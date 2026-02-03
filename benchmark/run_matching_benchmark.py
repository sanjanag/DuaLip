"""
Single-GPU benchmark for matching problem.
Edit the CONFIG section below to change parameters.
"""

import argparse
import csv
import time

import torch
from generate_synthetic_data import generate_synthetic_matching_input_args

from dualip.objectives.matching import MatchingInputArgs, MatchingSolverDualObjectiveFunction
from dualip.optimizers.agd import AcceleratedGradientDescent
from dualip.preprocessing.precondition import jacobi_precondition

# =============================================================================
# CONFIG - Edit these values
# =============================================================================

# Data parameters (fixed across all runs)
NUM_SOURCES = 25_000_000
NUM_DESTINATIONS = 10_000
TARGET_SPARSITY = 0.001
SEED = 42
DTYPE = torch.float32  # Floating point precision (torch.float32, torch.float64)

# Base directory for all experiment data
BASE_DIR = "./benchmark_data"  # Base directory for cache and results

# Solver parameters (fixed across all runs)
FINAL_GAMMA = 1e-3  # Final gamma value (used directly if no decay, or as target if decay enabled)
MAX_ITER = 500
INITIAL_STEP_SIZE = 1e-3
MAX_STEP_SIZE = 1e-1
WARMUP_ITERS = 100  # Number of warmup iterations to exclude from timing statistics

# Ablation toggles
USE_GPU = True  # False = CPU, True = GPU
USE_PRECONDITIONING = False  # Jacobi preconditioning
USE_GAMMA_DECAY = False  # Regularization decay

# Gamma decay settings (only used if USE_GAMMA_DECAY=True)
GAMMA_DECAY_STEPS = 35
GAMMA_DECAY_FACTOR = 0.7


def compute_initial_gamma():
    """Compute initial gamma so that we end at FINAL_GAMMA after all decay steps."""
    if not USE_GAMMA_DECAY:
        return FINAL_GAMMA
    num_decays = MAX_ITER // GAMMA_DECAY_STEPS
    return FINAL_GAMMA / (GAMMA_DECAY_FACTOR**num_decays)


def get_output_filename(num_sources, dtype):
    """Generate informative filename based on parameters."""
    dtype_str = str(dtype).replace("torch.", "")
    parts = [
        f"s{num_sources//1_000_000}M",
        f"d{NUM_DESTINATIONS//1_000}K",
        f"sp{TARGET_SPARSITY}",
        f"dtype{dtype_str}",
        f"g{FINAL_GAMMA}",
        f"iter{MAX_ITER}",
    ]
    if USE_PRECONDITIONING:
        parts.append("precon")
    if USE_GAMMA_DECAY:
        parts.append(f"decay{GAMMA_DECAY_FACTOR}x{GAMMA_DECAY_STEPS}")
    return "_".join(parts) + ".csv"


# =============================================================================
# END CONFIG
# =============================================================================


def get_experiment_dirs(base_dir, num_sources, num_destinations, target_sparsity, dtype_str, num_compute_devices=1):
    """Generate cache and output directory names based on experiment parameters."""
    import os

    # Cache dir: shared across all device configurations for same data
    cache_suffix = f"s{num_sources//1_000_000}M_d{num_destinations//1_000}K_sp{target_sparsity}_{dtype_str}"
    cache_dir = os.path.join(base_dir, "cache", cache_suffix)

    # Results dir: includes device count to separate multi-device experiments
    results_suffix = f"{cache_suffix}_dev{num_compute_devices}"
    output_dir = os.path.join(base_dir, "results", results_suffix)

    return cache_dir, output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-GPU benchmark for matching problem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_sources", type=int, default=NUM_SOURCES, help="Number of sources in the matching problem"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Floating point precision",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=BASE_DIR,
        help="Base directory for cache and results (subdirs created automatically based on params)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=MAX_ITER,
        help="Maximum number of iterations for the solver",
    )
    parser.add_argument(
        "--batching",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable batching for projection operations (true/false)",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=WARMUP_ITERS,
        help="Number of warmup iterations to exclude from timing statistics",
    )
    return parser.parse_args()


def run_benchmark(num_sources, dtype=DTYPE, cache_dir=None, output_dir=None, base_dir=BASE_DIR, max_iter=MAX_ITER, batching=False, warmup_iters=WARMUP_ITERS):
    import os

    # Generate cache and output directories based on parameters if not provided
    dtype_str = str(dtype).replace("torch.", "")
    if cache_dir is None or output_dir is None:
        auto_cache_dir, auto_output_dir = get_experiment_dirs(
            base_dir, num_sources, NUM_DESTINATIONS, TARGET_SPARSITY, dtype_str, num_compute_devices=1
        )
        if cache_dir is None:
            cache_dir = auto_cache_dir
        if output_dir is None:
            output_dir = auto_output_dir

    device = "cuda:0" if USE_GPU else "cpu"
    rng = None  # np.random.default_rng(SEED)
    initial_gamma = compute_initial_gamma()

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 60)
    print("CONFIG")
    print("=" * 60)
    print(f"  Data: {num_sources} sources x {NUM_DESTINATIONS} destinations")
    print(f"  Sparsity: {TARGET_SPARSITY}")
    print(f"  Seed: {SEED}")
    print(f"  Dtype: {dtype}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Device: {device}")
    print(f"  Preconditioning: {USE_PRECONDITIONING}")
    print(f"  Gamma decay: {USE_GAMMA_DECAY}")
    if USE_GAMMA_DECAY:
        print(f"  Gamma: {initial_gamma} -> {FINAL_GAMMA}, Max iter: {max_iter}")
    else:
        print(f"  Gamma: {FINAL_GAMMA}, Max iter: {max_iter}")
    print("=" * 60)

    # Generate data
    print("\n[1/3] Generating data...")
    t0 = time.time()
    input_args: MatchingInputArgs = generate_synthetic_matching_input_args(
        num_sources=num_sources,
        num_destinations=NUM_DESTINATIONS,
        target_sparsity=TARGET_SPARSITY,
        device=device,
        dtype=dtype,
        cache_dir=cache_dir,
        rng=rng,
    )
    data_time = time.time() - t0
    print(f"      {data_time:.3f}s | NNZ: {input_args.A._nnz()}")

    # Preconditioning
    if USE_PRECONDITIONING:
        print("      Applying preconditioning...")
        jacobi_precondition(input_args.A, input_args.b_vec)

    # Create objective
    print("[2/3] Creating objective...")
    t0 = time.time()
    objective = MatchingSolverDualObjectiveFunction(
        matching_input_args=input_args,
        gamma=initial_gamma,
        batching=batching,
    )
    obj_time = time.time() - t0
    print(f"      {obj_time:.3f}s")

    # Create solver
    print("[3/3] Running solver...")
    solver = AcceleratedGradientDescent(
        max_iter=max_iter,
        gamma=initial_gamma,
        initial_step_size=INITIAL_STEP_SIZE,
        max_step_size=MAX_STEP_SIZE,
        gamma_decay_type="step" if USE_GAMMA_DECAY else None,
        gamma_decay_params=(
            {"decay_steps": GAMMA_DECAY_STEPS, "decay_factor": GAMMA_DECAY_FACTOR} if USE_GAMMA_DECAY else None
        ),
        save_primal=True,
    )

    initial_dual = torch.zeros_like(input_args.b_vec)

    t0 = time.perf_counter()
    result = solver.maximize(objective, initial_dual)
    solve_time = time.perf_counter() - t0

    # Results
    warmup_excluded_times = result.iteration_time_log[warmup_iters:]
    avg_iter_time = sum(warmup_excluded_times) / len(warmup_excluded_times) if warmup_excluded_times else 0
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Solve time: {solve_time:.3f}s ({solve_time/max_iter*1000:.2f} ms/iter)")
    print(f"  Avg iteration time: {avg_iter_time*1000:.2f} ms/iter")
    print(f"  Dual objective: {result.dual_objective:.6f}")
    print(f"  Primal objective: {result.objective_result.primal_objective.item():.6f}")
    print(f"  Reg penalty: {result.objective_result.reg_penalty.item():.6f}")
    max_slack = result.objective_result.max_pos_slack
    max_slack = max_slack.item() if hasattr(max_slack, "item") else max_slack
    print(f"  Max positive slack: {max_slack:.6e}")
    print(f"  Sum positive slack: {result.objective_result.sum_pos_slack.item():.6e}")
    print("=" * 60)

    # Save dual objective and iteration times to CSV
    import os

    filename = get_output_filename(num_sources, dtype)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "dual_objective", "iteration_time"])
        for i, (obj, iter_time) in enumerate(zip(result.dual_objective_log, result.iteration_time_log), 1):
            writer.writerow([i, obj, iter_time])
    print(f"\nDual objective curve saved to: {filepath}")

    return {
        "solve_time": solve_time,
        "avg_iter_time": avg_iter_time,
        "dual_objective": result.dual_objective,
        "reg_penalty": result.objective_result.reg_penalty.item(),
        "max_pos_slack": max_slack,
        "sum_pos_slack": result.objective_result.sum_pos_slack.item(),
        "dual_objective_file": filepath,
    }


if __name__ == "__main__":
    args = parse_args()

    # Convert dtype string to torch.dtype
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[args.dtype]

    run_benchmark(
        num_sources=args.num_sources,
        dtype=dtype,
        base_dir=args.base_dir,
        max_iter=args.max_iter,
        batching=args.batching,
        warmup_iters=args.warmup_iters,
    )

"""
Single-GPU benchmark for matching problem.
Edit the CONFIG section below to change parameters.
"""

import argparse
import csv
import time

import torch
from generate_synthetic_data import generate_synthetic_matching_input_args

from dualip.objectives.matching import MatchingInputArgs, MatchingSolverDualObjectiveFunctionDistributed
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

# Solver parameters (fixed across all runs)
FINAL_GAMMA = 1e-3  # Final gamma value (used directly if no decay, or as target if decay enabled)
MAX_ITER = 1000
INITIAL_STEP_SIZE = 1e-3
MAX_STEP_SIZE = 1e-1

# Ablation toggles
USE_GPU = True  # False = CPU, True = GPU
USE_PRECONDITIONING = False  # Jacobi preconditioning
USE_GAMMA_DECAY = False  # Regularization decay

# Gamma decay settings (only used if USE_GAMMA_DECAY=True)
GAMMA_DECAY_STEPS = 35
GAMMA_DECAY_FACTOR = 0.7

# Compute settings
NUM_COMPUTE_DEVICES = 2
HOST_DEVICE = "cuda:0"


# =============================================================================
# END CONFIG
# =============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Distributed benchmark for matching problem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num_sources",
        type=int,
        default=NUM_SOURCES,
        help="Number of sources in the matching problem",
    )

    parser.add_argument(
        "--num_compute_devices",
        type=int,
        default=NUM_COMPUTE_DEVICES,
        help="Number of compute devices (GPUs) to use",
    )

    parser.add_argument(
        "--host_device",
        type=str,
        default=HOST_DEVICE,
        help="Host device (e.g., 'cuda:0')",
    )

    return parser.parse_args()


def compute_initial_gamma():
    """Compute initial gamma so that we end at FINAL_GAMMA after all decay steps."""
    if not USE_GAMMA_DECAY:
        return FINAL_GAMMA
    num_decays = MAX_ITER // GAMMA_DECAY_STEPS
    return FINAL_GAMMA / (GAMMA_DECAY_FACTOR**num_decays)


def get_output_filename(num_sources):
    """Generate informative filename based on parameters."""
    parts = [
        f"s{num_sources//1_000_000}M",
        f"d{NUM_DESTINATIONS//1_000}K",
        f"sp{TARGET_SPARSITY}",
        f"g{FINAL_GAMMA}",
        f"iter{MAX_ITER}",
    ]
    if USE_PRECONDITIONING:
        parts.append("precon")
    if USE_GAMMA_DECAY:
        parts.append(f"decay{GAMMA_DECAY_FACTOR}x{GAMMA_DECAY_STEPS}")
    return "_".join(parts) + ".csv"


def run_benchmark(num_sources, num_compute_devices, host_device):
    compute_devices = [f"cuda:{i}" for i in range(num_compute_devices)]
    rng = None  # np.random.default_rng(SEED)
    initial_gamma = compute_initial_gamma()

    print("=" * 60)
    print("CONFIG")
    print("=" * 60)
    print(f"  Data: {num_sources} sources x {NUM_DESTINATIONS} destinations")
    print(f"  Sparsity: {TARGET_SPARSITY}")
    print(f"  Seed: {SEED}")
    print(f"  Device: {host_device}")
    print(f"  Num compute devices: {num_compute_devices}")
    print(f"  Preconditioning: {USE_PRECONDITIONING}")
    print(f"  Gamma decay: {USE_GAMMA_DECAY}")
    if USE_GAMMA_DECAY:
        print(f"  Gamma: {initial_gamma} -> {FINAL_GAMMA}, Max iter: {MAX_ITER}")
    else:
        print(f"  Gamma: {FINAL_GAMMA}, Max iter: {MAX_ITER}")
    print("=" * 60)

    # Generate data
    print("\n[1/3] Generating data...")
    t0 = time.time()
    input_args: MatchingInputArgs = generate_synthetic_matching_input_args(
        num_sources=num_sources,
        num_destinations=NUM_DESTINATIONS,
        target_sparsity=TARGET_SPARSITY,
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
    objective = MatchingSolverDualObjectiveFunctionDistributed(
        matching_input_args=input_args,
        gamma=initial_gamma,
        host_device=host_device,
        compute_devices=compute_devices,
    )

    obj_time = time.time() - t0
    print(f"      {obj_time:.3f}s")

    # Create solver
    print("[3/3] Running solver...")
    save_primal = False
    solver = AcceleratedGradientDescent(
        max_iter=MAX_ITER,
        gamma=initial_gamma,
        initial_step_size=INITIAL_STEP_SIZE,
        max_step_size=MAX_STEP_SIZE,
        gamma_decay_type="step" if USE_GAMMA_DECAY else None,
        gamma_decay_params=(
            {"decay_steps": GAMMA_DECAY_STEPS, "decay_factor": GAMMA_DECAY_FACTOR} if USE_GAMMA_DECAY else None
        ),
        save_primal=save_primal,
    )

    initial_dual = torch.zeros_like(input_args.b_vec).to(host_device)

    t0 = time.time()
    result = solver.maximize(objective, initial_dual)
    solve_time = time.time() - t0

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Solve time: {solve_time:.3f}s ({solve_time/MAX_ITER*1000:.2f} ms/iter)")
    print(f"  Dual objective: {result.dual_objective:.6f}")
    if save_primal:
        print(f"  Primal objective: {result.objective_result.primal_objective.item():.6f}")
    print(f"  Reg penalty: {result.objective_result.reg_penalty.item():.6f}")
    max_slack = result.objective_result.max_pos_slack
    max_slack = max_slack.item() if hasattr(max_slack, "item") else max_slack
    print(f"  Max positive slack: {max_slack:.6e}")
    print(f"  Sum positive slack: {result.objective_result.sum_pos_slack.item():.6e}")
    print("=" * 60)

    # Save dual objective curve to CSV
    filename = get_output_filename(num_sources)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "dual_objective"])
        for i, obj in enumerate(result.dual_objective_log, 1):
            writer.writerow([i, obj])
    print(f"\nDual objective curve saved to: {filename}")

    return {
        "solve_time": solve_time,
        "dual_objective": result.dual_objective,
        "reg_penalty": result.objective_result.reg_penalty.item(),
        "max_pos_slack": max_slack,
        "sum_pos_slack": result.objective_result.sum_pos_slack.item(),
    }


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        num_sources=args.num_sources,
        num_compute_devices=args.num_compute_devices,
        host_device=args.host_device,
    )

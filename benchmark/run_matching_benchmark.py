"""
Single-GPU benchmark for matching problem.
Edit the CONFIG section below to change parameters.
"""

import argparse
import json
import time

import config
import torch
from benchmark_utils import generate_benchmark_data, get_output_filename, print_config, print_results, save_dual_curve

from dualip.objectives.matching import MatchingSolverDualObjectiveFunction
from dualip.optimizers.agd import AcceleratedGradientDescent

# =============================================================================
# CONFIG - Edit these values
# =============================================================================

# Ablation toggles
USE_GPU = True  # False = CPU, True = GPU
USE_GAMMA_DECAY = False  # Regularization decay

# Gamma settings
FINAL_GAMMA = 1e-3  # Final gamma value (used directly if no decay, or as target if decay enabled)

# Gamma decay settings (only used if USE_GAMMA_DECAY=True)
GAMMA_DECAY_STEPS = 35
GAMMA_DECAY_FACTOR = 0.7


def compute_initial_gamma():
    """Compute initial gamma so that we end at FINAL_GAMMA after all decay steps."""
    if not USE_GAMMA_DECAY:
        return FINAL_GAMMA
    num_decays = config.MAX_ITER // GAMMA_DECAY_STEPS
    return FINAL_GAMMA / (GAMMA_DECAY_FACTOR**num_decays)


# =============================================================================
# END CONFIG
# =============================================================================


def run_benchmark(cache_dir: str | None = None):
    device = torch.device("cuda:0" if USE_GPU else "cpu")
    initial_gamma = compute_initial_gamma()

    # Print configuration
    print_config(
        num_sources=config.NUM_SOURCES,
        num_destinations=config.NUM_DESTINATIONS,
        target_sparsity=config.TARGET_SPARSITY,
        seed=config.SEED,
        gamma=FINAL_GAMMA,
        max_iter=config.MAX_ITER,
        use_preconditioning=config.USE_PRECONDITIONING,
        device=device,
        use_gamma_decay=USE_GAMMA_DECAY,
        initial_gamma=initial_gamma if USE_GAMMA_DECAY else None,
        final_gamma=FINAL_GAMMA if USE_GAMMA_DECAY else None,
    )

    # Generate data
    print("\n[1/3] Generating data...")
    input_args, data_time = generate_benchmark_data(
        num_sources=config.NUM_SOURCES,
        num_destinations=config.NUM_DESTINATIONS,
        target_sparsity=config.TARGET_SPARSITY,
        device=device,
        dtype=config.DTYPE,
        seed=config.SEED,
        use_preconditioning=config.USE_PRECONDITIONING,
        cache_dir=cache_dir,
    )

    # Create objective
    print("[2/3] Creating objective...")
    t0 = time.perf_counter()
    objective = MatchingSolverDualObjectiveFunction(
        matching_input_args=input_args,
        gamma=initial_gamma,
        batching=config.BATCHING,
    )
    obj_time = time.perf_counter() - t0
    print(f"      {obj_time:.3f}s")

    # Create solver
    print("[3/3] Running solver...")
    solver = AcceleratedGradientDescent(
        max_iter=config.MAX_ITER,
        gamma=initial_gamma,
        initial_step_size=config.INITIAL_STEP_SIZE,
        max_step_size=config.MAX_STEP_SIZE,
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

    # Print results
    print_results(result, solve_time, config.MAX_ITER)

    # Save dual objective curve to CSV
    filename = get_output_filename(
        num_sources=config.NUM_SOURCES,
        num_destinations=config.NUM_DESTINATIONS,
        target_sparsity=config.TARGET_SPARSITY,
        gamma=FINAL_GAMMA,
        max_iter=config.MAX_ITER,
        use_preconditioning=config.USE_PRECONDITIONING,
        gamma_decay_factor=GAMMA_DECAY_FACTOR if USE_GAMMA_DECAY else None,
        gamma_decay_steps=GAMMA_DECAY_STEPS if USE_GAMMA_DECAY else None,
    )
    save_dual_curve(result, filename)

    # Return metrics dictionary
    metrics = {
        "num_gpus": 1,
        "num_sources": config.NUM_SOURCES,
        "num_destinations": config.NUM_DESTINATIONS,
        "target_sparsity": config.TARGET_SPARSITY,
        "max_iter": config.MAX_ITER,
        "solve_time": solve_time,
        "dual_objective": float(result.dual_objective),
        "primal_objective": (
            float(result.objective_result.primal_objective.item())
            if result.objective_result.primal_objective is not None
            else None
        ),
        "reg_penalty": float(result.objective_result.reg_penalty.item()),
        "max_pos_slack": float(
            result.objective_result.max_pos_slack.item()
            if hasattr(result.objective_result.max_pos_slack, "item")
            else result.objective_result.max_pos_slack
        ),
        "sum_pos_slack": float(result.objective_result.sum_pos_slack.item()),
    }

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-GPU benchmark for matching problem")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for cached data (default: ./benchmark_data)",
    )
    parser.add_argument(
        "--num-sources",
        type=int,
        default=None,
        help=f"Number of sources (default: {config.NUM_SOURCES})",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help=f"Maximum iterations (default: {config.MAX_ITER})",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Save metrics to JSON file",
    )
    args = parser.parse_args()

    # Override config if specified
    if args.num_sources is not None:
        config.NUM_SOURCES = args.num_sources
    if args.max_iter is not None:
        config.MAX_ITER = args.max_iter

    metrics = run_benchmark(cache_dir=args.cache_dir)

    # Save metrics to JSON if requested
    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n📊 Metrics saved to {args.json_output}")

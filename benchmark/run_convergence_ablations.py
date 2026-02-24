"""
Run convergence ablation benchmarks for preconditioning and gamma decay.

Produces 4 CSV files with dual objective convergence curves:
  1. No preconditioning (γ=1e-3)
  2. With preconditioning (γ=1e-3)
  3. No gamma decay (γ=1e-2)
  4. With gamma decay (initial γ=0.16, halving every 25 iters toward γ=0.01)

Usage:
    python run_convergence_ablations.py [--cache-dir DIR] [--output-dir DIR] [--ablation precon|decay|all]
"""

import argparse
import os
import time

import config
import torch
from benchmark_utils import generate_benchmark_data, save_dual_curve

from dualip.objectives.matching import MatchingSolverDualObjectiveFunction
from dualip.optimizers.agd import AcceleratedGradientDescent


def run_single(
    label: str,
    gamma: float,
    use_preconditioning: bool,
    use_gamma_decay: bool,
    gamma_decay_factor: float | None = None,
    gamma_decay_steps: int | None = None,
    gamma_decay_min: float | None = None,
    cache_dir: str | None = None,
    output_dir: str = ".",
):
    """Run a single benchmark configuration and save the convergence curve."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  gamma={gamma}, precon={use_preconditioning}, decay={use_gamma_decay}")
    if use_gamma_decay:
        print(f"  decay_factor={gamma_decay_factor}, decay_steps={gamma_decay_steps}")
    print(f"  device={device}")
    print(f"{'='*60}")

    # Generate data
    input_args, _ = generate_benchmark_data(
        num_sources=config.NUM_SOURCES,
        num_destinations=config.NUM_DESTINATIONS,
        target_sparsity=config.TARGET_SPARSITY,
        device=device,
        dtype=config.DTYPE,
        seed=config.SEED,
        use_preconditioning=use_preconditioning,
        cache_dir=cache_dir,
    )

    # Create objective
    objective = MatchingSolverDualObjectiveFunction(
        matching_input_args=input_args,
        gamma=gamma,
        batching=config.BATCHING,
    )

    # Create solver
    solver = AcceleratedGradientDescent(
        max_iter=config.MAX_ITER,
        gamma=gamma,
        initial_step_size=config.INITIAL_STEP_SIZE,
        max_step_size=config.MAX_STEP_SIZE,
        gamma_decay_type="step" if use_gamma_decay else None,
        gamma_decay_params=(
            {
                "decay_steps": gamma_decay_steps,
                "decay_factor": gamma_decay_factor,
                **({"min_gamma": gamma_decay_min} if gamma_decay_min is not None else {}),
            }
            if use_gamma_decay
            else None
        ),
        save_primal=True,
    )

    initial_dual = torch.zeros_like(input_args.b_vec)

    t0 = time.perf_counter()
    result = solver.maximize(objective, initial_dual)
    solve_time = time.perf_counter() - t0

    print(f"\n  Solve time: {solve_time:.1f}s")
    print(f"  Final dual objective: {result.dual_objective:.2f}")

    # Save convergence curve
    filename = os.path.join(output_dir, f"{label}.csv")
    save_dual_curve(result, filename)
    return filename


def run_precon_ablation(cache_dir: str | None = None, output_dir: str = "."):
    """Run preconditioning ablation (Figure 1)."""
    gamma = 1e-3

    run_single(
        label="no_precon_dual_objective",
        gamma=gamma,
        use_preconditioning=False,
        use_gamma_decay=False,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )
    run_single(
        label="precon_dual_objective",
        gamma=gamma,
        use_preconditioning=True,
        use_gamma_decay=False,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )


def run_decay_ablation(cache_dir: str | None = None, output_dir: str = "."):
    """Run gamma decay ablation (Figure 2)."""
    # No decay baseline: constant γ=1e-2
    run_single(
        label="no_gamma_decay_dual_objective",
        gamma=1e-2,
        use_preconditioning=False,
        use_gamma_decay=False,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )
    # With decay: start at γ=0.16, halve every 25 iterations, floor at 0.01
    run_single(
        label="gamma_decay_dual_objective",
        gamma=0.16,
        use_preconditioning=False,
        use_gamma_decay=True,
        gamma_decay_factor=0.5,
        gamma_decay_steps=25,
        gamma_decay_min=0.01,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run convergence ablation benchmarks")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory for cached data")
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Directory for output CSV files (default: cwd)"
    )
    parser.add_argument(
        "--ablation",
        choices=["precon", "decay", "all"],
        default="all",
        help="Which ablation to run (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.ablation in ("precon", "all"):
        print("\n" + "#" * 60)
        print("# PRECONDITIONING ABLATION (Figure 1)")
        print("#" * 60)
        run_precon_ablation(cache_dir=args.cache_dir, output_dir=args.output_dir)

    if args.ablation in ("decay", "all"):
        print("\n" + "#" * 60)
        print("# GAMMA DECAY ABLATION (Figure 2)")
        print("#" * 60)
        run_decay_ablation(cache_dir=args.cache_dir, output_dir=args.output_dir)

    print("\nAll ablations complete. Run plot_convergence.py to generate figures.")

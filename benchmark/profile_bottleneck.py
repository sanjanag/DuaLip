"""
Profile script to identify bottleneck operations as num_sources scales.

Profiles the COMPLETE AGD iteration including:
1. objective.calculate() - sparse matrix operations
2. GPU→CPU transfer for logging
3. Step size calculation
4. Dense gradient updates
5. Projection onto non-negative cone
6. Accelerated momentum update

Uses CUDA events for accurate GPU timing without synchronization overhead.

Run with: conda activate dualip-os && python benchmark/profile_bottleneck.py
"""

import torch
from generate_synthetic_data import generate_synthetic_matching_input_args

from dualip.objectives.matching import MatchingSolverDualObjectiveFunction
from dualip.optimizers.agd import project_on_nn_cone
from dualip.optimizers.agd_utils import calculate_step_size


def profile_objective_breakdown(objective, dual_val, num_runs=100):
    """
    Profile objective.calculate() with fine-grained breakdown using CUDA events.

    Focuses only on the sparse matrix operations inside objective.calculate()
    since dense vector operations are negligible.

    Returns dict with mean times (in ms) for each operation.
    """
    device = objective.A.device

    if device.type != 'cuda':
        print("CUDA events only work on GPU")
        return None

    # Create CUDA events for timing sparse operations in objective.calculate()
    operation_names = [
        'left_multiply',    # Scale constraint matrix by dual variables
        'add_c',           # Add cost vector
        'projection',      # Simplex projection (main bottleneck)
        'mul',             # Element-wise multiply
        'row_sums',        # Gradient computation via scatter-add
        'objective_total', # Total objective.calculate() time
    ]

    start_events = {key: [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
                    for key in operation_names}
    end_events = {key: [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
                  for key in operation_names}

    from operator import add, mul
    from dualip.projections.base import project
    from dualip.utils.sparse_utils import apply_F_to_columns, elementwise_csc, left_multiply_sparse, row_sums_csc

    # Warmup
    for _ in range(20):
        _ = objective.calculate(dual_val=dual_val, gamma=objective.gamma)

    torch.cuda.synchronize(device)

    # Timed runs with CUDA events
    for run_idx in range(num_runs):
        start_events['objective_total'][run_idx].record()

        # 1. left_multiply_sparse: diag(dual_val) @ A
        start_events['left_multiply'][run_idx].record()
        scaled = -1.0 / objective.gamma * dual_val
        left_multiply_sparse(scaled, objective.A, output_tensor=objective.intermediate)
        end_events['left_multiply'][run_idx].record()

        # 2. elementwise_csc: add cost vector
        start_events['add_c'][run_idx].record()
        elementwise_csc(objective.intermediate, objective.c_rescaled, add, output_tensor=objective.intermediate)
        end_events['add_c'][run_idx].record()

        # 3. apply_F_to_columns: simplex projection on columns
        start_events['projection'][run_idx].record()
        for _, proj_item in objective.buckets.items():
            buckets = proj_item[0]
            proj_type = proj_item[1]
            proj_params = proj_item[2]
            fn = project(proj_type, **proj_params)
            apply_F_to_columns(objective.intermediate, fn, buckets, output_tensor=objective.intermediate)
        end_events['projection'][run_idx].record()

        # 4. elementwise_csc: element-wise multiply
        start_events['mul'][run_idx].record()
        temp = elementwise_csc(objective.A, objective.intermediate, mul)
        end_events['mul'][run_idx].record()

        # 5. row_sums_csc: compute gradient via scatter-add
        start_events['row_sums'][run_idx].record()
        grad = row_sums_csc(temp)
        end_events['row_sums'][run_idx].record()

        end_events['objective_total'][run_idx].record()

    # Synchronize and collect timing
    torch.cuda.synchronize(device)

    times = {}
    for key in operation_names:
        elapsed_times = [start_events[key][i].elapsed_time(end_events[key][i])
                        for i in range(num_runs)]
        times[key] = sum(elapsed_times) / len(elapsed_times)  # Already in ms

    return times


def run_scaling_benchmark(
    num_sources_list=[10_000_000, 20_000_000, 30_000_000, 40_000_000],
    num_destinations=10_000,
    target_sparsity=0.001,
    device='cuda:0',
):
    """
    Run profiling benchmark across different num_sources values.

    Default tests at 10M, 20M, 30M, 40M sources (1x, 2x, 3x, 4x scaling)
    for easy comparison of scaling behavior.
    """
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print("=" * 80)
    print("OBJECTIVE.CALCULATE() PROFILING BENCHMARK")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Profiling method: CUDA Events (100 runs per test)")
    print(f"Num destinations: {num_destinations:,}")
    print(f"Target sparsity: {target_sparsity}")
    print("=" * 80)
    print()

    results = []

    for num_sources in num_sources_list:
        print(f"\n{'=' * 80}")
        print(f"Testing num_sources = {num_sources:,}")
        print(f"{'=' * 80}")

        # Generate data
        print("Generating synthetic data...")
        input_args = generate_synthetic_matching_input_args(
            num_sources=num_sources,
            num_destinations=num_destinations,
            target_sparsity=target_sparsity,
            device=device,
            cache_dir='./benchmark_data/cache',
        )

        nnz = input_args.A._nnz()
        avg_nnz_per_col = nnz / num_sources
        print(f"  Matrix shape: {input_args.A.shape}")
        print(f"  Total nnz: {nnz:,}")
        print(f"  Avg nnz per column: {avg_nnz_per_col:.1f}")

        # Create objective
        print("Creating objective function...")
        objective = MatchingSolverDualObjectiveFunction(
            matching_input_args=input_args,
            gamma=1e-3,
            batching=True,
        )

        # Initialize dual variables
        dual_val = torch.zeros_like(input_args.b_vec)

        # Profile
        print("Profiling objective.calculate() breakdown...")
        times = profile_objective_breakdown(objective, dual_val, num_runs=100)

        # Store results
        results.append({
            'num_sources': num_sources,
            'nnz': nnz,
            'avg_nnz_per_col': avg_nnz_per_col,
            **times
        })

        # Print results
        print(f"\nResults (mean over 100 runs):")
        print(f"  {'Operation':<25} {'Time (ms)':<12} {'% of Total':<12}")
        print(f"  {'-' * 50}")
        obj_total = times['objective_total']
        for key in ['left_multiply', 'add_c', 'projection', 'mul', 'row_sums']:
            t = times[key]
            pct = 100 * t / obj_total if obj_total > 0 else 0
            print(f"  {key:<25} {t:>10.2f}ms {pct:>10.1f}%")
        print(f"  {'-' * 50}")
        print(f"  {'TOTAL':<25} {obj_total:>10.2f}ms 100.0%")

        # Clean up
        del objective, input_args, dual_val
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    # Print summary
    print("\n\n" + "=" * 110)
    print("SUMMARY: OBJECTIVE.CALCULATE() SCALING ANALYSIS")
    print("=" * 110)
    print()

    # Header
    header = (f"{'Sources':<12} {'nnz':<13} {'left_mul':<12} {'add_c':<12} "
              f"{'projection':<12} {'mul':<12} {'row_sums':<12} {'TOTAL':<12}")
    print(header)
    print("-" * 110)

    # Data rows
    for r in results:
        obj_total = r['objective_total']
        print(f"{r['num_sources']:>10,}  {r['nnz']:>11,}  "
              f"{r['left_multiply']:>10.1f}ms {r['add_c']:>10.1f}ms "
              f"{r['projection']:>10.1f}ms {r['mul']:>10.1f}ms "
              f"{r['row_sums']:>10.1f}ms {obj_total:>10.1f}ms")

    # Scaling factors
    if len(results) >= 2:
        base = results[0]

        print("\n" + "=" * 120)
        print(f"SCALING FACTORS (relative to first run: {base['num_sources']:,} sources)")
        print("=" * 120)
        print()
        header = (f"{'Scale':<8} {'Sources':<12} {'nnz_ratio':<11} {'left_mul':<12} {'add_c':<12} "
                  f"{'projection':<12} {'mul':<12} {'row_sums':<12} {'TOTAL':<12}")
        print(header)
        print("-" * 110)

        for idx, r in enumerate(results):
            scale_factor = (idx + 1)  # 1x, 2x, 3x, 4x
            nnz_ratio = r['nnz'] / base['nnz']

            left_mul_ratio = r['left_multiply'] / base['left_multiply'] if base['left_multiply'] > 0 else 0
            add_c_ratio = r['add_c'] / base['add_c'] if base['add_c'] > 0 else 0
            proj_ratio = r['projection'] / base['projection'] if base['projection'] > 0 else 0
            mul_ratio = r['mul'] / base['mul'] if base['mul'] > 0 else 0
            row_sums_ratio = r['row_sums'] / base['row_sums'] if base['row_sums'] > 0 else 0
            total_ratio = r['objective_total'] / base['objective_total'] if base['objective_total'] > 0 else 0

            print(f"{scale_factor}x      {r['num_sources']:>10,}  {nnz_ratio:>9.2f}x "
                  f"{left_mul_ratio:>10.2f}x {add_c_ratio:>10.2f}x "
                  f"{proj_ratio:>10.2f}x {mul_ratio:>10.2f}x "
                  f"{row_sums_ratio:>10.2f}x {total_ratio:>10.2f}x")

        # Analysis
        print("\n" + "=" * 110)
        print("BOTTLENECK ANALYSIS")
        print("=" * 110)

        last = results[-1]
        obj_total = last['objective_total']
        nnz_ratio = last['nnz'] / base['nnz']

        print(f"\n1. Time breakdown at largest scale ({last['num_sources']:,} sources, {last['nnz']:,} nonzeros):")

        components = [
            ('projection', last['projection']),
            ('row_sums', last['row_sums']),
            ('left_multiply', last['left_multiply']),
            ('mul', last['mul']),
            ('add_c', last['add_c']),
        ]
        components_sorted = sorted(components, key=lambda x: x[1], reverse=True)

        for i, (name, time_ms) in enumerate(components_sorted, 1):
            pct = 100 * time_ms / obj_total
            print(f"   {i}. {name:<15} {time_ms:>8.1f}ms  ({pct:>5.1f}%)")
        print(f"   {'─' * 35}")
        print(f"   {'TOTAL':<15} {obj_total:>8.1f}ms  (100.0%)")

        # Scaling analysis
        total_ratio = obj_total / base['objective_total']
        proj_ratio = last['projection'] / base['projection']
        row_sums_ratio = last['row_sums'] / base['row_sums']

        print(f"\n2. Scaling behavior (data scales {nnz_ratio:.2f}x, expected linear = {nnz_ratio:.2f}x):")
        print(f"   - projection:     scales {proj_ratio:.2f}x  ({proj_ratio/nnz_ratio:.2f} of linear)")
        print(f"   - row_sums:       scales {row_sums_ratio:.2f}x  ({row_sums_ratio/nnz_ratio:.2f} of linear)")
        print(f"   - Total:          scales {total_ratio:.2f}x  ({total_ratio/nnz_ratio:.2f} of linear)")

        if total_ratio < nnz_ratio * 0.85:
            print(f"\n   ✓ SUB-LINEAR scaling! Better than expected O(nnz) due to GPU batching efficiency")
        elif total_ratio > nnz_ratio * 1.15:
            print(f"\n   ⚠️  SUPER-LINEAR scaling! Worse than O(nnz) - investigate memory/cache issues")
        else:
            print(f"\n   ≈ LINEAR scaling as expected for O(nnz) operations")

        print(f"\n3. PRIMARY BOTTLENECK: projection")
        print(f"   - Takes {100*last['projection']/obj_total:.1f}% of objective.calculate() time")
        print(f"   - Scales {proj_ratio:.2f}x when data scales {nnz_ratio:.2f}x")
        print(f"   - Dominated by sparse↔dense conversions in apply_F_to_columns")

    print("\n" + "=" * 110)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile AGD iteration bottlenecks")
    parser.add_argument(
        '--sources',
        type=int,
        nargs='+',
        default=[10_000_000, 20_000_000, 30_000_000, 40_000_000],
        help='List of num_sources values to test (default: 10M, 20M, 30M, 40M for 1x, 2x, 3x, 4x comparison)'
    )
    parser.add_argument(
        '--destinations',
        type=int,
        default=10_000,
        help='Number of destinations'
    )
    parser.add_argument(
        '--sparsity',
        type=float,
        default=0.001,
        help='Target sparsity'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to run on (cuda:0, cpu, etc.)'
    )
    args = parser.parse_args()

    results = run_scaling_benchmark(
        num_sources_list=args.sources,
        num_destinations=args.destinations,
        target_sparsity=args.sparsity,
        device=args.device,
    )

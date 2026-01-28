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


def profile_full_agd_iteration(objective, dual_val, beta, num_runs=50):
    """
    Profile the COMPLETE AGD iteration using CUDA events.

    This profiles all operations including dense vector updates that happen
    in the main AGD loop, not just objective.calculate().

    Returns dict with mean times (in ms) for each operation.
    """
    device = objective.A.device

    if device.type != 'cuda':
        print("CUDA events only work on GPU, falling back to CPU timing")
        return None

    # Create CUDA events for timing all AGD iteration steps
    operation_names = [
        'left_multiply', 'add_c', 'projection', 'mul', 'row_sums',  # objective.calculate() breakdown
        'objective_total',  # Total time for objective.calculate()
        'cpu_transfer',  # GPU→CPU transfer for logging
        'step_size_calc',  # Step size calculation
        'gradient_step',  # y_new = x + grad * step_size
        'nn_cone_project',  # Project onto non-negative cone
        'momentum_update',  # Accelerated update x = (y_new * a) + (y * b)
        'agd_iteration_total',  # Total AGD iteration
    ]

    start_events = {key: [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
                    for key in operation_names}
    end_events = {key: [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
                  for key in operation_names}

    from operator import add, mul
    from dualip.projections.base import project
    from dualip.utils.sparse_utils import apply_F_to_columns, elementwise_csc, left_multiply_sparse, row_sums_csc

    # Initialize AGD state variables
    x = dual_val.clone()
    y = dual_val.clone()
    grad_history = []
    dual_history = []
    equality_mask = None

    # Warmup
    for _ in range(10):
        _ = objective.calculate(dual_val=x, gamma=objective.gamma)

    torch.cuda.synchronize(device)

    # Timed runs with CUDA events
    for run_idx in range(num_runs):
        start_events['agd_iteration_total'][run_idx].record()

        # ========== PART 1: objective.calculate() breakdown ==========
        start_events['objective_total'][run_idx].record()

        # 1. left_multiply_sparse
        start_events['left_multiply'][run_idx].record()
        scaled = -1.0 / objective.gamma * x
        left_multiply_sparse(scaled, objective.A, output_tensor=objective.intermediate)
        end_events['left_multiply'][run_idx].record()

        # 2. elementwise_csc (add c_rescaled)
        start_events['add_c'][run_idx].record()
        elementwise_csc(objective.intermediate, objective.c_rescaled, add, output_tensor=objective.intermediate)
        end_events['add_c'][run_idx].record()

        # 3. apply_F_to_columns (projection)
        start_events['projection'][run_idx].record()
        for _, proj_item in objective.buckets.items():
            buckets = proj_item[0]
            proj_type = proj_item[1]
            proj_params = proj_item[2]
            fn = project(proj_type, **proj_params)
            apply_F_to_columns(objective.intermediate, fn, buckets, output_tensor=objective.intermediate)
        end_events['projection'][run_idx].record()

        # 4. elementwise_csc (multiply)
        start_events['mul'][run_idx].record()
        temp = elementwise_csc(objective.A, objective.intermediate, mul)
        end_events['mul'][run_idx].record()

        # 5. row_sums_csc (gradient computation)
        start_events['row_sums'][run_idx].record()
        grad = row_sums_csc(temp)
        end_events['row_sums'][run_idx].record()

        end_events['objective_total'][run_idx].record()

        # ========== PART 2: Post-objective operations ==========

        # 6. CPU transfer for logging (mimics dual_obj.cpu().item())
        start_events['cpu_transfer'][run_idx].record()
        # Compute dual objective on GPU
        vals = objective.intermediate.values()
        dual_obj = torch.dot(objective.c.values(), vals)
        # Transfer to CPU
        _ = dual_obj.cpu().item()
        end_events['cpu_transfer'][run_idx].record()

        # 7. Step size calculation
        start_events['step_size_calc'][run_idx].record()
        step_size = calculate_step_size(
            grad, y, grad_history, dual_history,
            initial_step_size=1e-3, max_step_size=1e-1
        )
        end_events['step_size_calc'][run_idx].record()

        # 8. Gradient ascent step (DENSE vector operation)
        start_events['gradient_step'][run_idx].record()
        y_new = x + grad * step_size
        end_events['gradient_step'][run_idx].record()

        # 9. Project onto non-negative cone (DENSE vector operation)
        start_events['nn_cone_project'][run_idx].record()
        y_new = project_on_nn_cone(y_new, equality_mask)
        end_events['nn_cone_project'][run_idx].record()

        # 10. Accelerated momentum update (DENSE vector operations)
        start_events['momentum_update'][run_idx].record()
        x = (y_new * (1.0 - beta)) + (y * beta)
        y = y_new
        end_events['momentum_update'][run_idx].record()

        end_events['agd_iteration_total'][run_idx].record()

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
    print("FULL AGD ITERATION PROFILING BENCHMARK")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Profiling method: CUDA Events")
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

        # Initialize dual variables and AGD parameters
        dual_val = torch.zeros_like(input_args.b_vec)
        beta = 0.5  # AGD momentum parameter

        # Profile
        print("Profiling full AGD iteration...")
        times = profile_full_agd_iteration(objective, dual_val, beta, num_runs=50)

        # Store results
        results.append({
            'num_sources': num_sources,
            'nnz': nnz,
            'avg_nnz_per_col': avg_nnz_per_col,
            **times
        })

        # Print results - organized by category
        print(f"\nResults (mean over 50 runs):")
        print(f"\n{'='*60}")
        print("OBJECTIVE.CALCULATE() BREAKDOWN:")
        print(f"{'='*60}")
        print(f"  {'Operation':<25} {'Time (ms)':<12} {'% of Obj':<12}")
        print(f"  {'-' * 50}")
        obj_total = times['objective_total']
        for key in ['left_multiply', 'add_c', 'projection', 'mul', 'row_sums']:
            t = times[key]
            pct = 100 * t / obj_total if obj_total > 0 else 0
            print(f"  {key:<25} {t:>10.2f}ms {pct:>10.1f}%")
        print(f"  {'-' * 50}")
        print(f"  {'OBJECTIVE TOTAL':<25} {obj_total:>10.2f}ms")

        print(f"\n{'='*60}")
        print("DENSE VECTOR OPERATIONS:")
        print(f"{'='*60}")
        print(f"  {'Operation':<25} {'Time (ms)':<12} {'% of Iter':<12}")
        print(f"  {'-' * 50}")
        iter_total = times['agd_iteration_total']
        for key in ['cpu_transfer', 'step_size_calc', 'gradient_step', 'nn_cone_project', 'momentum_update']:
            t = times[key]
            pct = 100 * t / iter_total if iter_total > 0 else 0
            print(f"  {key:<25} {t:>10.2f}ms {pct:>10.1f}%")
        dense_total = sum(times[k] for k in ['cpu_transfer', 'step_size_calc', 'gradient_step', 'nn_cone_project', 'momentum_update'])
        print(f"  {'-' * 50}")
        print(f"  {'DENSE OPS TOTAL':<25} {dense_total:>10.2f}ms {100*dense_total/iter_total:>10.1f}%")

        print(f"\n{'='*60}")
        print("FULL AGD ITERATION:")
        print(f"{'='*60}")
        print(f"  {'Component':<25} {'Time (ms)':<12} {'% of Iter':<12}")
        print(f"  {'-' * 50}")
        print(f"  {'objective.calculate()':<25} {obj_total:>10.2f}ms {100*obj_total/iter_total:>10.1f}%")
        print(f"  {'Dense vector ops':<25} {dense_total:>10.2f}ms {100*dense_total/iter_total:>10.1f}%")
        print(f"  {'-' * 50}")
        print(f"  {'ITERATION TOTAL':<25} {iter_total:>10.2f}ms 100.0%")

        # Clean up
        del objective, input_args, dual_val
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    # Print summary
    print("\n\n" + "=" * 120)
    print("SUMMARY: SCALING ANALYSIS")
    print("=" * 120)
    print()

    # Header
    header = (f"{'Sources':<12} {'nnz':<12} {'objective':<12} {'dense_ops':<12} "
              f"{'projection':<12} {'row_sums':<12} {'grad_step':<12} {'momentum':<12} {'iter_total':<12}")
    print(header)
    print("-" * 120)

    # Data rows
    for r in results:
        obj_total = r['objective_total']
        dense_total = sum(r[k] for k in ['cpu_transfer', 'step_size_calc', 'gradient_step', 'nn_cone_project', 'momentum_update'])
        print(f"{r['num_sources']:>10,}  {r['nnz']:>10,}  "
              f"{obj_total:>10.1f}ms {dense_total:>10.1f}ms "
              f"{r['projection']:>10.1f}ms {r['row_sums']:>10.1f}ms "
              f"{r['gradient_step']:>10.1f}ms {r['momentum_update']:>10.1f}ms "
              f"{r['agd_iteration_total']:>10.1f}ms")

    # Scaling factors
    if len(results) >= 2:
        print("\n" + "=" * 120)
        print(f"SCALING FACTORS (relative to first run: {base['num_sources']:,} sources)")
        print("=" * 120)
        print()

        base = results[0]
        header = (f"{'Scale':<8} {'Sources':<12} {'nnz_ratio':<12} {'objective':<12} {'dense_ops':<12} "
                  f"{'projection':<12} {'grad_step':<12} {'momentum':<12} {'iter_total':<12}")
        print(header)
        print("-" * 120)

        for idx, r in enumerate(results):
            scale_factor = (idx + 1)  # 1x, 2x, 3x, 4x
            nnz_ratio = r['nnz'] / base['nnz']
            obj_total = r['objective_total']
            base_obj_total = base['objective_total']
            dense_total = sum(r[k] for k in ['cpu_transfer', 'step_size_calc', 'gradient_step', 'nn_cone_project', 'momentum_update'])
            base_dense_total = sum(base[k] for k in ['cpu_transfer', 'step_size_calc', 'gradient_step', 'nn_cone_project', 'momentum_update'])

            obj_ratio = obj_total / base_obj_total if base_obj_total > 0 else 0
            dense_ratio = dense_total / base_dense_total if base_dense_total > 0 else 0
            proj_ratio = r['projection'] / base['projection'] if base['projection'] > 0 else 0
            grad_ratio = r['gradient_step'] / base['gradient_step'] if base['gradient_step'] > 0 else 0
            mom_ratio = r['momentum_update'] / base['momentum_update'] if base['momentum_update'] > 0 else 0
            iter_ratio = r['agd_iteration_total'] / base['agd_iteration_total'] if base['agd_iteration_total'] > 0 else 0

            print(f"{scale_factor}x      {r['num_sources']:>10,}  {nnz_ratio:>10.2f}x "
                  f"{obj_ratio:>10.2f}x {dense_ratio:>10.2f}x "
                  f"{proj_ratio:>10.2f}x {grad_ratio:>10.2f}x "
                  f"{mom_ratio:>10.2f}x {iter_ratio:>10.2f}x")

        # Analysis
        print("\n" + "=" * 120)
        print("BOTTLENECK ANALYSIS")
        print("=" * 120)

        last = results[-1]
        obj_total = last['objective_total']
        dense_total = sum(last[k] for k in ['cpu_transfer', 'step_size_calc', 'gradient_step', 'nn_cone_project', 'momentum_update'])
        iter_total = last['agd_iteration_total']

        print(f"\n1. Time breakdown at largest scale ({last['num_sources']:,} sources):")
        print(f"   - objective.calculate(): {obj_total:.1f}ms ({100*obj_total/iter_total:.1f}%)")
        print(f"   - Dense vector ops:      {dense_total:.1f}ms ({100*dense_total/iter_total:.1f}%)")
        print(f"   - Total iteration:       {iter_total:.1f}ms")

        # Component-level analysis
        components = [
            ('projection', last['projection'], obj_total),
            ('row_sums', last['row_sums'], obj_total),
            ('gradient_step', last['gradient_step'], iter_total),
            ('momentum_update', last['momentum_update'], iter_total),
            ('nn_cone_project', last['nn_cone_project'], iter_total),
        ]

        print(f"\n2. Most expensive individual operations:")
        components_sorted = sorted(components, key=lambda x: x[1], reverse=True)
        for i, (name, time_ms, total) in enumerate(components_sorted[:5], 1):
            pct = 100 * time_ms / iter_total
            print(f"   {i}. {name}: {time_ms:.1f}ms ({pct:.1f}% of iteration)")

        # Scaling analysis
        base_obj_total = base['objective_total']
        base_dense_total = sum(base[k] for k in ['cpu_transfer', 'step_size_calc', 'gradient_step', 'nn_cone_project', 'momentum_update'])

        obj_scale = obj_total / base_obj_total
        dense_scale = dense_total / base_dense_total
        nnz_ratio = last['nnz'] / base['nnz']

        print(f"\n3. Scaling behavior (data scales {nnz_ratio:.2f}x):")
        print(f"   Expected linear scaling: {nnz_ratio:.2f}x")
        print(f"   - objective.calculate(): scales {obj_scale:.2f}x ({obj_scale/nnz_ratio:.2f} of linear)")
        print(f"   - Dense vector ops:      scales {dense_scale:.2f}x ({dense_scale/nnz_ratio:.2f} of linear)")

        if obj_scale < nnz_ratio * 0.8:
            print(f"\n   ✓ objective.calculate() is SUB-LINEAR (good batching efficiency!)")
        if abs(dense_scale - nnz_ratio) < 0.2 * nnz_ratio:
            print(f"   ⚠️  Dense vector ops scale LINEARLY (memory bandwidth bound)")

        print(f"\n4. PRIMARY BOTTLENECK: ", end="")
        if dense_total > obj_total:
            print("Dense vector operations")
            print(f"   - Take {100*dense_total/iter_total:.1f}% of iteration time")
            print(f"   - Scale nearly linearly ({dense_scale:.2f}x)")
            print(f"   - Memory bandwidth bound (hard to optimize)")
        else:
            print("objective.calculate() [projection]")
            print(f"   - Take {100*obj_total/iter_total:.1f}% of iteration time")
            print(f"   - Dominated by projection: {100*last['projection']/obj_total:.1f}% of objective time")
            print(f"   - Scale sub-linearly ({obj_scale:.2f}x) - good batching efficiency")

    print("\n" + "=" * 120)

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

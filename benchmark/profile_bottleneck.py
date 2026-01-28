"""
Profile script to identify bottleneck operations as num_sources scales.

Uses PyTorch's profiler for accurate GPU timing without synchronization overhead.

Run with: conda activate dualip-os && python benchmark/profile_bottleneck.py
"""

import torch
from generate_synthetic_data import generate_synthetic_matching_input_args

from dualip.objectives.matching import MatchingSolverDualObjectiveFunction


def profile_with_torch_profiler(objective, dual_val, num_warmup=5, num_runs=10):
    """
    Profile using torch.profiler for accurate GPU timing.

    Returns dict with mean times (in ms) for each operation.
    """
    # Warmup
    for _ in range(num_warmup):
        _ = objective.calculate(dual_val=dual_val, gamma=objective.gamma)

    # Profile with torch.profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(num_runs):
            with torch.profiler.record_function("AGD_iteration"):
                _ = objective.calculate(dual_val=dual_val, gamma=objective.gamma)

    # Aggregate timing by operation name
    events = prof.key_averages()

    # Map PyTorch operation names to our operation categories
    timing_map = {
        'left_multiply': [],
        'add_c': [],
        'projection': [],
        'mul': [],
        'row_sums': [],
        'other': [],
    }

    for event in events:
        name = event.key
        cuda_time = event.cuda_time_total / num_runs / 1000  # Convert to ms
        cpu_time = event.cpu_time_total / num_runs / 1000

        # Use CUDA time for GPU ops, CPU time otherwise
        time_ms = cuda_time if cuda_time > 0 else cpu_time

        # Categorize operations
        if 'mul' in name.lower() and 'left_multiply' not in name.lower():
            timing_map['mul'].append((name, time_ms))
        elif 'scatter' in name.lower() or 'row_sums' in name.lower():
            timing_map['row_sums'].append((name, time_ms))
        elif 'sort' in name.lower() or 'topk' in name.lower() or 'projection' in name.lower():
            timing_map['projection'].append((name, time_ms))
        elif 'add' in name.lower():
            timing_map['add_c'].append((name, time_ms))
        elif 'index' in name.lower() or 'gather' in name.lower():
            timing_map['other'].append((name, time_ms))
        else:
            timing_map['other'].append((name, time_ms))

    # Sum up times per category
    result = {}
    for category, ops in timing_map.items():
        result[category] = sum(t for _, t in ops)

    result['total'] = sum(result.values())

    return result, events


def profile_with_cuda_events(objective, dual_val, num_runs=100):
    """
    Profile using CUDA events for precise GPU timing.

    Returns dict with mean times (in ms) for each operation.
    """
    device = objective.A.device

    if device.type != 'cuda':
        print("CUDA events only work on GPU, falling back to CPU timing")
        return None

    # Create CUDA events for timing
    start_events = {key: [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
                    for key in ['left_multiply', 'add_c', 'projection', 'mul', 'row_sums', 'total']}
    end_events = {key: [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
                  for key in ['left_multiply', 'add_c', 'projection', 'mul', 'row_sums', 'total']}

    from operator import add, mul
    from dualip.projections.base import project
    from dualip.utils.sparse_utils import apply_F_to_columns, elementwise_csc, left_multiply_sparse, row_sums_csc

    # Warmup
    for _ in range(10):
        _ = objective.calculate(dual_val=dual_val, gamma=objective.gamma)

    torch.cuda.synchronize(device)

    # Timed runs with CUDA events
    for run_idx in range(num_runs):
        start_events['total'][run_idx].record()

        # 1. left_multiply_sparse
        start_events['left_multiply'][run_idx].record()
        scaled = -1.0 / objective.gamma * dual_val
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

        # 5. row_sums_csc
        start_events['row_sums'][run_idx].record()
        grad = row_sums_csc(temp)
        end_events['row_sums'][run_idx].record()

        end_events['total'][run_idx].record()

    # Synchronize and collect timing
    torch.cuda.synchronize(device)

    times = {}
    for key in ['left_multiply', 'add_c', 'projection', 'mul', 'row_sums', 'total']:
        elapsed_times = [start_events[key][i].elapsed_time(end_events[key][i])
                        for i in range(num_runs)]
        times[key] = sum(elapsed_times) / len(elapsed_times)  # Already in ms

    return times


def run_scaling_benchmark(
    num_sources_list=[1_000_000, 5_000_000, 10_000_000, 25_000_000],
    num_destinations=10_000,
    target_sparsity=0.001,
    device='cuda:0',
    use_cuda_events=True,
):
    """
    Run profiling benchmark across different num_sources values.
    """
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
        use_cuda_events = False

    print("=" * 80)
    print("BOTTLENECK PROFILING BENCHMARK")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Profiling method: {'CUDA Events' if use_cuda_events else 'PyTorch Profiler'}")
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
        print("Profiling...")
        if use_cuda_events:
            times = profile_with_cuda_events(objective, dual_val, num_runs=50)
        else:
            times, _ = profile_with_torch_profiler(objective, dual_val, num_warmup=5, num_runs=20)

        # Store results
        results.append({
            'num_sources': num_sources,
            'nnz': nnz,
            'avg_nnz_per_col': avg_nnz_per_col,
            **times
        })

        # Print results
        print(f"\nResults (mean over multiple runs):")
        print(f"  {'Operation':<20} {'Time (ms)':<12} {'% of Total':<12}")
        print(f"  {'-' * 44}")
        total = times['total']
        for key in ['left_multiply', 'add_c', 'projection', 'mul', 'row_sums']:
            t = times[key]
            pct = 100 * t / total if total > 0 else 0
            print(f"  {key:<20} {t:>10.2f}ms {pct:>10.1f}%")
        print(f"  {'-' * 44}")
        print(f"  {'TOTAL':<20} {total:>10.2f}ms")

        # Clean up
        del objective, input_args, dual_val
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    # Print summary
    print("\n\n" + "=" * 100)
    print("SUMMARY: SCALING ANALYSIS")
    print("=" * 100)
    print()

    # Header
    header = (f"{'Sources':<12} {'nnz':<12} {'left_mul':<11} {'add_c':<11} "
              f"{'projection':<11} {'mul':<11} {'row_sums':<11} {'total':<11}")
    print(header)
    print("-" * 100)

    # Data rows
    for r in results:
        print(f"{r['num_sources']:>10,}  {r['nnz']:>10,}  "
              f"{r['left_multiply']:>9.1f}ms {r['add_c']:>9.1f}ms "
              f"{r['projection']:>9.1f}ms {r['mul']:>9.1f}ms "
              f"{r['row_sums']:>9.1f}ms {r['total']:>9.1f}ms")

    # Scaling factors
    if len(results) >= 2:
        print("\n" + "=" * 100)
        print("SCALING FACTORS (relative to first run)")
        print("=" * 100)
        print()

        base = results[0]
        header = (f"{'Sources':<12} {'nnz_ratio':<11} {'left_mul':<11} {'add_c':<11} "
                  f"{'projection':<11} {'mul':<11} {'row_sums':<11} {'total':<11}")
        print(header)
        print("-" * 100)

        for r in results:
            nnz_ratio = r['nnz'] / base['nnz']
            ratios = {
                key: r[key] / base[key] if base[key] > 0 else 0
                for key in ['left_multiply', 'add_c', 'projection', 'mul', 'row_sums', 'total']
            }

            print(f"{r['num_sources']:>10,}  {nnz_ratio:>9.2f}x "
                  f"{ratios['left_multiply']:>9.2f}x {ratios['add_c']:>9.2f}x "
                  f"{ratios['projection']:>9.2f}x {ratios['mul']:>9.2f}x "
                  f"{ratios['row_sums']:>9.2f}x {ratios['total']:>9.2f}x")

        # Analysis
        print("\n" + "=" * 100)
        print("BOTTLENECK ANALYSIS")
        print("=" * 100)

        last = results[-1]

        # Most time-consuming at scale
        time_ranking = sorted(
            [(key, last[key]) for key in ['left_multiply', 'add_c', 'projection', 'mul', 'row_sums']],
            key=lambda x: x[1],
            reverse=True
        )

        print("\n1. Most time-consuming operations at largest scale:")
        for i, (key, time_ms) in enumerate(time_ranking[:3], 1):
            pct = 100 * time_ms / last['total']
            print(f"   {i}. {key}: {time_ms:.1f}ms ({pct:.1f}% of total)")

        # Worst scaling
        scaling_ranking = sorted(
            [(key, last[key] / base[key]) for key in ['left_multiply', 'add_c', 'projection', 'mul', 'row_sums']],
            key=lambda x: x[1],
            reverse=True
        )

        print(f"\n2. Operations with worst scaling (nnz scales {nnz_ratio:.2f}x):")
        for i, (key, scale_factor) in enumerate(scaling_ranking[:3], 1):
            print(f"   {i}. {key}: scales {scale_factor:.2f}x")

        # Bottleneck score (combines time % and scaling)
        bottleneck_scores = {
            key: (last[key] / last['total']) * (last[key] / base[key])
            for key in ['left_multiply', 'add_c', 'projection', 'mul', 'row_sums']
        }
        bottleneck = max(bottleneck_scores.items(), key=lambda x: x[1])

        print(f"\n3. PRIMARY BOTTLENECK: {bottleneck[0]}")
        print(f"   (combines high time % at scale with poor scaling behavior)")
        print(f"   Time at largest scale: {last[bottleneck[0]]:.1f}ms ({100*last[bottleneck[0]]/last['total']:.1f}%)")
        print(f"   Scaling factor: {last[bottleneck[0]] / base[bottleneck[0]]:.2f}x")

    print("\n" + "=" * 100)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile AGD iteration bottlenecks")
    parser.add_argument(
        '--sources',
        type=int,
        nargs='+',
        default=[1_000_000, 5_000_000, 10_000_000, 25_000_000],
        help='List of num_sources values to test'
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
    parser.add_argument(
        '--use-profiler',
        action='store_true',
        help='Use torch.profiler instead of CUDA events'
    )

    args = parser.parse_args()

    results = run_scaling_benchmark(
        num_sources_list=args.sources,
        num_destinations=args.destinations,
        target_sparsity=args.sparsity,
        device=args.device,
        use_cuda_events=not args.use_profiler,
    )

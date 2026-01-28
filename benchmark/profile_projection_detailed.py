"""
Detailed profiling of projection operation bottleneck.

Breaks down the projection timing into fine-grained components:
1. apply_F_to_columns wrapper overhead:
   - Bucket metadata computation
   - Dense block allocation
   - Sparse→Dense gathering
   - Dense→Sparse scattering
2. Simplex projection math:
   - Clamping & feasibility checks
   - Shortcut detection (topk)
   - Sorting
   - Threshold computation
   - Final projection

Run with: conda activate dualip-os && python benchmark/profile_projection_detailed.py
"""

import torch
from operator import add, mul

from generate_synthetic_data import generate_synthetic_matching_input_args
from dualip.objectives.matching import MatchingSolverDualObjectiveFunction
from dualip.projections.base import project


def profile_projection_breakdown(objective, dual_val, num_runs=50):
    """
    Profile projection operation with detailed breakdown using CUDA events.
    """
    device = objective.A.device

    if device.type != 'cuda':
        print("CUDA events only work on GPU")
        return None

    # Import utilities
    from dualip.utils.sparse_utils import elementwise_csc, left_multiply_sparse

    # Warmup
    for _ in range(10):
        _ = objective.calculate(dual_val=dual_val, gamma=objective.gamma)

    torch.cuda.synchronize(device)

    # Create timing events
    operation_names = [
        'setup_intermediate',      # Steps before projection
        'bucket_metadata',         # Computing starts/ends/lengths per bucket
        'allocate_dense',          # Allocating dense [L x K] block
        'sparse_to_dense',         # Gathering sparse values into dense block
        'simplex_math',            # Actual simplex projection computation
        'dense_to_sparse',         # Scattering dense results back to sparse
        'total_projection',        # Total projection time
    ]

    start_events = {key: [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
                    for key in operation_names}
    end_events = {key: [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
                  for key in operation_names}

    # Timed runs
    for run_idx in range(num_runs):
        # Setup: left_multiply + add_c (needed to get to projection)
        start_events['setup_intermediate'][run_idx].record()
        scaled = -1.0 / objective.gamma * dual_val
        left_multiply_sparse(scaled, objective.A, output_tensor=objective.intermediate)
        elementwise_csc(objective.intermediate, objective.c_rescaled, add, output_tensor=objective.intermediate)
        end_events['setup_intermediate'][run_idx].record()

        # Now profile projection in detail
        start_events['total_projection'][run_idx].record()

        # Get sparse matrix data
        M = objective.intermediate
        ccol = M.ccol_indices()
        rowi = M.row_indices()
        vals = M.values()
        new_vals = torch.empty_like(vals)

        # Process each bucket
        for _, proj_item in objective.buckets.items():
            buckets = proj_item[0]
            proj_type = proj_item[1]
            proj_params = proj_item[2]
            fn = project(proj_type, **proj_params)

            for cols in buckets:
                K = cols.numel()
                if K == 0:
                    continue

                # ========== BUCKET METADATA ==========
                start_events['bucket_metadata'][run_idx].record()
                starts = ccol[cols].to(device)
                ends = ccol[cols + 1].to(device)
                lengths = ends - starts
                total = int(lengths.sum().item())

                if total == 0:
                    end_events['bucket_metadata'][run_idx].record()
                    continue

                L = int(lengths.max().item())

                # Compute indices
                prefix = torch.cat([
                    torch.tensor([0], device=device, dtype=lengths.dtype),
                    torch.cumsum(lengths[:-1], dim=0),
                ])
                prefix_rep = prefix.repeat_interleave(lengths)
                idx_in_col = torch.arange(total, device=device) - prefix_rep
                offs = starts.repeat_interleave(lengths)
                flat_indices = offs + idx_in_col
                cols_rep = torch.arange(K, device=device).repeat_interleave(lengths)
                end_events['bucket_metadata'][run_idx].record()

                # ========== ALLOCATE DENSE BLOCK ==========
                start_events['allocate_dense'][run_idx].record()
                block = torch.zeros((L, K), device=device, dtype=vals.dtype)
                end_events['allocate_dense'][run_idx].record()

                # ========== SPARSE → DENSE (GATHER) ==========
                start_events['sparse_to_dense'][run_idx].record()
                block[idx_in_col, cols_rep] = vals[flat_indices]
                end_events['sparse_to_dense'][run_idx].record()

                # ========== SIMPLEX PROJECTION MATH ==========
                start_events['simplex_math'][run_idx].record()
                proj_block = fn(block)
                end_events['simplex_math'][run_idx].record()

                # ========== DENSE → SPARSE (SCATTER) ==========
                start_events['dense_to_sparse'][run_idx].record()
                new_vals[flat_indices] = proj_block[idx_in_col, cols_rep]
                end_events['dense_to_sparse'][run_idx].record()

        end_events['total_projection'][run_idx].record()

    # Synchronize and collect timing
    torch.cuda.synchronize(device)

    times = {}
    for key in operation_names:
        elapsed_times = [start_events[key][i].elapsed_time(end_events[key][i])
                        for i in range(num_runs)]
        times[key] = sum(elapsed_times) / len(elapsed_times)

    return times


def profile_simplex_internals(objective, dual_val, num_runs=50):
    """
    Profile simplex projection internals by instrumenting the simplex code.

    This requires modifying the simplex projection to add timing events.
    For now, we'll just time the overall simplex call and provide estimates.
    """
    device = objective.A.device

    if device.type != 'cuda':
        return None

    from dualip.utils.sparse_utils import elementwise_csc, left_multiply_sparse

    # Warmup
    for _ in range(10):
        _ = objective.calculate(dual_val=dual_val, gamma=objective.gamma)

    torch.cuda.synchronize(device)

    # Create a test block matching typical size
    # From data: L ≈ 10-25 nonzeros per column, K = batch size
    test_sizes = [
        (10, 5000),   # Small columns, large batch
        (25, 5000),   # Medium columns, large batch
        (50, 1000),   # Larger columns, smaller batch
    ]

    results = {}

    for L, K in test_sizes:
        print(f"\n  Testing simplex projection: L={L} rows, K={K} columns")

        # Create synthetic test data
        x = torch.randn(L, K, device=device, dtype=torch.float32) * 0.5 + 0.5
        x = torch.clamp(x, min=0.0)

        # Get projection function
        fn = project("simplex", z=1.0)

        # Time the projection
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

        for run_idx in range(num_runs):
            start_events[run_idx].record()
            _ = fn(x)
            end_events[run_idx].record()

        torch.cuda.synchronize(device)

        elapsed_times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_runs)]
        avg_time = sum(elapsed_times) / len(elapsed_times)

        results[f"simplex_L{L}_K{K}"] = avg_time
        print(f"    Avg time: {avg_time:.3f}ms ({avg_time*1e6/(L*K):.1f}ns per element)")

    return results


def run_detailed_profiling_single(
    num_sources,
    num_destinations,
    target_sparsity,
    device,
):
    """
    Run detailed projection profiling for a single configuration.
    """
    print(f"\n{'=' * 80}")
    print(f"Testing num_sources = {num_sources:,}")
    print(f"{'=' * 80}")

    # Generate data
    print("Generating data...")
    input_args = generate_synthetic_matching_input_args(
        num_sources=num_sources,
        num_destinations=num_destinations,
        target_sparsity=target_sparsity,
        device=device,
        cache_dir='./benchmark_data/cache',
    )

    print(f"  Matrix shape: {input_args.A.shape}")
    print(f"  Total nnz: {input_args.A._nnz():,}")

    # Create objective
    print("Creating objective...")
    objective = MatchingSolverDualObjectiveFunction(
        matching_input_args=input_args,
        gamma=1e-3,
        batching=True,
    )

    dual_val = torch.zeros_like(input_args.b_vec)

    # Profile projection breakdown
    print("Profiling projection breakdown (50 runs)...")
    times = profile_projection_breakdown(objective, dual_val, num_runs=50)

    # Print results for this run
    print(f"\nResults (mean over 50 runs):")
    total_proj = times['total_projection']

    print(f"  {'Operation':<30} {'Time (ms)':<12} {'% of Projection':<15}")
    print(f"  {'-' * 60}")

    operations = [
        ('bucket_metadata', 'Bucket metadata'),
        ('allocate_dense', 'Allocate dense block'),
        ('sparse_to_dense', 'Sparse→Dense gather'),
        ('simplex_math', 'Simplex projection'),
        ('dense_to_sparse', 'Dense→Sparse scatter'),
    ]

    for key, label in operations:
        t = times[key]
        pct = 100 * t / total_proj if total_proj > 0 else 0
        print(f"  {label:<30} {t:>10.2f}ms {pct:>13.1f}%")

    print(f"  {'-' * 60}")
    print(f"  {'TOTAL':<30} {total_proj:>10.2f}ms {'100.0%':>13}")

    memory_ops = times['sparse_to_dense'] + times['dense_to_sparse']
    memory_pct = 100 * memory_ops / total_proj
    print(f"\n  Memory ops (gather+scatter): {memory_ops:.1f}ms ({memory_pct:.1f}%)")

    return times


def run_scaling_profiling(
    num_sources_list=[10_000_000, 20_000_000, 30_000_000, 40_000_000],
    num_destinations=10_000,
    target_sparsity=0.001,
    device='cuda:0',
):
    """
    Run detailed projection profiling across multiple problem sizes.

    Default uses same sizes as objective.calculate() profiling (1x, 2x, 3x, 4x).
    """
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available")
        device = 'cpu'

    print("=" * 80)
    print("DETAILED PROJECTION PROFILING - SCALING ANALYSIS")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Destinations: {num_destinations:,}")
    print(f"Sparsity: {target_sparsity}")
    print(f"Testing sizes: {[f'{s:,}' for s in num_sources_list]}")
    print("=" * 80)

    results = []

    for num_sources in num_sources_list:
        times = run_detailed_profiling_single(
            num_sources=num_sources,
            num_destinations=num_destinations,
            target_sparsity=target_sparsity,
            device=device,
        )

        if times:
            results.append({
                'num_sources': num_sources,
                **times
            })

        # Clean up
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    # Print summary
    if len(results) >= 2:
        print("\n\n" + "=" * 100)
        print("SUMMARY: PROJECTION BREAKDOWN SCALING")
        print("=" * 100)
        print()

        header = (f"{'Sources':<12} {'Metadata':<11} {'Allocate':<11} {'S→D Gather':<12} "
                  f"{'Simplex':<11} {'D→S Scatter':<12} {'TOTAL':<11}")
        print(header)
        print("-" * 100)

        for r in results:
            print(f"{r['num_sources']:>10,}  "
                  f"{r['bucket_metadata']:>9.1f}ms {r['allocate_dense']:>9.1f}ms "
                  f"{r['sparse_to_dense']:>10.1f}ms {r['simplex_math']:>9.1f}ms "
                  f"{r['dense_to_sparse']:>10.1f}ms {r['total_projection']:>9.1f}ms")

        # Scaling analysis
        print("\n" + "=" * 100)
        print(f"SCALING FACTORS (relative to first run: {results[0]['num_sources']:,} sources)")
        print("=" * 100)
        print()

        base = results[0]
        header = (f"{'Scale':<8} {'Sources':<12} {'Metadata':<11} {'Allocate':<11} {'S→D Gather':<12} "
                  f"{'Simplex':<11} {'D→S Scatter':<12} {'TOTAL':<11}")
        print(header)
        print("-" * 100)

        for idx, r in enumerate(results):
            scale_factor = (idx + 1)

            meta_ratio = r['bucket_metadata'] / base['bucket_metadata'] if base['bucket_metadata'] > 0 else 0
            alloc_ratio = r['allocate_dense'] / base['allocate_dense'] if base['allocate_dense'] > 0 else 0
            s2d_ratio = r['sparse_to_dense'] / base['sparse_to_dense'] if base['sparse_to_dense'] > 0 else 0
            simplex_ratio = r['simplex_math'] / base['simplex_math'] if base['simplex_math'] > 0 else 0
            d2s_ratio = r['dense_to_sparse'] / base['dense_to_sparse'] if base['dense_to_sparse'] > 0 else 0
            total_ratio = r['total_projection'] / base['total_projection'] if base['total_projection'] > 0 else 0

            print(f"{scale_factor}x      {r['num_sources']:>10,}  "
                  f"{meta_ratio:>9.2f}x {alloc_ratio:>9.2f}x "
                  f"{s2d_ratio:>10.2f}x {simplex_ratio:>9.2f}x "
                  f"{d2s_ratio:>10.2f}x {total_ratio:>9.2f}x")

        # Analysis
        print("\n" + "=" * 100)
        print("BOTTLENECK IDENTIFICATION")
        print("=" * 100)
        print()

        last = results[-1]
        total = last['total_projection']

        memory_ops = last['sparse_to_dense'] + last['dense_to_sparse']
        memory_pct = 100 * memory_ops / total

        print(f"At largest scale ({last['num_sources']:,} sources):")
        print()
        print(f"1. Memory operations: {memory_ops:.1f}ms ({memory_pct:.1f}%)")
        print(f"   - Sparse→Dense gather: {last['sparse_to_dense']:.1f}ms")
        print(f"   - Dense→Sparse scatter: {last['dense_to_sparse']:.1f}ms")
        print()
        print(f"2. Simplex projection math: {last['simplex_math']:.1f}ms ({100*last['simplex_math']/total:.1f}%)")
        print()
        print(f"3. Overhead: {last['bucket_metadata'] + last['allocate_dense']:.1f}ms ({100*(last['bucket_metadata'] + last['allocate_dense'])/total:.1f}%)")
        print()

        if memory_pct > 55:
            print("⚠️  PRIMARY BOTTLENECK: Memory operations (gather/scatter)")
            print("   → Memory bandwidth bound")
            print("   → Random access patterns hurt cache efficiency")
            print("   → Consider: fused kernels, avoid dense intermediate format")
        elif last['simplex_math'] / total > 0.50:
            print("⚠️  PRIMARY BOTTLENECK: Simplex projection math")
            print("   → Computation bound")
            print("   → Consider: optimized projection algorithm")
        else:
            print("✓ Relatively balanced - no single dominant bottleneck")

    print("\n" + "=" * 100)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detailed projection profiling")
    parser.add_argument(
        '--sources',
        type=int,
        nargs='+',
        default=[10_000_000, 20_000_000, 30_000_000, 40_000_000],
        help='List of num_sources values (default: 10M, 20M, 30M, 40M matching objective profiling)'
    )
    parser.add_argument('--destinations', type=int, default=10_000, help='Number of destinations')
    parser.add_argument('--sparsity', type=float, default=0.001, help='Target sparsity')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')

    args = parser.parse_args()

    results = run_scaling_profiling(
        num_sources_list=args.sources,
        num_destinations=args.destinations,
        target_sparsity=args.sparsity,
        device=args.device,
    )

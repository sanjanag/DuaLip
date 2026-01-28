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


def run_detailed_profiling(
    num_sources=25_000_000,
    num_destinations=10_000,
    target_sparsity=0.001,
    device='cuda:0',
):
    """
    Run detailed projection profiling.
    """
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("=" * 80)
    print("DETAILED PROJECTION PROFILING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Sources: {num_sources:,}")
    print(f"Destinations: {num_destinations:,}")
    print(f"Sparsity: {target_sparsity}")
    print("=" * 80)
    print()

    # Generate data
    print("Generating data...")
    input_args = generate_synthetic_matching_input_args(
        num_sources=num_sources,
        num_destinations=num_destinations,
        target_sparsity=target_sparsity,
        device=device,
        cache_dir='./benchmark_data/cache',
    )

    print(f"Matrix shape: {input_args.A.shape}")
    print(f"Total nnz: {input_args.A._nnz():,}")
    print()

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

    # Print results
    print("\n" + "=" * 80)
    print("PROJECTION BREAKDOWN (mean over 50 runs)")
    print("=" * 80)
    print()

    total_proj = times['total_projection']

    print(f"{'Operation':<30} {'Time (ms)':<12} {'% of Projection':<15}")
    print("-" * 80)

    operations = [
        ('bucket_metadata', 'Bucket metadata setup'),
        ('allocate_dense', 'Allocate dense block'),
        ('sparse_to_dense', 'Sparse→Dense gather'),
        ('simplex_math', 'Simplex projection math'),
        ('dense_to_sparse', 'Dense→Sparse scatter'),
    ]

    for key, label in operations:
        t = times[key]
        pct = 100 * t / total_proj if total_proj > 0 else 0
        print(f"{label:<30} {t:>10.2f}ms {pct:>13.1f}%")

    print("-" * 80)
    print(f"{'TOTAL PROJECTION':<30} {total_proj:>10.2f}ms {'100.0%':>13}")
    print()

    # Analyze results
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    memory_ops = times['sparse_to_dense'] + times['dense_to_sparse']
    memory_pct = 100 * memory_ops / total_proj

    print(f"1. Memory operations (gather + scatter): {memory_ops:.1f}ms ({memory_pct:.1f}%)")
    print(f"   - Sparse→Dense: {times['sparse_to_dense']:.1f}ms")
    print(f"   - Dense→Sparse: {times['dense_to_sparse']:.1f}ms")
    print()

    print(f"2. Simplex projection math: {times['simplex_math']:.1f}ms ({100*times['simplex_math']/total_proj:.1f}%)")
    print()

    print(f"3. Overhead (metadata + allocation): {times['bucket_metadata'] + times['allocate_dense']:.1f}ms")
    print()

    if memory_pct > 60:
        print("⚠️  PRIMARY BOTTLENECK: Memory operations (gather/scatter)")
        print("   → This is memory bandwidth bound")
        print("   → Consider fused kernels to eliminate intermediate dense format")
    elif times['simplex_math'] / total_proj > 0.5:
        print("⚠️  PRIMARY BOTTLENECK: Simplex projection computation")
        print("   → Consider optimizing the projection algorithm")
    else:
        print("✓ No single dominant bottleneck - relatively balanced")

    print()

    # Profile simplex internals
    print("=" * 80)
    print("SIMPLEX PROJECTION MICRO-BENCHMARKS")
    print("=" * 80)
    simplex_results = profile_simplex_internals(objective, dual_val, num_runs=100)

    print("\n" + "=" * 80)
    print()

    return times


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detailed projection profiling")
    parser.add_argument('--sources', type=int, default=25_000_000, help='Number of sources')
    parser.add_argument('--destinations', type=int, default=10_000, help='Number of destinations')
    parser.add_argument('--sparsity', type=float, default=0.001, help='Target sparsity')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')

    args = parser.parse_args()

    results = run_detailed_profiling(
        num_sources=args.sources,
        num_destinations=args.destinations,
        target_sparsity=args.sparsity,
        device=args.device,
    )

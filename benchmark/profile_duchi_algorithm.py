"""
Profile the Duchi simplex projection algorithm to identify bottleneck operations.

This script instruments _duchi_proj to measure time spent in each sub-operation:
1. Clamping and setup
2. Feasibility checks (inequality case)
3. Topk shortcut detection
4. Sort operation
5. Cumulative sum
6. Threshold computation
7. Final projection

Run with: conda activate dualip-os && python benchmark/profile_duchi_algorithm.py
"""

import torch
from generate_synthetic_data import generate_synthetic_matching_input_args

from dualip.objectives.matching import MatchingSolverDualObjectiveFunction


def profile_duchi_breakdown(objective, dual_val, num_runs=50):
    """
    Profile the Duchi simplex projection algorithm with fine-grained breakdown.

    Instruments the projection call to measure time spent in each operation
    within _duchi_proj.

    Returns dict with mean times (in ms) for each sub-operation.
    """
    device = objective.A.device

    if device.type != 'cuda':
        print("CUDA events only work on GPU")
        return None

    # Import what we need to manually replicate the objective.calculate() flow
    from operator import add, mul
    from dualip.projections.simplex import SimplexIneq
    from dualip.utils.sparse_utils import apply_F_to_columns, elementwise_csc, left_multiply_sparse, row_sums_csc

    # Warmup
    for _ in range(10):
        _ = objective.calculate(dual_val=dual_val, gamma=objective.gamma)

    torch.cuda.synchronize(device)

    # We'll accumulate detailed timing across all buckets and runs
    operation_names = [
        'clamp_setup',
        'feasibility_check',
        'topk_shortcut',
        'sort',
        'cumsum',
        'threshold_compute',
        'final_projection',
        'total_duchi',
    ]

    accumulated_times = {key: 0.0 for key in operation_names}
    total_bucket_count = 0

    for run_idx in range(num_runs):
        # Run the full objective calculation up to projection
        scaled = -1.0 / objective.gamma * dual_val
        left_multiply_sparse(scaled, objective.A, output_tensor=objective.intermediate)
        elementwise_csc(objective.intermediate, objective.c_rescaled, add, output_tensor=objective.intermediate)

        # Now manually do projection with detailed timing
        for _, proj_item in objective.buckets.items():
            buckets = proj_item[0]
            proj_params = proj_item[2]

            # Create projection operator
            z = proj_params.get('z', 1.0)
            fn = SimplexIneq(z=z, method='duchi')

            for cols in buckets:
                if len(cols) == 0:
                    continue

                total_bucket_count += 1

                # Extract the columns (this is what apply_F_to_columns does internally)
                cols_tensor = torch.tensor(cols, dtype=torch.long, device=device)
                block = objective.intermediate.index_select(1, cols_tensor).to_dense()

                L, B = block.shape
                tol = 1e-6
                inequality = True  # Matching uses inequality constraint

                # Create CUDA events for this bucket
                events = {}
                for key in operation_names:
                    events[f'{key}_start'] = torch.cuda.Event(enable_timing=True)
                    events[f'{key}_end'] = torch.cuda.Event(enable_timing=True)

                # Start total timing
                events['total_duchi_start'].record()

                # ===== INSTRUMENTED _duchi_proj =====

                w = torch.empty_like(block)

                # 1. Clamp and setup
                events['clamp_setup_start'].record()
                x = torch.clamp(block, min=0.0)
                to_project_mask = torch.zeros(B, dtype=torch.bool, device=device)
                events['clamp_setup_end'].record()

                # 2. Feasibility check (inequality case)
                events['feasibility_check_start'].record()
                if inequality:
                    is_feasible = (x.sum(dim=0) <= z + tol) & (x >= -tol).all(dim=0)
                    w[:, is_feasible] = x[:, is_feasible]
                    infeasible_mask = ~is_feasible
                    if not infeasible_mask.any():
                        events['feasibility_check_end'].record()
                        events['topk_shortcut_start'].record()
                        events['topk_shortcut_end'].record()
                        events['sort_start'].record()
                        events['sort_end'].record()
                        events['cumsum_start'].record()
                        events['cumsum_end'].record()
                        events['threshold_compute_start'].record()
                        events['threshold_compute_end'].record()
                        events['final_projection_start'].record()
                        events['final_projection_end'].record()
                        events['total_duchi_end'].record()
                        torch.cuda.synchronize(device)
                        for key in operation_names:
                            accumulated_times[key] += events[f'{key}_start'].elapsed_time(events[f'{key}_end'])
                        continue
                    to_project_mask[infeasible_mask] = True
                else:
                    to_project_mask.fill_(True)
                events['feasibility_check_end'].record()

                if not to_project_mask.any():
                    events['topk_shortcut_start'].record()
                    events['topk_shortcut_end'].record()
                    events['sort_start'].record()
                    events['sort_end'].record()
                    events['cumsum_start'].record()
                    events['cumsum_end'].record()
                    events['threshold_compute_start'].record()
                    events['threshold_compute_end'].record()
                    events['final_projection_start'].record()
                    events['final_projection_end'].record()
                    events['total_duchi_end'].record()
                    torch.cuda.synchronize(device)
                    for key in operation_names:
                        accumulated_times[key] += events[f'{key}_start'].elapsed_time(events[f'{key}_end'])
                    continue

                # 3. Topk shortcut detection
                events['topk_shortcut_start'].record()
                if L > 1:
                    to_project_indices = to_project_mask.nonzero(as_tuple=True)[0]
                    to_project = x[:, to_project_indices]
                    to_project_normalized = to_project / z
                    vals, indices = torch.topk(to_project_normalized, 2, dim=0)
                    shortcut_mask = (vals[0] - vals[1]) > 1.0

                    if shortcut_mask.any():
                        shortcut_cols_idx = to_project_indices[shortcut_mask]
                        shortcut_max_indices = indices[0, shortcut_mask]
                        solution = torch.zeros(L, shortcut_cols_idx.shape[0], device=device, dtype=block.dtype)
                        solution[shortcut_max_indices, torch.arange(solution.shape[1])] = z
                        w[:, shortcut_cols_idx] = solution
                        to_project_mask[shortcut_cols_idx] = False
                events['topk_shortcut_end'].record()

                if not to_project_mask.any():
                    events['sort_start'].record()
                    events['sort_end'].record()
                    events['cumsum_start'].record()
                    events['cumsum_end'].record()
                    events['threshold_compute_start'].record()
                    events['threshold_compute_end'].record()
                    events['final_projection_start'].record()
                    events['final_projection_end'].record()
                    events['total_duchi_end'].record()
                    torch.cuda.synchronize(device)
                    for key in operation_names:
                        accumulated_times[key] += events[f'{key}_start'].elapsed_time(events[f'{key}_end'])
                    continue

                to_project = x[:, to_project_mask]
                proj_cols_idx = to_project_mask.nonzero(as_tuple=True)[0]

                _, K = to_project.shape
                cols_per_chunk = 10000

                # Initialize sort/cumsum/threshold/projection timing accumulators
                sort_time = 0.0
                cumsum_time = 0.0
                threshold_time = 0.0
                projection_time = 0.0

                for c0 in range(0, K, cols_per_chunk):
                    c1 = min(c0 + cols_per_chunk, K)
                    to_project_sub = to_project[:, c0:c1]
                    proj_cols_idx_sub = proj_cols_idx[c0:c1]

                    # 4. Sort
                    sort_start = torch.cuda.Event(enable_timing=True)
                    sort_end = torch.cuda.Event(enable_timing=True)
                    sort_start.record()
                    u_sorted, _ = to_project_sub.sort(dim=0, descending=True)
                    sort_end.record()
                    torch.cuda.synchronize(device)
                    sort_time += sort_start.elapsed_time(sort_end)

                    # 5. Cumsum
                    cumsum_start = torch.cuda.Event(enable_timing=True)
                    cumsum_end = torch.cuda.Event(enable_timing=True)
                    cumsum_start.record()
                    cumsum_u = u_sorted.cumsum(dim=0)
                    cumsum_end.record()
                    torch.cuda.synchronize(device)
                    cumsum_time += cumsum_start.elapsed_time(cumsum_end)

                    # 6. Threshold computation
                    threshold_start = torch.cuda.Event(enable_timing=True)
                    threshold_end = torch.cuda.Event(enable_timing=True)
                    threshold_start.record()
                    idx = torch.arange(1, L + 1, device=device)
                    idx0 = idx - 1
                    idx_f = idx.to(block.dtype).view(L, 1)
                    cond = u_sorted - (cumsum_u - z) / idx_f > 0
                    mask = cond.to(torch.long) * idx0.view(L, 1)
                    rho = mask.max(dim=0).values
                    col_indices = torch.arange(rho.size(0), device=device)
                    cumsum_at_rho = cumsum_u[rho, col_indices]
                    theta = (cumsum_at_rho - z) / (rho.to(block.dtype) + 1)
                    threshold_end.record()
                    torch.cuda.synchronize(device)
                    threshold_time += threshold_start.elapsed_time(threshold_end)

                    # 7. Final projection
                    projection_start = torch.cuda.Event(enable_timing=True)
                    projection_end = torch.cuda.Event(enable_timing=True)
                    projection_start.record()
                    proj_cols = (to_project_sub - theta.unsqueeze(0)).clamp(min=0)
                    w[:, proj_cols_idx_sub] = proj_cols
                    projection_end.record()
                    torch.cuda.synchronize(device)
                    projection_time += projection_start.elapsed_time(projection_end)

                # Record accumulated chunk times
                events['sort_start'].record()
                events['sort_end'].record()
                torch.cuda.synchronize(device)
                accumulated_times['sort'] += sort_time

                events['cumsum_start'].record()
                events['cumsum_end'].record()
                torch.cuda.synchronize(device)
                accumulated_times['cumsum'] += cumsum_time

                events['threshold_compute_start'].record()
                events['threshold_compute_end'].record()
                torch.cuda.synchronize(device)
                accumulated_times['threshold_compute'] += threshold_time

                events['final_projection_start'].record()
                events['final_projection_end'].record()
                torch.cuda.synchronize(device)
                accumulated_times['final_projection'] += projection_time

                events['total_duchi_end'].record()

                # Synchronize and accumulate non-loop timings
                torch.cuda.synchronize(device)
                accumulated_times['clamp_setup'] += events['clamp_setup_start'].elapsed_time(events['clamp_setup_end'])
                accumulated_times['feasibility_check'] += events['feasibility_check_start'].elapsed_time(events['feasibility_check_end'])
                accumulated_times['topk_shortcut'] += events['topk_shortcut_start'].elapsed_time(events['topk_shortcut_end'])
                accumulated_times['total_duchi'] += events['total_duchi_start'].elapsed_time(events['total_duchi_end'])

    # Average over all runs and buckets
    if total_bucket_count > 0:
        for key in operation_names:
            accumulated_times[key] /= total_bucket_count

    return accumulated_times


def run_duchi_profiling(
    num_sources=25_000_000,
    num_destinations=10_000,
    target_sparsity=0.001,
    device='cuda:0',
):
    """
    Profile the Duchi algorithm breakdown at a specific scale.
    """
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print("=" * 80)
    print("DUCHI SIMPLEX PROJECTION ALGORITHM PROFILING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Profiling method: CUDA Events (50 runs, averaged over all buckets)")
    print(f"Num sources: {num_sources:,}")
    print(f"Num destinations: {num_destinations:,}")
    print(f"Target sparsity: {target_sparsity}")
    print("=" * 80)
    print()

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
    print("Profiling Duchi algorithm breakdown...")
    times = profile_duchi_breakdown(objective, dual_val, num_runs=50)

    # Print results
    print(f"\nResults (mean over 50 runs, averaged across all buckets):")
    print(f"  {'Operation':<25} {'Time (ms)':<12} {'% of Total':<12}")
    print(f"  {'-' * 50}")
    total_time = times['total_duchi']
    for key in ['clamp_setup', 'feasibility_check', 'topk_shortcut', 'sort',
                'cumsum', 'threshold_compute', 'final_projection']:
        t = times[key]
        pct = 100 * t / total_time if total_time > 0 else 0
        print(f"  {key:<25} {t:>10.2f}ms {pct:>10.1f}%")
    print(f"  {'-' * 50}")
    print(f"  {'TOTAL':<25} {total_time:>10.2f}ms 100.0%")

    # Analysis
    print("\n" + "=" * 80)
    print("BOTTLENECK ANALYSIS")
    print("=" * 80)

    components = [
        ('sort', times['sort']),
        ('threshold_compute', times['threshold_compute']),
        ('cumsum', times['cumsum']),
        ('topk_shortcut', times['topk_shortcut']),
        ('final_projection', times['final_projection']),
        ('feasibility_check', times['feasibility_check']),
        ('clamp_setup', times['clamp_setup']),
    ]
    components_sorted = sorted(components, key=lambda x: x[1], reverse=True)

    print(f"\nTime breakdown within Duchi simplex projection:")
    for i, (name, time_ms) in enumerate(components_sorted, 1):
        pct = 100 * time_ms / total_time
        print(f"   {i}. {name:<20} {time_ms:>8.2f}ms  ({pct:>5.1f}%)")

    # Map operations to code lines
    print("\n" + "=" * 80)
    print("CODE LOCATION REFERENCE")
    print("=" * 80)
    print("\nAll operations are in src/dualip/projections/simplex.py (_duchi_proj):")
    print(f"  clamp_setup:        Lines 147-151 (torch.clamp, mask initialization)")
    print(f"  feasibility_check:  Lines 153-159 (early exit if already feasible)")
    print(f"  topk_shortcut:      Lines 166-194 (torch.topk for one-hot detection)")
    print(f"  sort:               Line 208 (to_project_sub.sort())")
    print(f"  cumsum:             Line 211 (u_sorted.cumsum())")
    print(f"  threshold_compute:  Lines 214-228 (rho, theta calculation)")
    print(f"  final_projection:   Lines 231-234 (clamp and scatter)")

    print("\n" + "=" * 80)

    # Clean up
    del objective, input_args, dual_val
    if device.startswith('cuda'):
        torch.cuda.empty_cache()

    return times


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile Duchi simplex projection algorithm")
    parser.add_argument(
        '--sources',
        type=int,
        default=25_000_000,
        help='Number of sources'
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

    results = run_duchi_profiling(
        num_sources=args.sources,
        num_destinations=args.destinations,
        target_sparsity=args.sparsity,
        device=args.device,
    )

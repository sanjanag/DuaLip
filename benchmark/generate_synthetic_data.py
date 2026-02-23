import json
import os

import numpy as np
import torch

from dualip.objectives.matching import MatchingInputArgs
from dualip.projections import create_projection_map

# -------------------------------------------------------------------------
# Disk-backed cache config (via numpy.memmap)
# -------------------------------------------------------------------------

# Default cache directory in benchmark directory
_DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "benchmark_data")

# Optional in-process cache to avoid re-mmapping within one process.
# Maps (cache_dir, cache_key) -> (ccol_indices, row_indices, a_values, c_values, b_vec_np)
_in_process_cache = {}


# -------------------------------------------------------------------------
# Low-level synthetic generator (NO caching)
# -------------------------------------------------------------------------


def _generate_matching_numpy(
    num_sources: int,
    num_destinations: int,
    target_sparsity: float,
    rng: np.random.Generator | None = None,
):
    """
    Pure generator: returns raw NumPy arrays for a matching LP.

    Returns
    -------
    ccol_indices : np.ndarray[int64], shape (num_sources + 1,)
        CSC column pointer array for sources (columns).
    row_indices  : np.ndarray[int64], shape (E,)
        Row indices (destinations) for each nonzero.
    a_values     : np.ndarray[float64], shape (E,)
        Constraint coefficients a_{ij}.
    c_values     : np.ndarray[float64], shape (E,)
        Value coefficients c_{ij} (>=0).
    b_vec        : np.ndarray[float64], shape (num_destinations,)
        Destination capacity constraints b_j (constructed to be nontrivial).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_sources = num_sources
    n_destinations = num_destinations

    # --- hyperparameters for distributions (tweak as you like) -------------
    mu_p, sigma_p = 0.0, 1.0  # destination breadth
    mu_s, sigma_s = 0.0, 1.0  # destination scale for a_{ij}
    mu_v, sigma_v = -4.0, 0.75  # destination base value
    mu_u, sigma_u = 0.0, 0.5  # source affinity
    sigma_eps = 0.5  # per-edge multiplicative noise in c_{ij}
    c_max = 0.5  # cap values
    # -----------------------------------------------------------------------

    # derive avg_degree_per_source from target sparsity (fraction nnz)
    avg_degree_per_source = target_sparsity * n_destinations  # E ≈ n_sources * avg_degree_per_source

    # destination breadth weights Z_j -> probabilities p_j, sum_j p_j = avg_degree_per_source
    Z = rng.lognormal(mean=mu_p, sigma=sigma_p, size=n_destinations)
    Z_sum = Z.sum()
    if Z_sum == 0:
        Z[:] = 1.0
        Z_sum = float(n_destinations)
    p = Z / Z_sum * avg_degree_per_source

    # destination scale s_j for a_{ij}
    s = rng.lognormal(mean=mu_s, sigma=sigma_s, size=n_destinations)

    # destination base value v_j
    v = rng.lognormal(mean=mu_v, sigma=sigma_v, size=n_destinations)

    # source affinity u_i
    u = rng.lognormal(mean=mu_u, sigma=sigma_u, size=n_sources)

    # --- edge generation: which sources each destination can connect to -----
    expected_edges_per_dest = p * n_sources
    K = rng.poisson(expected_edges_per_dest)  # K_j sources per destination j
    K = np.minimum(K, n_sources)

    total_edges = int(K.sum())
    if total_edges == 0:
        raise ValueError("No edges generated; increase target_sparsity.")

    source_ids = np.empty(total_edges, dtype=np.int64)  # column indices (sources)
    dest_ids = np.empty(total_edges, dtype=np.int64)  # row indices (destinations)
    c_values = np.empty(total_edges, dtype=np.float64)
    a_values = np.empty(total_edges, dtype=np.float64)

    offset = 0
    for j in range(n_destinations):
        k = int(K[j])
        if k == 0:
            continue

        # choose K_j distinct sources for destination j
        sources_j = rng.choice(n_sources, size=k, replace=False)

        # c_{ij} = v_j * u_i * eps_ij, clipped
        u_vals = u[sources_j]
        eps_ij = rng.lognormal(mean=0.0, sigma=sigma_eps, size=k)
        c_ij = v[j] * u_vals * eps_ij
        c_ij = np.minimum(c_ij, c_max)

        # a_{ij} = s_j * c_{ij}
        a_ij = s[j] * c_ij

        source_ids[offset : offset + k] = sources_j
        dest_ids[offset : offset + k] = j
        c_values[offset : offset + k] = c_ij
        a_values[offset : offset + k] = a_ij

        offset += k

    assert offset == total_edges

    # --- build CSC matrices (rows=destinations, cols=sources) ---------------
    # sort by column (source) for CSC layout
    order = np.argsort(source_ids, kind="stable")
    source_sorted = source_ids[order]
    dest_sorted = dest_ids[order]
    a_sorted = a_values[order]
    c_sorted = c_values[order]

    # column pointer array (length n_sources+1)
    counts = np.bincount(source_sorted, minlength=n_sources)
    ccol_indices = np.empty(n_sources + 1, dtype=np.int64)
    ccol_indices[0] = 0
    np.cumsum(counts, out=ccol_indices[1:])

    row_indices = dest_sorted.astype(np.int64, copy=False)

    # --- capacity constraints: make them non-trivial via greedy assignment --
    #
    # Approximate max feasible load per destination under per-source simplex by
    # assigning each source i to the edge (i, j) with largest a_{ij}.
    greedy_loads = np.zeros(n_destinations, dtype=np.float64)

    for i in range(n_sources):
        start = ccol_indices[i]
        end = ccol_indices[i + 1]
        if start == end:
            continue  # no edges for this source
        col_rows = row_indices[start:end]
        col_vals = a_sorted[start:end]
        kmax = col_vals.argmax()
        j = int(col_rows[kmax])
        greedy_loads[j] += float(col_vals[kmax])

    # Make capacity constraints a random fraction of this greedy load so some
    # are tight, some slack, and almost none are "infinite".
    eps = 1e-8
    rho = rng.uniform(0.5, 1.0, size=n_destinations)  # <= 1 => typically binding
    b_vec = rho * (greedy_loads + eps)

    return ccol_indices, row_indices, a_sorted, c_sorted, b_vec


# -------------------------------------------------------------------------
# Cache helpers (memmap + JSON metadata)
# -------------------------------------------------------------------------


def _get_cache_key(
    num_sources: int,
    num_destinations: int,
    target_sparsity: float,
    dtype: torch.dtype,
    seed: int,
) -> tuple:
    """Generate cache key tuple from parameters."""
    return (int(num_sources), int(num_destinations), float(target_sparsity), str(dtype), int(seed))


def _get_cache_prefix(cache_key: tuple) -> str:
    """Generate filename prefix from cache key."""
    num_sources, num_destinations, target_sparsity, dtype_str, seed = cache_key
    # Create a compact but readable prefix
    return f"s{num_sources}_d{num_destinations}_sp{target_sparsity}_{dtype_str.replace('torch.', '')}_seed{seed}"


def _get_meta_path(cache_key: tuple, cache_dir: str) -> str:
    """Get metadata file path for a cache key."""
    prefix = _get_cache_prefix(cache_key)
    return os.path.join(cache_dir, f"{prefix}_meta.json")


def _array_path(cache_key: tuple, suffix: str, cache_dir: str) -> str:
    """Get array file path for a cache key and suffix (e.g., 'A_ccol')."""
    prefix = _get_cache_prefix(cache_key)
    return os.path.join(cache_dir, f"{prefix}_{suffix}.dat")


def _save_array_to_memmap(path: str, arr: np.ndarray) -> None:
    """Create/overwrite a memmap file and copy arr into it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mm = np.memmap(path, dtype=arr.dtype, mode="w+", shape=arr.shape)
    mm[...] = arr
    mm.flush()
    del mm


def _load_array_from_memmap(path: str, shape, dtype) -> np.memmap:
    """Attach to an existing memmap file as a NumPy array."""
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _load_cached_numpy(cache_key: tuple, cache_dir: str):
    """
    Try to load a cached instance (NumPy arrays) from in-process cache or memmap.
    Returns (ccol_indices, row_indices, a_values, c_values, b_vec_np) or None.
    """
    global _in_process_cache

    # In-process cache first (keyed by both cache_dir and cache_key)
    in_process_key = (cache_dir, cache_key)
    if in_process_key in _in_process_cache:
        return _in_process_cache[in_process_key]

    # Disk cache via metadata + memmaps
    try:
        meta_path = _get_meta_path(cache_key, cache_dir)
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Validate metadata matches cache key
        num_sources, num_destinations, target_sparsity, dtype_str, seed = cache_key
        if (
            int(meta.get("num_sources", -1)) != int(num_sources)
            or int(meta.get("num_destinations", -1)) != int(num_destinations)
            or float(meta.get("target_sparsity", -1.0)) != float(target_sparsity)
            or str(meta.get("dtype", "")) != dtype_str
            or int(meta.get("seed", -1)) != int(seed)
        ):
            return None

        shapes = meta["shapes"]
        dtypes_meta = {k: np.dtype(v) for k, v in meta["array_dtypes"].items()}

        ccol_np = _load_array_from_memmap(
            _array_path(cache_key, "A_ccol", cache_dir),
            tuple(shapes["A_ccol"]),
            dtypes_meta["A_ccol"],
        )
        row_np = _load_array_from_memmap(
            _array_path(cache_key, "A_row", cache_dir),
            tuple(shapes["A_row"]),
            dtypes_meta["A_row"],
        )
        A_vals_np = _load_array_from_memmap(
            _array_path(cache_key, "A_vals", cache_dir),
            tuple(shapes["A_vals"]),
            dtypes_meta["A_vals"],
        )
        c_vals_np = _load_array_from_memmap(
            _array_path(cache_key, "c_vals", cache_dir),
            tuple(shapes["c_vals"]),
            dtypes_meta["c_vals"],
        )
        b_vec_np = _load_array_from_memmap(
            _array_path(cache_key, "b_vec", cache_dir),
            tuple(shapes["b_vec"]),
            dtypes_meta["b_vec"],
        )

        # Convert memmaps to normal ndarrays
        ccol = np.asarray(ccol_np)
        row = np.asarray(row_np)
        A_vals = np.asarray(A_vals_np)
        c_vals = np.asarray(c_vals_np)
        b_vec = np.asarray(b_vec_np)

        result = (ccol, row, A_vals, c_vals, b_vec)
        _in_process_cache[in_process_key] = result
        return result

    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return None


def _save_cached_numpy(
    cache_key: tuple,
    ccol_indices: np.ndarray,
    row_indices: np.ndarray,
    a_values: np.ndarray,
    c_values: np.ndarray,
    b_vec_np: np.ndarray,
    cache_dir: str,
) -> None:
    """Save NumPy arrays + metadata to memmaps and update in-process cache."""
    global _in_process_cache

    num_sources, num_destinations, target_sparsity, dtype_str, seed = cache_key

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Save arrays to memmap files
    _save_array_to_memmap(_array_path(cache_key, "A_ccol", cache_dir), ccol_indices)
    _save_array_to_memmap(_array_path(cache_key, "A_row", cache_dir), row_indices)
    _save_array_to_memmap(_array_path(cache_key, "A_vals", cache_dir), a_values)
    _save_array_to_memmap(_array_path(cache_key, "c_vals", cache_dir), c_values)
    _save_array_to_memmap(_array_path(cache_key, "b_vec", cache_dir), b_vec_np)

    # Save metadata
    meta = {
        "num_sources": int(num_sources),
        "num_destinations": int(num_destinations),
        "target_sparsity": float(target_sparsity),
        "dtype": dtype_str,
        "seed": int(seed),
        "shapes": {
            "A_ccol": list(ccol_indices.shape),
            "A_row": list(row_indices.shape),
            "A_vals": list(a_values.shape),
            "c_vals": list(c_values.shape),
            "b_vec": list(b_vec_np.shape),
        },
        "array_dtypes": {
            "A_ccol": str(ccol_indices.dtype),
            "A_row": str(row_indices.dtype),
            "A_vals": str(a_values.dtype),
            "c_vals": str(c_values.dtype),
            "b_vec": str(b_vec_np.dtype),
        },
    }
    meta_path = _get_meta_path(cache_key, cache_dir)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Update in-process cache
    result = (ccol_indices, row_indices, a_values, c_values, b_vec_np)
    in_process_key = (cache_dir, cache_key)
    _in_process_cache[in_process_key] = result


# -------------------------------------------------------------------------
# Main API: generator + caching wrapper
# -------------------------------------------------------------------------


def generate_synthetic_matching_input_args(
    num_sources: int,
    num_destinations: int,
    target_sparsity: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    cache_dir: str | None = None,
) -> MatchingInputArgs:
    """
    Public API: generate a synthetic matching LP instance as MatchingInputArgs.

    A: (num_destinations, num_sources) sparse CSC
    c: (num_destinations, num_sources) sparse CSC  (values negated for minimization)
    b_vec: (num_destinations,) dense

    Columns correspond to sources (i), rows correspond to destinations (j),
    so A[j, i] and c[j, i] are the coefficients for x_{ij}.

    target_sparsity ~ fraction of nonzero entries out of (num_destinations * num_sources).

    Parameters
    ----------
    num_sources : int
        Number of source nodes
    num_destinations : int
        Number of destination nodes
    target_sparsity : float
        Target fraction of nonzero entries
    device : str, default "cpu"
        Device to place tensors on
    dtype : torch.dtype, default torch.float32
        Dtype for output tensors
    seed : int | None, default None
        Random seed for reproducibility. If provided, enables caching.
    rng : np.random.Generator | None, default None
        Explicit RNG (overrides seed if provided). Bypasses caching.
    cache_dir : str | None, default None
        Directory for cached data. If None, uses default ./benchmark_data.

    Caching behavior
    ----------------
    - If seed is provided and rng is None: uses disk cache in cache_dir
      keyed by (num_sources, num_destinations, target_sparsity, dtype, seed).
      Data is generated with np.random.default_rng(seed) for reproducibility.

    - If rng is provided: bypasses cache entirely and uses the provided RNG.

    - If both seed and rng are None: bypasses cache and generates with unseeded RNG.
    """
    # Determine cache directory
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR

    # Determine whether to use cache
    use_cache = (seed is not None) and (rng is None)

    if use_cache:
        cache_key = _get_cache_key(num_sources, num_destinations, target_sparsity, dtype, seed)
        cached = _load_cached_numpy(cache_key, cache_dir)
    else:
        cached = None

    if cached is None:
        # Cache miss or caching disabled: generate fresh arrays
        if rng is None and seed is not None:
            rng = np.random.default_rng(seed)

        ccol_indices, row_indices, a_values, c_values, b_vec_np = _generate_matching_numpy(
            num_sources=num_sources,
            num_destinations=num_destinations,
            target_sparsity=target_sparsity,
            rng=rng,
        )

        # Save to cache if using cache
        if use_cache:
            _save_cached_numpy(
                cache_key=cache_key,
                ccol_indices=ccol_indices,
                row_indices=row_indices,
                a_values=a_values,
                c_values=c_values,
                b_vec_np=b_vec_np,
                cache_dir=cache_dir,
            )
    else:
        ccol_indices, row_indices, a_values, c_values, b_vec_np = cached

    n_sources = num_sources
    n_destinations = num_destinations

    # Build torch CSC tensors (canonical CPU/float32)
    ccol_t = torch.from_numpy(ccol_indices)
    row_t = torch.from_numpy(row_indices)
    A_vals_t = torch.from_numpy(a_values.astype(np.float32))
    # Negate c for minimization convention (if your solver expects that)
    c_vals_t = -torch.from_numpy(c_values.astype(np.float32))
    b_vec_t = torch.from_numpy(b_vec_np.astype(np.float32))

    A_base = torch.sparse_csc_tensor(ccol_t, row_t, A_vals_t, size=(n_destinations, n_sources))
    c_base = torch.sparse_csc_tensor(ccol_t, row_t, c_vals_t, size=(n_destinations, n_sources))

    # Projection map is device-agnostic
    projection_map = create_projection_map(
        proj_type="simplex",
        proj_params={"z": 1.0},
        num_indices=n_sources,
    )

    # Move to requested device/dtype as the final step
    A_out = A_base.to(device=device, dtype=dtype)
    c_out = c_base.to(device=device, dtype=dtype)
    b_out = b_vec_t.to(device=device, dtype=dtype)

    return MatchingInputArgs(
        A=A_out,
        c=c_out,
        projection_map=projection_map,
        b_vec=b_out,
    )


def load_local_partition_from_cache(
    num_sources: int,
    num_destinations: int,
    target_sparsity: float,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
    cache_dir: str | None = None,
) -> MatchingInputArgs:
    """
    Load only this rank's column partition directly from the memmap cache.

    Each rank independently opens the shared memmap files read-only and slices
    out its contiguous block of columns, avoiding the need to load the full
    dataset into memory and scatter via NCCL.

    The column-split formula matches ``split_tensors_to_devices`` in
    ``dualip.utils.dist_utils`` exactly:

        base = num_cols // world_size
        remainder = num_cols % world_size
        start_col = rank * base + min(rank, remainder)
        width = base + (1 if rank < remainder else 0)

    Parameters
    ----------
    num_sources, num_destinations, target_sparsity, dtype, seed :
        Must match the parameters used when the cache was created.
    rank : int
        This process's rank in the distributed group.
    world_size : int
        Total number of ranks.
    cache_dir : str | None
        Directory containing the memmap cache files.

    Returns
    -------
    MatchingInputArgs
        Local partition with A, c, projection_map, and b_vec on CPU.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR

    cache_key = _get_cache_key(num_sources, num_destinations, target_sparsity, dtype, seed)

    # --- Load metadata to get array shapes and dtypes -----------------------
    meta_path = _get_meta_path(cache_key, cache_dir)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    shapes = meta["shapes"]
    dtypes_meta = {k: np.dtype(v) for k, v in meta["array_dtypes"].items()}

    # --- Open memmap files read-only ----------------------------------------
    ccol_mm = _load_array_from_memmap(
        _array_path(cache_key, "A_ccol", cache_dir),
        tuple(shapes["A_ccol"]),
        dtypes_meta["A_ccol"],
    )
    row_mm = _load_array_from_memmap(
        _array_path(cache_key, "A_row", cache_dir),
        tuple(shapes["A_row"]),
        dtypes_meta["A_row"],
    )
    a_vals_mm = _load_array_from_memmap(
        _array_path(cache_key, "A_vals", cache_dir),
        tuple(shapes["A_vals"]),
        dtypes_meta["A_vals"],
    )
    c_vals_mm = _load_array_from_memmap(
        _array_path(cache_key, "c_vals", cache_dir),
        tuple(shapes["c_vals"]),
        dtypes_meta["c_vals"],
    )
    b_vec_mm = _load_array_from_memmap(
        _array_path(cache_key, "b_vec", cache_dir),
        tuple(shapes["b_vec"]),
        dtypes_meta["b_vec"],
    )

    # --- Compute this rank's column range (matches split_tensors_to_devices) -
    num_cols = num_sources
    base = num_cols // world_size
    remainder = num_cols % world_size
    start_col = rank * base + min(rank, remainder)
    width = base + (1 if rank < remainder else 0)
    end_col = start_col + width  # exclusive

    # --- Slice CSC arrays for this rank's columns (matches split_csc_by_cols) -
    nnz_start = int(ccol_mm[start_col])
    nnz_end = int(ccol_mm[end_col])

    sub_ccol = np.array(ccol_mm[start_col : end_col + 1]) - ccol_mm[start_col]
    sub_row = np.array(row_mm[nnz_start:nnz_end])
    sub_a_vals = np.array(a_vals_mm[nnz_start:nnz_end])
    sub_c_vals = np.array(c_vals_mm[nnz_start:nnz_end])
    sub_b_vec = np.array(b_vec_mm)

    # --- Build torch CSC tensors --------------------------------------------
    ccol_t = torch.from_numpy(sub_ccol.astype(np.int64))
    row_t = torch.from_numpy(sub_row.astype(np.int64))
    A_vals_t = torch.from_numpy(sub_a_vals.astype(np.float32))
    c_vals_t = -torch.from_numpy(sub_c_vals.astype(np.float32))  # negate for minimization
    b_vec_t = torch.from_numpy(sub_b_vec.astype(np.float32))

    A_local = torch.sparse_csc_tensor(ccol_t, row_t, A_vals_t, size=(num_destinations, width))
    c_local = torch.sparse_csc_tensor(ccol_t, row_t, c_vals_t, size=(num_destinations, width))

    # --- Projection map for this contiguous block of columns ----------------
    projection_map = create_projection_map(
        proj_type="simplex",
        proj_params={"z": 1.0},
        num_indices=width,
    )

    return MatchingInputArgs(
        A=A_local,
        c=c_local,
        projection_map=projection_map,
        b_vec=b_vec_t,
        equality_mask=None,
    )


if __name__ == "__main__":
    args = generate_synthetic_matching_input_args(
        num_sources=10,
        num_destinations=5,
        target_sparsity=0.9,
        device="cpu",
    )
    print(args)

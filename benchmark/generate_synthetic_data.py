import json
import os

import numpy as np
import torch

from dualip.objectives.matching import MatchingInputArgs
from dualip.projections import create_projection_map

# -------------------------------------------------------------------------
# RAM-backed cache config (via /dev/shm + numpy.memmap)
# -------------------------------------------------------------------------

# Base directory for memmap files: /dev/shm if available (tmpfs = RAM),
# otherwise current directory.
_default_cache_dir = "/dev/shm"
if not os.path.isdir(_default_cache_dir):
    _default_cache_dir = "."

# Default cache directory (can be overridden via environment variable or parameter)
_DEFAULT_CACHE_DIR = os.environ.get("MATCHING_SYNTH_CACHE_DIR", _default_cache_dir)

# Optional in-process cache to avoid re-mmapping within one process.
_cached_key = None
_cached_base_numpy = None  # (ccol_indices, row_indices, a_values, c_values, b_vec_np)


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

    print(f"Generating data: sources: {num_sources} destinations: {num_destinations}  sparsity:{target_sparsity:.2%}")

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

    print("Step 1/5: Generating destination breadth weights...")
    # destination breadth weights Z_j -> probabilities p_j, sum_j p_j = avg_degree_per_source
    Z = rng.lognormal(mean=mu_p, sigma=sigma_p, size=n_destinations)
    Z_sum = Z.sum()
    if Z_sum == 0:
        Z[:] = 1.0
        Z_sum = float(n_destinations)
    p = Z / Z_sum * avg_degree_per_source

    print("Step 2/5: Generating destination scales and base values...")
    # destination scale s_j for a_{ij}
    s = rng.lognormal(mean=mu_s, sigma=sigma_s, size=n_destinations)

    # destination base value v_j
    v = rng.lognormal(mean=mu_v, sigma=sigma_v, size=n_destinations)

    # source affinity u_i
    u = rng.lognormal(mean=mu_u, sigma=sigma_u, size=n_sources)

    print("Step 3/5: Generating edges and coefficients...")
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

    print(f"Step 4/5: Building CSC matrices ({total_edges} edges)...")
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

    print("Step 5/5: Computing capacity constraints via greedy assignment...")
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

    print("Data generation complete!\n")
    return ccol_indices, row_indices, a_sorted, c_sorted, b_vec


# -------------------------------------------------------------------------
# Cache helpers (memmap + JSON metadata)
# -------------------------------------------------------------------------


def _get_cache_paths(cache_dir: str):
    """Get all cache file paths for a given cache directory."""
    return {
        "meta": os.path.join(cache_dir, "matching_synth_meta.json"),
        "A_ccol": os.path.join(cache_dir, "matching_synth_A_ccol.dat"),
        "A_row": os.path.join(cache_dir, "matching_synth_A_row.dat"),
        "A_vals": os.path.join(cache_dir, "matching_synth_A_vals.dat"),
        "c_vals": os.path.join(cache_dir, "matching_synth_c_vals.dat"),
        "b_vec": os.path.join(cache_dir, "matching_synth_b_vec.dat"),
    }


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


def _load_cached_numpy(
    num_sources: int,
    num_destinations: int,
    target_sparsity: float,
    dtype: torch.dtype,
    cache_dir: str,
):
    """
    Try to load a cached instance (NumPy arrays) from in-process cache or memmap.
    Returns (ccol_indices, row_indices, a_values, c_values, b_vec_np) or None.
    """
    global _cached_key, _cached_base_numpy

    dtype_str = str(dtype)
    key = (int(num_sources), int(num_destinations), float(target_sparsity), dtype_str, cache_dir)

    # In-process cache first
    if _cached_key == key and _cached_base_numpy is not None:
        print("Loaded cached data from in-process cache")
        return _cached_base_numpy

    # Cross-process cache via metadata + memmaps
    try:
        print(f"Loading cached data from {cache_dir}")
        cache_paths = _get_cache_paths(cache_dir)
        with open(cache_paths["meta"], "r") as f:
            meta = json.load(f)

        if (
            int(meta.get("num_sources", -1)) != int(num_sources)
            or int(meta.get("num_destinations", -1)) != int(num_destinations)
            or float(meta.get("target_sparsity", -1.0)) != float(target_sparsity)
            or str(meta.get("dtype", "")) != dtype_str
        ):
            return None

        shapes = meta["shapes"]
        dtypes_meta = {k: np.dtype(v) for k, v in meta["dtypes"].items()}

        ccol_np = _load_array_from_memmap(
            cache_paths["A_ccol"],
            tuple(shapes["A_ccol"]),
            dtypes_meta["A_ccol"],
        )
        row_np = _load_array_from_memmap(
            cache_paths["A_row"],
            tuple(shapes["A_row"]),
            dtypes_meta["A_row"],
        )
        A_vals_np = _load_array_from_memmap(
            cache_paths["A_vals"],
            tuple(shapes["A_vals"]),
            dtypes_meta["A_vals"],
        )
        c_vals_np = _load_array_from_memmap(
            cache_paths["c_vals"],
            tuple(shapes["c_vals"]),
            dtypes_meta["c_vals"],
        )
        b_vec_np = _load_array_from_memmap(
            cache_paths["b_vec"],
            tuple(shapes["b_vec"]),
            dtypes_meta["b_vec"],
        )

        # Convert memmaps to normal ndarrays (or leave as memmaps if you like)
        ccol = np.asarray(ccol_np)
        row = np.asarray(row_np)
        A_vals = np.asarray(A_vals_np)
        c_vals = np.asarray(c_vals_np)
        b_vec = np.asarray(b_vec_np)

        _cached_key = key
        _cached_base_numpy = (ccol, row, A_vals, c_vals, b_vec)
        return _cached_base_numpy

    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return None


def _save_cached_numpy(
    num_sources: int,
    num_destinations: int,
    target_sparsity: float,
    dtype: torch.dtype,
    cache_dir: str,
    ccol_indices: np.ndarray,
    row_indices: np.ndarray,
    a_values: np.ndarray,
    c_values: np.ndarray,
    b_vec_np: np.ndarray,
) -> None:
    """Save NumPy arrays + metadata to memmaps and update in-process cache."""
    global _cached_key, _cached_base_numpy

    dtype_str = str(dtype)
    key = (int(num_sources), int(num_destinations), float(target_sparsity), dtype_str, cache_dir)

    cache_paths = _get_cache_paths(cache_dir)
    _save_array_to_memmap(cache_paths["A_ccol"], ccol_indices)
    _save_array_to_memmap(cache_paths["A_row"], row_indices)
    _save_array_to_memmap(cache_paths["A_vals"], a_values)
    _save_array_to_memmap(cache_paths["c_vals"], c_values)
    _save_array_to_memmap(cache_paths["b_vec"], b_vec_np)

    meta = {
        "num_sources": int(num_sources),
        "num_destinations": int(num_destinations),
        "target_sparsity": float(target_sparsity),
        "dtype": dtype_str,
        "shapes": {
            "A_ccol": list(ccol_indices.shape),
            "A_row": list(row_indices.shape),
            "A_vals": list(a_values.shape),
            "c_vals": list(c_values.shape),
            "b_vec": list(b_vec_np.shape),
        },
        "dtypes": {
            "A_ccol": str(ccol_indices.dtype),
            "A_row": str(row_indices.dtype),
            "A_vals": str(a_values.dtype),
            "c_vals": str(c_values.dtype),
            "b_vec": str(b_vec_np.dtype),
        },
    }
    os.makedirs(os.path.dirname(cache_paths["meta"]), exist_ok=True)
    with open(cache_paths["meta"], "w") as f:
        json.dump(meta, f)

    _cached_key = key
    _cached_base_numpy = (ccol_indices, row_indices, a_values, c_values, b_vec_np)


# -------------------------------------------------------------------------
# Main API: generator + caching wrapper
# -------------------------------------------------------------------------


def generate_synthetic_matching_input_args(
    num_sources: int,
    num_destinations: int,
    target_sparsity: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
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

    Caching behavior
    ----------------
    - If rng is None: use a cache keyed by
        (num_sources, num_destinations, target_sparsity, dtype, cache_dir).
      The cache is backed by memmap files in cache_dir,
      plus a tiny JSON metadata file. This works across Python processes.

    - If rng is not None: bypass cache entirely and generate fresh data.

    - cache_dir: Directory where cached data should be saved. If None, uses _DEFAULT_CACHE_DIR.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR

    # Try cache first if rng is None
    if rng is None:
        cached = _load_cached_numpy(num_sources, num_destinations, target_sparsity, dtype, cache_dir)
    else:
        cached = None

    if cached is None:
        # Cache miss or rng explicitly provided: generate fresh arrays
        ccol_indices, row_indices, a_values, c_values, b_vec_np = _generate_matching_numpy(
            num_sources=num_sources,
            num_destinations=num_destinations,
            target_sparsity=target_sparsity,
            rng=rng,
        )

        # Save to cache if rng is None
        if rng is None:
            _save_cached_numpy(
                num_sources=num_sources,
                num_destinations=num_destinations,
                target_sparsity=target_sparsity,
                dtype=dtype,
                cache_dir=cache_dir,
                ccol_indices=ccol_indices,
                row_indices=row_indices,
                a_values=a_values,
                c_values=c_values,
                b_vec_np=b_vec_np,
            )
    else:
        ccol_indices, row_indices, a_values, c_values, b_vec_np = cached

    n_sources = num_sources
    n_destinations = num_destinations

    # Convert dtype to numpy dtype
    if dtype == torch.float32:
        np_dtype = np.float32
    elif dtype == torch.float64:
        np_dtype = np.float64
    else:
        np_dtype = np.float32  # fallback

    # Build torch CSC tensors with the requested dtype
    ccol_t = torch.from_numpy(ccol_indices)
    row_t = torch.from_numpy(row_indices)
    A_vals_t = torch.from_numpy(a_values.astype(np_dtype))
    # Negate c for minimization convention (if your solver expects that)
    c_vals_t = -torch.from_numpy(c_values.astype(np_dtype))
    b_vec_t = torch.from_numpy(b_vec_np.astype(np_dtype))

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


if __name__ == "__main__":
    args = generate_synthetic_matching_input_args(
        num_sources=10,
        num_destinations=5,
        target_sparsity=0.9,
        device="cpu",
    )
    print(args)

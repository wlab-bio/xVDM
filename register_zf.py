#!/usr/bin/env python3
"""
This module computes slice-side gene-pole fields, aggregated-node fields, and
an exact sparse capacitated transport from aggregate GSE nodes onto raw slice
coordinates, optionally followed by graph-regularized refinement. In the GSE
production route it is called after final Infomap coarsening via
``get_aligned_coords_ensemble()``; legacy direct consumers such as
``get_aligned_coords()`` and visualization scripts remain supported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit, prange
from scipy import stats
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

try:
    import scanpy as sc
except (ModuleNotFoundError, RuntimeError):
    import anndata as ad

    class _ScanpyCompat:
        AnnData = ad.AnnData
        read_h5ad = staticmethod(ad.read_h5ad)

    sc = _ScanpyCompat()


EPS = 1e-10
DEFAULT_TIME_VALUE = "12hpf"
GENE_POLE_SMOOTH_K = 30
SOURCE_RATIO_SMOOTH_K = 30
AGG_RATIO_SMOOTH_K = 10
NUM_POLE_PAIRS = 3
GENES_PER_POLE = 3
ABUNDANCE_THRESHOLD = 10
MIN_FEATURE_COUNT = 5
LOCAL_RADIUS_FRACTION = 0.15
OUTPUT_SCHEMA_VERSION = 6
PIPELINE_MODE = "ortools_capacitated_sparse_transport_slice_to_aggregated_ratio_vectors_with_graph_refinement_and_sparse_ensemble_context"
MATCHING_CONTEXT_FILENAME = "matching_context_base.npz"
REFINEMENT_CONTEXT_FILENAME = "matching_refinement_context.npz"
ENSEMBLE_COORDS_FILENAME = "aggregated_nodes_slice_mapped_coords_ensemble.npz"
ENSEMBLE_METADATA_FILENAME = "ensemble_metadata.json"

# Threading policy:
#   * The parent process keeps the Slurm-exported thread budget used by optimOps.
#   * register_zf ensemble replay uses process-level parallelism; ensemble worker
#     processes are locally capped to a small native-thread budget, default 1.
_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "TBB_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
_THREADPOOL_LIMITER = None


def _int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        ivalue = int(str(value).strip())
    except Exception:
        return None
    return ivalue if ivalue > 0 else None


def _affinity_cpu_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return None


def _slurm_or_affinity_cpus() -> int:
    return int(
        _int_env("SLURM_CPUS_PER_TASK")
        or _affinity_cpu_count()
        or os.cpu_count()
        or 1
    )


def _slurm_or_cgroup_mem_limit_bytes() -> int | None:
    """Best-effort memory limit for the current job/container in bytes."""
    candidates: list[int] = []

    mem_node_mb = _int_env("SLURM_MEM_PER_NODE")
    if mem_node_mb is not None:
        candidates.append(int(mem_node_mb) * 1024 * 1024)

    mem_cpu_mb = _int_env("SLURM_MEM_PER_CPU")
    if mem_cpu_mb is not None:
        candidates.append(int(mem_cpu_mb) * max(1, _slurm_or_affinity_cpus()) * 1024 * 1024)

    # cgroup v2, then common cgroup v1 location.  Ignore unlimited sentinels.
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path, "r", encoding="utf-8").read().strip()
            if raw and raw != "max":
                value = int(raw)
                if 0 < value < (1 << 60):
                    candidates.append(value)
        except Exception:
            pass

    return min(candidates) if candidates else None


def _matching_context_arc_count(context_path: str) -> int | None:
    """Read only the tiny indptr vector to estimate sparse context size."""
    try:
        with np.load(context_path, allow_pickle=False) as z:
            if "indptr" not in z.files:
                return None
            indptr = np.asarray(z["indptr"], dtype=np.int64)
            if indptr.size == 0:
                return None
            return int(indptr[-1])
    except Exception:
        return None


def _cap_ensemble_workers_for_memory(
    *,
    context_path: str,
    requested_workers: int,
    output_dir: str | None,
) -> tuple[int, dict[str, object]]:
    """Clamp process parallelism for huge OR-Tools replay contexts.

    Each ensemble worker reloads the sparse matching context and builds an
    independent SimpleMinCostFlow graph.  For 100M+ candidate arcs this is many
    GB per worker, so CPU-count based defaults are unsafe.
    """
    requested_workers = int(max(1, requested_workers))
    arc_count = _matching_context_arc_count(context_path)
    mem_limit = _slurm_or_cgroup_mem_limit_bytes()
    guard_disabled = str(os.getenv("REGZF_DISABLE_ENSEMBLE_MEMORY_GUARD", "")).strip().lower() in {"1", "true", "yes", "on"}

    meta: dict[str, object] = {
        "requested_workers_before_memory_guard": int(requested_workers),
        "matching_context_arc_count": None if arc_count is None else int(arc_count),
        "memory_limit_bytes": None if mem_limit is None else int(mem_limit),
        "memory_guard_disabled": bool(guard_disabled),
    }

    if guard_disabled or requested_workers <= 1:
        return requested_workers, meta

    # A conservative working-set estimate.  The context itself stores several
    # int64/float64 arrays over arcs; the per-solve OR-Tools graph and temporary
    # unit-cost arrays add substantially more.  This intentionally prefers a
    # safe serial run over cgroup OOM kills.
    bytes_per_arc_worker = int(_int_env("REGZF_ENSEMBLE_BYTES_PER_ARC") or 256)
    fixed_worker_overhead = int(_int_env("REGZF_ENSEMBLE_FIXED_WORKER_GB") or 6) * (1024 ** 3)
    reserve_fraction = float(os.getenv("REGZF_ENSEMBLE_MEMORY_RESERVE_FRACTION", "0.25"))
    reserve_fraction = min(max(reserve_fraction, 0.0), 0.90)

    if arc_count is not None and mem_limit is not None:
        usable = int(max(1, mem_limit * (1.0 - reserve_fraction)))
        worker_bytes = int(max(1, arc_count) * bytes_per_arc_worker + fixed_worker_overhead)
        by_mem = max(1, usable // max(1, worker_bytes))
        capped = int(max(1, min(requested_workers, by_mem)))
        meta.update({
            "ensemble_worker_bytes_estimate": int(worker_bytes),
            "ensemble_memory_usable_bytes": int(usable),
            "ensemble_memory_reserve_fraction": float(reserve_fraction),
            "ensemble_bytes_per_arc_worker": int(bytes_per_arc_worker),
            "ensemble_fixed_worker_overhead_bytes": int(fixed_worker_overhead),
            "memory_guard_worker_cap": int(by_mem),
        })
        if capped < requested_workers and output_dir is not None:
            _log_progress(
                output_dir,
                "ensemble_memory_guard",
                f"[ensemble] Memory guard reduced parallel sparse transport workers from {requested_workers:,} to {capped:,} for {arc_count:,} candidate arcs.",
                payload=meta,
            )
        return capped, meta

    # If we cannot estimate memory but the context is obviously huge, avoid the
    # previous CPU-count default.  Users can override with
    # REGZF_DISABLE_ENSEMBLE_MEMORY_GUARD=1 after sizing the job.
    large_arc_limit = int(_int_env("REGZF_ENSEMBLE_PARALLEL_ARC_LIMIT") or 20_000_000)
    if arc_count is not None and arc_count >= large_arc_limit:
        meta["large_arc_serial_limit"] = int(large_arc_limit)
        if output_dir is not None:
            _log_progress(
                output_dir,
                "ensemble_memory_guard",
                f"[ensemble] Memory guard using serial sparse transport replay for {arc_count:,} candidate arcs.",
                payload=meta,
            )
        return 1, meta

    return requested_workers, meta


def _current_process_thread_cap() -> int:
    return int(
        _int_env("REGZF_THREADS_PER_PROCESS")
        or _int_env("OMP_NUM_THREADS")
        or _int_env("NUMBA_NUM_THREADS")
        or _slurm_or_affinity_cpus()
    )


def _resolve_tree_workers(tree_workers: int | None = None) -> int:
    """Resolve SciPy cKDTree worker count without using workers=-1/all CPUs."""
    cap = max(1, _current_process_thread_cap())
    if tree_workers is None:
        env_workers = _int_env("REGZF_TREE_WORKERS")
        if env_workers is not None:
            return max(1, min(int(env_workers), cap))
        return cap
    try:
        requested = int(tree_workers)
    except Exception:
        requested = 0
    if requested <= 0:
        requested = cap
    return max(1, min(requested, cap))


def _set_thread_env_for_current_process(n_threads: int) -> None:
    n_threads = int(max(1, n_threads))
    for key in _THREAD_ENV_KEYS:
        os.environ[key] = str(n_threads)
    os.environ["REGZF_THREADS_PER_PROCESS"] = str(n_threads)
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")


def _save_thread_env() -> dict[str, str | None]:
    keys = set(_THREAD_ENV_KEYS) | {"REGZF_THREADS_PER_PROCESS"}
    return {key: os.environ.get(key) for key in keys}


def _restore_thread_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _limit_runtime_threadpools(n_threads: int) -> None:
    """Best-effort persistent native-thread cap for an ensemble worker process."""
    global _THREADPOOL_LIMITER
    n_threads = int(max(1, n_threads))
    _set_thread_env_for_current_process(n_threads)

    try:
        import numba as _numba
        max_numba = int(getattr(_numba.config, "NUMBA_NUM_THREADS", n_threads))
        _numba.set_num_threads(max(1, min(n_threads, max_numba)))
    except Exception:
        pass

    try:
        from threadpoolctl import threadpool_limits
        if _THREADPOOL_LIMITER is not None:
            try:
                _THREADPOOL_LIMITER.__exit__(None, None, None)
            except Exception:
                pass
        _THREADPOOL_LIMITER = threadpool_limits(limits=n_threads)
        _THREADPOOL_LIMITER.__enter__()
    except Exception:
        pass


def _default_ensemble_mp_start_method() -> str:
    requested = os.getenv("REGZF_ENSEMBLE_MP_START_METHOD")
    methods = multiprocessing.get_all_start_methods()
    if requested:
        requested = str(requested).strip().lower()
        if requested in methods:
            return requested
        raise ValueError(
            f"REGZF_ENSEMBLE_MP_START_METHOD={requested!r} is not available; "
            f"available methods are {methods}."
        )
    if "forkserver" in methods:
        return "forkserver"
    if "spawn" in methods:
        return "spawn"
    return methods[0] if methods else "fork"

GENE_FEATURE_TYPES = {
    "protein_coding",
    "rRNA",
    "Mt_rRNA",
    "lncRNA",
    "miRNA",
    "snRNA",
    "snoRNA",
    "misc_RNA",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_json(path: str, payload) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

def to_1d(x):
    if sp.issparse(x):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()

def norm_rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    ranks = stats.rankdata(x)
    max_rank = float(np.max(ranks))
    if max_rank <= 0:
        return np.zeros_like(ranks, dtype=np.float64)
    return ranks / max_rank

def build_knn_smoothing_operator(coords: np.ndarray, k: int, workers: int | None = None):
    coords = np.asarray(coords, dtype=np.float64)
    n_points = int(coords.shape[0])
    if n_points == 0:
        raise ValueError("Cannot build a smoothing operator with zero points.")

    k_eff = int(max(1, min(int(k), n_points)))
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k_eff, workers=_resolve_tree_workers(workers))
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim == 1:
        indices = indices[:, None]

    rows = np.repeat(np.arange(n_points, dtype=np.int64), indices.shape[1])
    cols = indices.ravel()
    data = np.full(rows.shape[0], 1.0 / indices.shape[1], dtype=np.float64)
    smoother = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    return smoother, indices

def apply_smoother(smoother: csr_matrix, values: np.ndarray) -> np.ndarray:
    return np.asarray(smoother @ np.asarray(values, dtype=np.float64)).ravel()

def compute_rowwise_correlation_matrix(field_matrix: np.ndarray) -> np.ndarray:
    field_matrix = np.asarray(field_matrix, dtype=np.float64)
    if field_matrix.ndim != 2:
        raise ValueError(f"field_matrix must be 2D, got shape {field_matrix.shape}")
    if field_matrix.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)

    centered = field_matrix - field_matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    normalized = np.zeros_like(centered)
    valid = norms > EPS
    if np.any(valid):
        normalized[valid] = centered[valid] / norms[valid, None]

    corr = normalized @ normalized.T
    return np.clip(corr, -1.0, 1.0)

def estimate_gene_pole_block_size(n_points: int) -> int:
    target_bytes = 64 * 1024 * 1024
    max_cols = 64
    n_points = int(max(1, n_points))
    est_cols = int(target_bytes // max(1, n_points * 16))
    return int(max(1, min(max_cols, est_cols)))

def select_case_insensitive_duplicate_genes(adata):
    lower_names = pd.Index([str(v).lower() for v in adata.var_names])
    duplicated_mask = lower_names.duplicated(keep=False)
    if not np.any(duplicated_mask):
        return adata

    if sp.issparse(adata.X):
        column_sums = np.asarray(adata.X.sum(axis=0)).ravel().astype(np.float64)
    else:
        column_sums = np.asarray(adata.X, dtype=np.float64).sum(axis=0)

    selected_indices = []
    for lower_name in pd.unique(lower_names):
        idxs = np.where(lower_names == lower_name)[0]
        if len(idxs) == 1:
            selected_indices.append(int(idxs[0]))
            continue
        best_local = int(np.argmax([float(column_sums[idx]) for idx in idxs]))
        selected_indices.append(int(idxs[best_local]))

    out = adata[:, selected_indices].copy()
    out.var = adata.var.iloc[selected_indices].copy()
    out.var_names = pd.Index([str(lower_names[i]) for i in selected_indices], name=adata.var_names.name)
    return out

def load_source_adata(adata_path: str, time_value=DEFAULT_TIME_VALUE, no_time_filter=False):
    adata = sc.read_h5ad(adata_path)
    if not no_time_filter and "time" in adata.obs.columns:
        time_mask = adata.obs["time"] == time_value
        if np.any(time_mask):
            adata = adata[time_mask].copy()
        else:
            warnings.warn(
                f"Source AnnData has a 'time' column but no '{time_value}' rows. Using all rows."
            )
    elif not no_time_filter and "time" not in adata.obs.columns:
        warnings.warn("Source AnnData has no 'time' column. Using all rows.")

    adata = select_case_insensitive_duplicate_genes(adata)
    adata.var_names = pd.Index([str(v).lower() for v in adata.var_names], name=adata.var_names.name)
    if adata.var_names.duplicated().any():
        adata = adata[:, ~adata.var_names.duplicated()].copy()
        adata.var_names = pd.Index([str(v).lower() for v in adata.var_names], name=adata.var_names.name)

    if "spatial_x" not in adata.obs.columns or "spatial_y" not in adata.obs.columns:
        raise KeyError("Source AnnData must contain obs columns 'spatial_x' and 'spatial_y'.")
    return adata

def get_spatial_coords_from_adata(adata) -> np.ndarray:
    return adata.obs[["spatial_x", "spatial_y"]].to_numpy(dtype=np.float64)

def _gse_sort_key(column_name):
    suffix = str(column_name).split("GSE_", 1)[-1]
    try:
        return (0, int(suffix))
    except ValueError:
        return (1, str(column_name))


def _norm_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _build_request_signature(
    *,
    slice_h5ad_path: str,
    agg_h5ad_path: str,
    time_value: str,
    no_time_filter: bool,
    num_pole_pairs: int,
    genes_per_pole: int,
    abundance_threshold: int,
    min_feature_count: int,
    slice_smooth_k: int,
    agg_smooth_k: int,
    rank_neutral: float,
    match_k0: int,
    match_k_max: int,
    match_lam_dir: float | None,
    match_refine_iter: int,
    pole_pairs_json: str | None = None,
    slice_capacity_mode: str = "mass_exact",
) -> dict[str, object]:
    return {
        "source_adata": _norm_path(slice_h5ad_path),
        "aggregated_h5ad": _norm_path(agg_h5ad_path),
        "time_value": str(time_value).strip().lower(),
        "no_time_filter": bool(no_time_filter),
        "num_pole_pairs": int(num_pole_pairs),
        "genes_per_pole": int(genes_per_pole),
        "abundance_threshold": int(abundance_threshold),
        "min_feature_count": int(min_feature_count),
        "slice_smooth_k": int(slice_smooth_k),
        "agg_smooth_k": int(agg_smooth_k),
        "rank_neutral": float(rank_neutral),
        "match_k0": int(match_k0),
        "match_k_max": int(match_k_max),
        "match_lam_dir": None if match_lam_dir is None else float(match_lam_dir),
        "match_refine_iter": int(match_refine_iter),
        "pole_pairs_json": None if pole_pairs_json is None else _norm_path(pole_pairs_json),
        "slice_capacity_mode": str(slice_capacity_mode or "mass_exact").strip().lower(),
    }



def _build_ensemble_signature(
    *,
    base_request_signature: dict[str, object],
    ensemble_size: int,
    ensemble_seed: int,
    ensemble_mode: str,
    ensemble_tie_max: int,
    ensemble_perturb_units: int,
    ensemble_rel_tol: float,
    ensemble_abs_tol: float,
) -> dict[str, object]:
    """
    Build a deterministic signature for ensemble-only solves.

    This is intentionally separate from _build_request_signature(): changing the
    ensemble size, seed, or perturbation/tie-breaking mode should not invalidate
    the expensive slice/aggregated feature construction or the sparse candidate
    graph cache.
    """
    return {
        "base_request_signature": base_request_signature,
        "ensemble_size": int(ensemble_size),
        "ensemble_seed": int(ensemble_seed),
        "ensemble_mode": str(ensemble_mode),
        "ensemble_tie_max": int(ensemble_tie_max),
        "ensemble_perturb_units": int(ensemble_perturb_units),
        "ensemble_rel_tol": float(ensemble_rel_tol),
        "ensemble_abs_tol": float(ensemble_abs_tol),
    }

def _npz_has_keys(path: str, required: set[str]) -> bool:
    try:
        with np.load(path, allow_pickle=True) as payload:
            return required.issubset(set(payload.files))
    except Exception:
        return False



def write_matching_context_base(
    output_dir: str,
    *,
    rows: np.ndarray,
    cols: np.ndarray,
    indptr: np.ndarray,
    base_costs: np.ndarray,
    base_unit_costs: np.ndarray,
    base_cost_scale: int,
    base_cost_shift: float,
    slice_capacities_active: np.ndarray,
    active_raw_slice_indices: np.ndarray,
    YB_coords_raw: np.ndarray,
    YB_coords_active_raw: np.ndarray,
    row_limits: np.ndarray,
) -> str:
    """
    Persist the sparse transport context needed to replay exact OR-Tools solves.

    The arrays here are the expensive, shared intermediate state for ensemble
    generation: the feasible sparse arc list, per-row CSR offsets, base float
    costs, the exact integer costs used by OR-Tools, active slice capacities,
    and the raw coordinate lookup tables.
    """
    ensure_dir(output_dir)
    path = os.path.join(output_dir, MATCHING_CONTEXT_FILENAME)
    np.savez(
        path,
        rows=np.asarray(rows, dtype=np.int64),
        cols=np.asarray(cols, dtype=np.int64),
        indptr=np.asarray(indptr, dtype=np.int64),
        base_costs=np.asarray(base_costs, dtype=np.float64),
        base_unit_costs=np.asarray(base_unit_costs, dtype=np.int64),
        base_cost_scale=np.asarray([int(base_cost_scale)], dtype=np.int64),
        base_cost_shift=np.asarray([float(base_cost_shift)], dtype=np.float64),
        slice_capacities_active=np.asarray(slice_capacities_active, dtype=np.int64),
        active_raw_slice_indices=np.asarray(active_raw_slice_indices, dtype=np.int64),
        YB_coords_raw=np.asarray(YB_coords_raw, dtype=np.float64),
        YB_coords_active_raw=np.asarray(YB_coords_active_raw, dtype=np.float64),
        row_limits=np.asarray(row_limits, dtype=np.int64),
    )
    return path


def load_matching_context_base(path: str) -> dict[str, object]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sparse matching context not found: {path}")

    with np.load(path, allow_pickle=False) as z:
        ctx = {k: z[k] for k in z.files}

    required = {
        "rows",
        "cols",
        "indptr",
        "base_costs",
        "base_unit_costs",
        "base_cost_scale",
        "base_cost_shift",
        "slice_capacities_active",
        "active_raw_slice_indices",
        "YB_coords_raw",
        "YB_coords_active_raw",
        "row_limits",
    }
    missing = required - set(ctx)
    if missing:
        raise KeyError(f"Sparse matching context {path!r} is missing keys: {sorted(missing)}")

    ctx["base_cost_scale"] = int(np.asarray(ctx["base_cost_scale"]).ravel()[0])
    ctx["base_cost_shift"] = float(np.asarray(ctx["base_cost_shift"]).ravel()[0])
    return ctx


def write_matching_refinement_context(
    output_dir: str,
    *,
    W_graph: csr_matrix,
    XA_features_01: np.ndarray,
    YB_features_active: np.ndarray,
    YB_coords_active_graph: np.ndarray,
    requested_refine_iter: int,
    lam_dir_used: float,
    graph_candidate_k: int,
) -> str:
    """Persist only the extra state needed to replay graph refinement.

    The large sparse feature-transport candidate graph remains in
    ``matching_context_base.npz``.  This sidecar is loaded only when ensemble
    members must undergo the same graph-regularized refinement as the single
    returned alignment.
    """
    ensure_dir(output_dir)
    path = os.path.join(output_dir, REFINEMENT_CONTEXT_FILENAME)
    W_graph = sp.csr_matrix(W_graph, dtype=np.float64)
    W_graph.sort_indices()
    np.savez(
        path,
        W_graph_data=np.asarray(W_graph.data, dtype=np.float64),
        W_graph_indices=np.asarray(W_graph.indices, dtype=np.int64),
        W_graph_indptr=np.asarray(W_graph.indptr, dtype=np.int64),
        W_graph_shape=np.asarray(W_graph.shape, dtype=np.int64),
        XA_features_01=np.asarray(XA_features_01, dtype=np.float64),
        YB_features_active=np.asarray(YB_features_active, dtype=np.float64),
        YB_coords_active_graph=np.asarray(YB_coords_active_graph, dtype=np.float64),
        requested_refine_iter=np.asarray([int(requested_refine_iter)], dtype=np.int64),
        lam_dir_used=np.asarray([float(lam_dir_used)], dtype=np.float64),
        graph_candidate_k=np.asarray([int(graph_candidate_k)], dtype=np.int64),
    )
    return path


def load_matching_refinement_context(path: str) -> dict[str, object]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Sparse matching refinement context not found: {path}")
    with np.load(path, allow_pickle=False) as z:
        required = {
            "W_graph_data",
            "W_graph_indices",
            "W_graph_indptr",
            "W_graph_shape",
            "XA_features_01",
            "YB_features_active",
            "YB_coords_active_graph",
            "requested_refine_iter",
            "lam_dir_used",
            "graph_candidate_k",
        }
        missing = required - set(z.files)
        if missing:
            raise KeyError(f"Sparse refinement context {path!r} is missing keys: {sorted(missing)}")
        shape = tuple(int(v) for v in np.asarray(z["W_graph_shape"]).ravel())
        W_graph = sp.csr_matrix(
            (
                np.asarray(z["W_graph_data"], dtype=np.float64),
                np.asarray(z["W_graph_indices"], dtype=np.int64),
                np.asarray(z["W_graph_indptr"], dtype=np.int64),
            ),
            shape=shape,
        )
        W_graph.sort_indices()
        return {
            "W_graph": W_graph,
            "XA_features_01": np.asarray(z["XA_features_01"], dtype=np.float64),
            "YB_features_active": np.asarray(z["YB_features_active"], dtype=np.float64),
            "YB_coords_active_graph": np.asarray(z["YB_coords_active_graph"], dtype=np.float64),
            "requested_refine_iter": int(np.asarray(z["requested_refine_iter"]).ravel()[0]),
            "lam_dir_used": float(np.asarray(z["lam_dir_used"]).ravel()[0]),
            "graph_candidate_k": int(np.asarray(z["graph_candidate_k"]).ravel()[0]),
        }

def extract_gse_coordinates(adata, min_dims=2, max_dims=3) -> np.ndarray:
    if "X_gse" in adata.obsm:
        coords = np.asarray(adata.obsm["X_gse"], dtype=np.float64)
    else:
        gse_cols = sorted(
            [c for c in adata.obs.columns if str(c).startswith("GSE_")],
            key=_gse_sort_key,
        )
        if not gse_cols:
            raise KeyError("Aggregated h5ad must contain obsm['X_gse'] or obs columns GSE_1, GSE_2, ...")
        coords = adata.obs[gse_cols].to_numpy(dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] < min_dims:
        raise ValueError(f"Need at least {min_dims} GSE dimensions, got shape {coords.shape}.")
    keep_dims = coords.shape[1] if max_dims is None else min(coords.shape[1], int(max_dims))
    return np.asarray(coords[:, :keep_dims], dtype=np.float64)


def load_aggregated_gse_h5ad(adata_path: str):
    adata = sc.read_h5ad(adata_path)
    coords = extract_gse_coordinates(adata, min_dims=2, max_dims=3)

    X = adata.X
    feature_names = pd.Index([str(v) for v in adata.var_names])
    lower_names = feature_names.str.lower()

    keep_mask = ~lower_names.str.startswith("__genome__:") & ~lower_names.str.startswith("seq:")
    if "feature_type" in adata.var.columns:
        keep_mask &= adata.var["feature_type"].astype(str).isin(GENE_FEATURE_TYPES).to_numpy()

    X = X[:, keep_mask]
    feature_names = feature_names[keep_mask]

    # IMPORTANT: counts are already filtered, binarized, and summed upstream.
    # We therefore collapse duplicate genes by summation only and do not re-threshold.
    counts, feature_names = collapse_feature_matrix_case_insensitive(X, feature_names)
    feature_names = pd.Index([str(v).lower() for v in feature_names])

    gene_total_counts = np.asarray(counts.sum(axis=0)).ravel().astype(np.float64)
    gene_totals = {
        str(g): float(c)
        for g, c in zip(feature_names, gene_total_counts)
        if float(c) > 0
    }

    return {
        "path": adata_path,
        "coords": coords,
        "counts": counts.tocsr(),
        "feature_names": np.asarray(feature_names, dtype=object),
        "gene_total_counts": gene_total_counts,
        "gene_totals": gene_totals,
        "n_supernodes": int(adata.n_obs),
    }


def collapse_feature_matrix_case_insensitive(matrix, feature_names):
    feature_names = pd.Index([str(v).lower() for v in feature_names])

    if sp.issparse(matrix):
        matrix = matrix.tocsr()
    else:
        matrix = sp.csr_matrix(np.asarray(matrix))

    codes, uniques = pd.factorize(feature_names, sort=False)
    if len(uniques) == matrix.shape[1]:
        return matrix.tocsr(), pd.Index(uniques)

    collapse = sp.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (np.arange(len(codes)), codes)),
        shape=(len(codes), len(uniques)),
    )
    collapsed = (matrix @ collapse).tocsr()
    return collapsed, pd.Index(uniques)

def collect_shared_genes(agg_dataset, source_adata, min_feature_count=MIN_FEATURE_COUNT):
    source_genes = {str(v).lower() for v in source_adata.var_names}
    shared = [
        gene for gene, total in agg_dataset["gene_totals"].items()
        if gene in source_genes and float(total) >= float(min_feature_count)
    ]
    shared = sorted(shared)
    if not shared:
        raise ValueError("No shared genes remain after source/aggregated intersection and thresholding.")
    return shared

def collect_gene_pole_statistics(
    adata,
    shared_gene_list,
    abundance_threshold=ABUNDANCE_THRESHOLD,
    smooth_k=GENE_POLE_SMOOTH_K,
    local_radius_fraction=LOCAL_RADIUS_FRACTION,
    coords=None,
    smoothing_operator=None,
):
    coords = get_spatial_coords_from_adata(adata) if coords is None else np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError(f"coords must be 2D, got shape {coords.shape}")

    if smoothing_operator is None:
        smoother, _ = build_knn_smoothing_operator(coords, smooth_k)
    else:
        smoother = smoothing_operator
        if smoother.shape != (len(coords), len(coords)):
            raise ValueError(
                "Provided smoothing_operator shape does not match the number of source coordinates."
            )

    spatial_scale = float(np.linalg.norm(np.ptp(coords, axis=0)))
    if not np.isfinite(spatial_scale) or spatial_scale < EPS:
        spatial_scale = 1.0
    local_radius = max(float(local_radius_fraction) * spatial_scale, EPS)
    local_radius2 = local_radius ** 2

    var_names_lower = pd.Index([str(v).lower() for v in adata.var_names])
    gene_to_col = {gene: idx for idx, gene in enumerate(var_names_lower)}

    candidate_genes = []
    candidate_indices = []
    for gene in shared_gene_list:
        gene = str(gene).lower()
        idx = gene_to_col.get(gene)
        if idx is not None:
            candidate_genes.append(gene)
            candidate_indices.append(int(idx))

    if len(candidate_indices) == 0:
        raise ValueError("No shared genes are present in the source AnnData after lowercasing.")

    candidate_matrix = adata.X[:, candidate_indices]
    if sp.issparse(candidate_matrix):
        candidate_matrix = candidate_matrix.tocsc()
        detection_counts = np.asarray((candidate_matrix > 0).sum(axis=0)).ravel().astype(np.int64)
    else:
        candidate_matrix = np.asarray(candidate_matrix)
        detection_counts = np.sum(candidate_matrix > 0, axis=0).astype(np.int64)

    keep = detection_counts >= int(abundance_threshold)
    candidate_genes = [candidate_genes[i] for i in np.where(keep)[0]]
    detection_counts = detection_counts[keep]
    candidate_matrix = candidate_matrix[:, keep]

    if len(candidate_genes) < 2:
        raise ValueError("Fewer than two shared genes passed the abundance/locality prefilter.")

    coord_sq_norms = np.einsum("ij,ij->i", coords, coords)
    block_size = estimate_gene_pole_block_size(len(coords))

    stats_records = []
    field_rows = []
    peak_rows = []

    for block_start in range(0, len(candidate_genes), block_size):
        block_end = min(block_start + block_size, len(candidate_genes))
        block = candidate_matrix[:, block_start:block_end]

        if sp.issparse(block):
            detected_block = (block > 0).astype(np.float64).toarray()
        else:
            detected_block = (np.asarray(block) > 0).astype(np.float64, copy=False)
        if detected_block.ndim == 1:
            detected_block = detected_block[:, None]

        smoothed_block = np.asarray(smoother @ detected_block, dtype=np.float64)
        if smoothed_block.ndim == 1:
            smoothed_block = smoothed_block[:, None]

        for local_idx in range(smoothed_block.shape[1]):
            gene_idx = block_start + local_idx
            gene = candidate_genes[gene_idx]
            abundance = int(detection_counts[gene_idx])
            smoothed_field = smoothed_block[:, local_idx]

            field_mass = float(np.sum(smoothed_field))
            if field_mass <= EPS:
                continue

            peak_idx = int(np.argmax(smoothed_field))
            peak_loc = coords[peak_idx]
            dist2 = np.maximum(coord_sq_norms - 2.0 * (coords @ peak_loc) + coord_sq_norms[peak_idx], 0.0)
            spread = float(np.sqrt(np.dot(smoothed_field, dist2) / (field_mass + EPS)))
            local_mass_fraction = float(np.sum(smoothed_field[dist2 <= local_radius2]) / (field_mass + EPS))
            peak_value = float(smoothed_field[peak_idx])

            spread_norm = spread / (spatial_scale + EPS)
            locality_score = float((peak_value * local_mass_fraction) / (spread_norm + EPS))

            stats_records.append(
                {
                    "gene": gene,
                    "abundance": abundance,
                    "peak_x": float(peak_loc[0]),
                    "peak_y": float(peak_loc[1]),
                    "peak_value": peak_value,
                    "field_mass": field_mass,
                    "spread": spread,
                    "spread_norm": float(spread_norm),
                    "local_mass_fraction": local_mass_fraction,
                    "locality_score": locality_score,
                }
            )
            field_rows.append(smoothed_field.copy())
            peak_rows.append(peak_loc.copy())

    if len(stats_records) < 2:
        raise ValueError("Fewer than two shared genes passed the abundance/locality prefilter.")

    sort_order = np.argsort(np.asarray([record["gene"] for record in stats_records], dtype=object))
    stats_records = [stats_records[idx] for idx in sort_order]
    field_matrix = np.vstack([field_rows[idx] for idx in sort_order]).astype(np.float64)
    peak_locations_matrix = np.vstack([peak_rows[idx] for idx in sort_order]).astype(np.float64)

    stats_df = pd.DataFrame(stats_records).reset_index(drop=True)
    stats_df["locality_rank"] = norm_rankdata(np.log1p(stats_df["locality_score"].to_numpy(dtype=np.float64)))

    genes = stats_df["gene"].tolist()
    gene_to_index = {gene: idx for idx, gene in enumerate(genes)}

    return {
        "stats_df": stats_df,
        "genes": genes,
        "gene_to_index": gene_to_index,
        "field_matrix": field_matrix,
        "peak_locations_matrix": peak_locations_matrix,
        "spatial_scale": spatial_scale,
        "local_radius": local_radius,
        "smooth_k": int(max(1, min(int(smooth_k), len(coords)))),
        "block_size": int(block_size),
        "smoothing_operator_reused": bool(smoothing_operator is not None),
    }

def subset_pole_stats(pole_stats, candidate_genes):
    candidate_genes = [str(g).lower() for g in candidate_genes]
    idx = [
        int(pole_stats["gene_to_index"][gene])
        for gene in candidate_genes
        if gene in pole_stats["gene_to_index"]
    ]
    if len(idx) < 2:
        raise ValueError("Need at least two candidate genes to form a pole pair.")

    stats_df = pole_stats["stats_df"].iloc[idx].reset_index(drop=True)
    field_matrix = np.asarray(pole_stats["field_matrix"], dtype=np.float64)[idx]
    peak_locations_matrix = np.asarray(pole_stats["peak_locations_matrix"], dtype=np.float64)[idx]

    subset = {
        "stats_df": stats_df,
        "genes": stats_df["gene"].tolist(),
        "gene_to_index": {gene: i for i, gene in enumerate(stats_df["gene"].tolist())},
        "field_matrix": field_matrix,
        "peak_locations_matrix": peak_locations_matrix,
        "spatial_scale": float(pole_stats["spatial_scale"]),
        "local_radius": float(pole_stats["local_radius"]),
        "smooth_k": int(pole_stats["smooth_k"]),
        "block_size": int(pole_stats["block_size"]),
        "smoothing_operator_reused": bool(pole_stats.get("smoothing_operator_reused", False)),
        "corr_matrix": compute_rowwise_correlation_matrix(field_matrix),
    }
    return subset

def choose_anchor_pole_pair(pole_stats, random_seed=None):
    stats_df = pole_stats["stats_df"].copy()
    field_matrix = np.asarray(pole_stats["field_matrix"], dtype=np.float64)
    peak_locations = np.asarray(pole_stats["peak_locations_matrix"], dtype=np.float64)
    spatial_scale = float(pole_stats["spatial_scale"])

    genes = stats_df["gene"].tolist()
    locality_rank = stats_df["locality_rank"].to_numpy(dtype=np.float64)

    corr_matrix = pole_stats.get("corr_matrix")
    if corr_matrix is None or corr_matrix.shape != (len(genes), len(genes)):
        corr_matrix = compute_rowwise_correlation_matrix(field_matrix)
        pole_stats["corr_matrix"] = corr_matrix

    positive_overlap = np.maximum(corr_matrix, 0.0)
    separation = np.linalg.norm(peak_locations[:, None, :] - peak_locations[None, :, :], axis=2)
    separation = separation / (spatial_scale + EPS)

    locality_pair_weight = np.sqrt(np.outer(locality_rank, locality_rank))
    pair_score_matrix = separation * locality_pair_weight * (1.0 - positive_overlap)
    np.fill_diagonal(pair_score_matrix, -np.inf)

    upper_i, upper_j = np.triu_indices(len(genes), k=1)
    if upper_i.size == 0:
        raise ValueError("Could not choose a pair of anchor poles.")

    candidate_scores = pair_score_matrix[upper_i, upper_j]
    if not np.any(np.isfinite(candidate_scores)):
        raise ValueError("Could not choose a pair of anchor poles.")

    rng = np.random.default_rng(0 if random_seed is None else random_seed)
    candidate_scores = candidate_scores + rng.uniform(0.0, 1e-12, size=candidate_scores.shape)
    best_flat = int(np.argmax(candidate_scores))
    i = int(upper_i[best_flat])
    j = int(upper_j[best_flat])

    centered_coords = peak_locations - peak_locations.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered_coords, full_matrices=False)
        dominant_axis = vt[0]
        if dominant_axis[0] < 0:
            dominant_axis = -dominant_axis
    except Exception:
        dominant_axis = np.zeros(peak_locations.shape[1], dtype=np.float64)
        dominant_axis[0] = 1.0

    if float(np.dot(peak_locations[i], dominant_axis)) <= float(np.dot(peak_locations[j], dominant_axis)):
        anchorA_idx, anchorB_idx = i, j
    else:
        anchorA_idx, anchorB_idx = j, i

    return genes[anchorA_idx], genes[anchorB_idx], {
        "pair_score": float(pair_score_matrix[i, j]),
        "separation_norm": float(separation[i, j]),
        "positive_overlap": float(positive_overlap[i, j]),
    }

def assign_genes_to_anchor_poles(pole_stats, anchorA_gene, anchorB_gene, num_top_genes=GENES_PER_POLE):
    stats_df = pole_stats["stats_df"].copy()
    genes = stats_df["gene"].tolist()
    gene_to_index = pole_stats.get("gene_to_index") or {gene: idx for idx, gene in enumerate(genes)}

    if anchorA_gene not in gene_to_index or anchorB_gene not in gene_to_index:
        raise KeyError("Anchor genes are missing from the pole statistics table.")

    corr_matrix = pole_stats.get("corr_matrix")
    if corr_matrix is None or corr_matrix.shape != (len(genes), len(genes)):
        corr_matrix = compute_rowwise_correlation_matrix(np.asarray(pole_stats["field_matrix"], dtype=np.float64))
        pole_stats["corr_matrix"] = corr_matrix

    idxA = int(gene_to_index[anchorA_gene])
    idxB = int(gene_to_index[anchorB_gene])
    corrA = np.maximum(corr_matrix[:, idxA], 0.0)
    corrB = np.maximum(corr_matrix[:, idxB], 0.0)

    locality_rank = stats_df["locality_rank"].to_numpy(dtype=np.float64)
    assigned_type = np.where(corrA >= corrB, "A", "B").astype(object)
    assignment_score = locality_rank * np.maximum(corrA, corrB)

    assigned_type[idxA] = "A"
    assigned_type[idxB] = "B"
    assignment_score[idxA] = np.inf
    assignment_score[idxB] = np.inf

    assignment_df = stats_df.copy()
    assignment_df["corr_to_anchorA"] = corrA
    assignment_df["corr_to_anchorB"] = corrB
    assignment_df["assigned_type"] = assigned_type
    assignment_df["assignment_score"] = assignment_score

    typeA_df = (
        assignment_df[assignment_df["assigned_type"] == "A"]
        .sort_values(["assignment_score", "locality_score", "gene"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    typeB_df = (
        assignment_df[assignment_df["assigned_type"] == "B"]
        .sort_values(["assignment_score", "locality_score", "gene"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    nA = min(int(num_top_genes), len(typeA_df))
    nB = min(int(num_top_genes), len(typeB_df))

    typeA_genes = typeA_df["gene"].head(nA).tolist()
    typeB_genes = typeB_df["gene"].head(nB).tolist()

    if anchorA_gene not in typeA_genes:
        typeA_genes = [anchorA_gene] + [g for g in typeA_genes if g != anchorA_gene]
        typeA_genes = typeA_genes[:max(1, int(num_top_genes))]
    if anchorB_gene not in typeB_genes:
        typeB_genes = [anchorB_gene] + [g for g in typeB_genes if g != anchorB_gene]
        typeB_genes = typeB_genes[:max(1, int(num_top_genes))]

    return typeA_genes, typeB_genes, assignment_df

def validate_pole_pairs_disjoint(pole_pairs):
    seen = {}
    for pair in pole_pairs:
        pair_id = str(pair["pair_id"])
        typeA_genes = [str(g).lower() for g in pair["typeA_genes"]]
        typeB_genes = [str(g).lower() for g in pair["typeB_genes"]]

        if set(typeA_genes) & set(typeB_genes):
            raise ValueError(f"Pole pair {pair_id} reuses a gene on both A and B sides.")

        for gene in typeA_genes + typeB_genes:
            if gene in seen:
                raise ValueError(
                    f"Gene '{gene}' is reused across pole pairs ({seen[gene]} and {pair_id}); "
                    "pole gene sets must be globally non-overlapping."
                )
            seen[gene] = pair_id


def _load_json_payload(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_explicit_pole_pairs_json(
    path: str,
    shared_gene_list,
    *,
    num_pairs: int | None = None,
    genes_per_pole: int | None = None,
):
    """Load caller-specified pole-pair definitions without inferring data origin.

    The accepted JSON form is either a list of pair records or an object with a
    ``pole_pairs`` / ``pairs`` list.  Each record must contain ``typeA_genes``
    and ``typeB_genes``.  This is intentionally a generic override: the caller
    may use it for controls, hand-curated biology, or simulator fixtures.
    """
    payload = _load_json_payload(path)
    if isinstance(payload, dict):
        raw_pairs = payload.get("pole_pairs", payload.get("pairs", None))
    else:
        raw_pairs = payload
    if not isinstance(raw_pairs, list) or len(raw_pairs) == 0:
        raise ValueError(f"Explicit pole-pairs JSON {path!r} must contain a nonempty list of pair records.")

    shared = {str(g).lower() for g in shared_gene_list}
    n_limit = None if num_pairs is None else int(max(1, int(num_pairs)))
    g_limit = None if genes_per_pole is None else int(max(1, int(genes_per_pole)))
    out = []
    for i, rec in enumerate(raw_pairs):
        if not isinstance(rec, dict):
            raise ValueError(f"Pole-pair record {i} in {path!r} is not a JSON object.")
        genes_a = [str(g).lower() for g in rec.get("typeA_genes", rec.get("genesA", []))]
        genes_b = [str(g).lower() for g in rec.get("typeB_genes", rec.get("genesB", []))]
        if g_limit is not None:
            genes_a = genes_a[:g_limit]
            genes_b = genes_b[:g_limit]
        if not genes_a or not genes_b:
            raise ValueError(f"Pole-pair record {i} must contain nonempty typeA_genes and typeB_genes.")
        missing = sorted((set(genes_a) | set(genes_b)) - shared)
        if missing:
            raise ValueError(
                f"Explicit pole-pairs JSON {path!r} references genes that are not shared by slice and aggregate: "
                + ", ".join(missing[:20])
            )
        pair_id = str(rec.get("pair_id", f"pair{len(out):02d}"))
        out.append({
            "pair_id": pair_id,
            "anchorA_gene": str(rec.get("anchorA_gene", genes_a[0])).lower(),
            "anchorB_gene": str(rec.get("anchorB_gene", genes_b[0])).lower(),
            "typeA_genes": genes_a,
            "typeB_genes": genes_b,
            "pair_score": float(rec.get("pair_score", 0.0)),
            "separation_norm": float(rec.get("separation_norm", 0.0)),
            "positive_overlap": float(rec.get("positive_overlap", 0.0)),
        })
        if n_limit is not None and len(out) >= n_limit:
            break

    if len(out) == 0:
        raise ValueError(f"Explicit pole-pairs JSON {path!r} yielded no usable pole pairs.")
    validate_pole_pairs_disjoint(out)
    return out


def write_pole_pairs_outputs(output_dir, pole_pairs):
    rows = []
    for pair in pole_pairs:
        for gene in pair["typeA_genes"]:
            rows.append({"pair_id": pair["pair_id"], "side": "A", "gene": gene})
        for gene in pair["typeB_genes"]:
            rows.append({"pair_id": pair["pair_id"], "side": "B", "gene": gene})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "pole_pairs_genes.csv"), index=False)
    with open(os.path.join(output_dir, "pole_pairs.json"), "w") as f:
        json.dump(pole_pairs, f, indent=2)

def identify_typeAB_gene_pairs(
    adata,
    shared_gene_list,
    output_dir,
    num_pairs=NUM_POLE_PAIRS,
    num_top_genes=GENES_PER_POLE,
    abundance_threshold=ABUNDANCE_THRESHOLD,
    coords=None,
    smoothing_operator=None,
):
    pole_stats = collect_gene_pole_statistics(
        adata,
        shared_gene_list,
        abundance_threshold=abundance_threshold,
        smooth_k=GENE_POLE_SMOOTH_K,
        local_radius_fraction=LOCAL_RADIUS_FRACTION,
        coords=coords,
        smoothing_operator=smoothing_operator,
    )

    remaining_genes = pole_stats["genes"].copy()
    used_genes = set()
    pair_specs = []
    diagnostics = []

    for pair_idx in range(int(max(1, num_pairs))):
        candidate_genes = [g for g in remaining_genes if g not in used_genes]
        if len(candidate_genes) < 2:
            break

        sub_stats = subset_pole_stats(pole_stats, candidate_genes)
        anchorA_gene, anchorB_gene, pair_info = choose_anchor_pole_pair(sub_stats)
        typeA_genes, typeB_genes, assignment_df = assign_genes_to_anchor_poles(
            sub_stats,
            anchorA_gene,
            anchorB_gene,
            num_top_genes=num_top_genes,
        )

        pair_id = f"pair{pair_idx:02d}"
        pair_spec = {
            "pair_id": pair_id,
            "anchorA_gene": anchorA_gene,
            "anchorB_gene": anchorB_gene,
            "typeA_genes": [str(g).lower() for g in typeA_genes],
            "typeB_genes": [str(g).lower() for g in typeB_genes],
            "pair_score": float(pair_info["pair_score"]),
            "separation_norm": float(pair_info["separation_norm"]),
            "positive_overlap": float(pair_info["positive_overlap"]),
        }
        pair_specs.append(pair_spec)

        assignment_df = assignment_df.copy()
        assignment_df["pair_id"] = pair_id
        diagnostics.append(assignment_df)

        used_genes.update(pair_spec["typeA_genes"])
        used_genes.update(pair_spec["typeB_genes"])

    if len(pair_specs) == 0:
        raise ValueError("Could not identify any pole pairs.")

    validate_pole_pairs_disjoint(pair_specs)

    write_pole_pairs_outputs(output_dir, pair_specs)

    if diagnostics:
        pd.concat(diagnostics, ignore_index=True).to_csv(
            os.path.join(output_dir, "pole_pair_diagnostics.csv"),
            index=False,
        )

    return pair_specs

def compute_source_ratio_fields_multi(
    adata,
    pole_pairs,
    coords,
    smooth_k=SOURCE_RATIO_SMOOTH_K,
    smoothing_operator=None,
):
    coords = np.asarray(coords, dtype=np.float64)
    n_pairs = int(len(pole_pairs))
    n_obs = int(len(coords))
    if n_pairs <= 0:
        raise ValueError("pole_pairs must contain at least one pair.")

    if smoothing_operator is None:
        smoother, _ = build_knn_smoothing_operator(coords, smooth_k)
    else:
        smoother = smoothing_operator

    typeA_signal = np.zeros((n_obs, n_pairs), dtype=np.float64)
    typeB_signal = np.zeros((n_obs, n_pairs), dtype=np.float64)

    for p, pair in enumerate(pole_pairs):
        maskA = adata.var_names.isin(pair["typeA_genes"])
        maskB = adata.var_names.isin(pair["typeB_genes"])
        typeA_expr = to_1d(adata[:, maskA].X.sum(axis=1))
        typeB_expr = to_1d(adata[:, maskB].X.sum(axis=1))
        typeA_signal[:, p] = apply_smoother(smoother, typeA_expr)
        typeB_signal[:, p] = apply_smoother(smoother, typeB_expr)

    support = typeA_signal + typeB_signal
    ratio = np.full_like(support, 0.5, dtype=np.float64)
    mask = support > EPS
    ratio[mask] = typeA_signal[mask] / (support[mask] + EPS)

    return {
        "coords": coords,
        "ratio": ratio,
        "support": support,
        "typeA_signal": typeA_signal,
        "typeB_signal": typeB_signal,
        "pair_ids": [str(pair.get("pair_id", f"pair{p:02d}")) for p, pair in enumerate(pole_pairs)],
        "pair_metadata": [dict(pair) for pair in pole_pairs],
        "smooth_k": int(max(1, min(int(smooth_k), len(coords)))),
    }

def _dense_pair_count_matrix(x):
    if sp.issparse(x):
        arr = x.toarray()
    else:
        arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[:, None]
    return np.asarray(arr, dtype=np.float64)

def build_pair_count_matrices_from_aggregated(agg_dataset, pole_pairs):
    X = agg_dataset["counts"]
    feature_index = pd.Index(agg_dataset["feature_names"]).str.lower()
    n_pairs = len(pole_pairs)

    typeA_masks = np.zeros((len(feature_index), n_pairs), dtype=np.uint8)
    typeB_masks = np.zeros((len(feature_index), n_pairs), dtype=np.uint8)

    for p, pair in enumerate(pole_pairs):
        genesA = [str(g).lower() for g in pair["typeA_genes"]]
        genesB = [str(g).lower() for g in pair["typeB_genes"]]
        typeA_masks[:, p] = feature_index.isin(genesA).astype(np.uint8)
        typeB_masks[:, p] = feature_index.isin(genesB).astype(np.uint8)

    A = sp.csr_matrix(typeA_masks)
    B = sp.csr_matrix(typeB_masks)

    typeA_counts = _dense_pair_count_matrix(X @ A)
    typeB_counts = _dense_pair_count_matrix(X @ B)
    return np.asarray(agg_dataset["coords"], dtype=np.float64), typeA_counts, typeB_counts

def perform_knn_analysis_with_support_multi(X, typeA_counts, typeB_counts, k=AGG_RATIO_SMOOTH_K, workers: int | None = None):
    X = np.asarray(X, dtype=np.float64)
    typeA_counts = np.asarray(typeA_counts, dtype=np.float64)
    typeB_counts = np.asarray(typeB_counts, dtype=np.float64)
    if typeA_counts.ndim == 1:
        typeA_counts = typeA_counts[:, None]
    if typeB_counts.ndim == 1:
        typeB_counts = typeB_counts[:, None]

    if typeA_counts.shape != typeB_counts.shape:
        raise ValueError(f"typeA_counts shape {typeA_counts.shape} != typeB_counts shape {typeB_counts.shape}")
    if typeA_counts.shape[0] != len(X):
        raise ValueError(f"Count matrices have {typeA_counts.shape[0]} rows for {len(X)} coordinates.")

    n_pairs = int(typeA_counts.shape[1])
    ratio = np.full((len(X), n_pairs), 0.5, dtype=np.float64)
    support = np.zeros((len(X), n_pairs), dtype=np.float64)
    meta = {"per_pair": [], "n_pairs": int(n_pairs), "knn_mode": "weighted-node-multi"}

    for p in range(n_pairs):
        labeled_mask = (typeA_counts[:, p] + typeB_counts[:, p]) > 0
        if not np.any(labeled_mask):
            meta["per_pair"].append({"knn_mode": "none", "n_labeled": 0})
            continue

        X_labeled = X[labeled_mask]
        A = np.asarray(typeA_counts[labeled_mask, p], dtype=np.float64).ravel()
        B = np.asarray(typeB_counts[labeled_mask, p], dtype=np.float64).ravel()
        k_eff = int(min(max(1, int(k)), len(X_labeled)))

        tree = cKDTree(X_labeled)
        _, indices = tree.query(X, k=k_eff, workers=_resolve_tree_workers(workers))
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim == 1:
            indices = indices[:, None]

        neighbor_A = A[indices].sum(axis=1)
        neighbor_B = B[indices].sum(axis=1)
        local_support = neighbor_A + neighbor_B

        support[:, p] = np.asarray(local_support, dtype=np.float64)
        ratio[:, p] = np.where(local_support > 0, neighbor_A / (local_support + EPS), 0.5)
        meta["per_pair"].append({"knn_mode": "weighted-node", "n_labeled": int(len(X_labeled)), "k_used": k_eff})

    return ratio, support, meta

def rank_transform_ratio_fields(ratio: np.ndarray, support: np.ndarray, neutral=0.5) -> np.ndarray:
    ratio = np.asarray(ratio, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    if ratio.ndim == 1:
        ratio = ratio[:, None]
    if support.ndim == 1:
        support = support[:, None]
    if ratio.shape != support.shape:
        raise ValueError(f"ratio shape {ratio.shape} != support shape {support.shape}")

    out = np.full_like(ratio, float(neutral), dtype=np.float64)
    for p in range(ratio.shape[1]):
        mask = support[:, p] > EPS
        if np.any(mask):
            out[mask, p] = norm_rankdata(ratio[mask, p])
    return out

def rescale_rank_features_to_unit_interval(
    rank_values: np.ndarray,
    support: np.ndarray,
    neutral: float = 0.5,
) -> np.ndarray:
    rank_values = np.asarray(rank_values, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    if rank_values.ndim == 1:
        rank_values = rank_values[:, None]
    if support.ndim == 1:
        support = support[:, None]
    if rank_values.shape != support.shape:
        raise ValueError(f"rank_values shape {rank_values.shape} != support shape {support.shape}")

    out = np.full_like(rank_values, float(neutral), dtype=np.float64)
    for p in range(rank_values.shape[1]):
        mask = support[:, p] > EPS
        if not np.any(mask):
            continue
        values = np.asarray(rank_values[mask, p], dtype=np.float64)
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmax > vmin:
            out[mask, p] = (values - vmin) / (vmax - vmin)
        else:
            out[mask, p] = float(neutral)
    return out

def load_aggregated_adjacency_from_transformed_matrix(agg_h5ad_path: str) -> tuple[str, csr_matrix]:
    matrix_path = os.path.join(
        os.path.dirname(os.path.abspath(agg_h5ad_path)),
        "transformed_matrix.npz",
    )
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(
            f"Expected aggregated adjacency at {matrix_path!r} next to the aggregated h5ad."
        )

    try:
        W = sp.load_npz(matrix_path).tocsr()
    except Exception:
        with np.load(matrix_path, allow_pickle=False) as payload:
            required = {"data", "indices", "indptr", "shape"}
            if not required.issubset(payload.files):
                raise
            shape = tuple(int(v) for v in np.asarray(payload["shape"]).ravel())
            W = sp.csr_matrix(
                (payload["data"], payload["indices"], payload["indptr"]),
                shape=shape,
            )

    W = W.tocsr().astype(np.float64)
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"Adjacency loaded from {matrix_path!r} must be square, got {W.shape}.")
    W.sort_indices()
    return matrix_path, W

def _one_hot_csr_from_pairs(
    rows: np.ndarray,
    cols: np.ndarray,
    shape: tuple[int, int],
    dtype=np.uint8,
) -> csr_matrix:
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    if rows.shape != cols.shape:
        raise ValueError(f"rows shape {rows.shape} != cols shape {cols.shape}.")
    if rows.size == 0:
        return csr_matrix(shape, dtype=dtype)
    if np.any(rows < 0) or np.any(rows >= int(shape[0])):
        raise ValueError("Row indices are outside the requested CSR shape.")
    if np.any(cols < 0) or np.any(cols >= int(shape[1])):
        raise ValueError("Column indices are outside the requested CSR shape.")

    data = np.ones(rows.size, dtype=dtype)
    mapping = csr_matrix((data, (rows, cols)), shape=(int(shape[0]), int(shape[1])), dtype=dtype)
    mapping.sort_indices()
    return mapping

def get_node_total_mass(adata) -> np.ndarray:
    """Return generic per-slice capacity weights.

    Prefer explicit observation-level sampling/capacity weights when present;
    otherwise fall back to total expression mass.  This keeps the registration
    code agnostic to the data origin while allowing a caller to provide a
    neutral density model through ordinary obs columns.
    """
    for key in ("slice_capacity_weight", "capacity_weight", "spatial_sampling_weight"):
        if hasattr(adata, "obs") and key in adata.obs.columns:
            # Pandas/AnnData can expose backed obs columns as read-only ndarray
            # views.  Always materialize a writable copy before sanitizing.
            values = adata.obs[key]
            try:
                values = pd.to_numeric(values, errors="coerce")
            except Exception:
                pass
            if hasattr(values, "to_numpy"):
                raw = values.to_numpy(dtype=np.float64, copy=True)
            else:
                raw = values
            mass = np.array(raw, dtype=np.float64, copy=True).ravel()
            mass[~np.isfinite(mass)] = 0.0
            return np.maximum(mass, 0.0)

    # Backed AnnData arrays / sparse reductions may also return read-only views;
    # this vector is sanitized in-place below, so copy unconditionally.
    mass = np.array(to_1d(adata.X.sum(axis=1)), dtype=np.float64, copy=True).ravel()
    mass[~np.isfinite(mass)] = 0.0
    return np.maximum(mass, 0.0)

def compute_mass_proportional_duplication_counts(node_mass: np.ndarray, target_total: int) -> np.ndarray:
    node_mass = np.asarray(node_mass, dtype=np.float64).ravel()
    target_total = int(target_total)
    if node_mass.ndim != 1:
        raise ValueError(f"node_mass must be 1D, got shape {node_mass.shape}.")
    if node_mass.size == 0:
        raise ValueError("node_mass must contain at least one entry.")
    if target_total <= 0:
        raise ValueError("target_total must be positive.")

    mass = node_mass.copy()
    mass[~np.isfinite(mass)] = 0.0
    mass = np.maximum(mass, 0.0)
    total_mass = float(mass.sum())
    if total_mass <= EPS:
        weights = np.full(mass.shape[0], 1.0 / mass.shape[0], dtype=np.float64)
    else:
        weights = mass / total_mass

    expected = weights * float(target_total)
    counts = np.floor(expected).astype(np.int64)
    remainder = int(target_total - counts.sum())

    if remainder > 0:
        fractional = expected - counts
        order = np.lexsort((np.arange(counts.size, dtype=np.int64), -fractional))
        counts[order[:remainder]] += 1
    elif remainder < 0:
        fractional = expected - counts
        reducible = np.where(counts > 0)[0]
        if reducible.size < -remainder:
            raise RuntimeError("Could not reduce duplicate counts to hit the requested target size.")
        order = reducible[np.lexsort((reducible, fractional[reducible]))]
        counts[order[:(-remainder)]] -= 1

    if int(counts.sum()) != target_total:
        raise RuntimeError(
            f"Duplicate-count allocation failed: expected total {target_total}, got {int(counts.sum())}."
        )
    return counts

def _finite_positive_median(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size == 0:
        return 0.0
    return float(np.median(x))


def _log_progress(output_dir: str | None, stage: str, message: str, payload: dict[str, object] | None = None) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    if not output_dir:
        return
    try:
        log_path = os.path.join(output_dir, "progress.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        status = {"timestamp": stamp, "stage": stage, "message": message}
        if payload:
            status["payload"] = payload
        status_path = os.path.join(output_dir, "progress_status.json")
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2, sort_keys=True)
    except Exception:
        pass

def _require_ortools_min_cost_flow():
    try:
        from ortools.graph.python import min_cost_flow as ort_min_cost_flow
    except Exception as exc:
        raise ImportError(
            "This script requires OR-Tools' SimpleMinCostFlow backend. Install OR-Tools, e.g. `pip install ortools`."
        ) from exc
    return ort_min_cost_flow

def _ortools_is_optimal(smcf, status) -> bool:
    optimal = getattr(smcf, "OPTIMAL", None)
    if optimal is not None:
        return status == optimal
    text = str(status).upper()
    return text.endswith("OPTIMAL") or text == "OPTIMAL"

def _ortools_add_arcs_with_capacity_and_unit_cost(smcf, start_nodes, end_nodes, capacities, unit_costs):
    start_nodes = np.asarray(start_nodes, dtype=np.int64).ravel()
    end_nodes = np.asarray(end_nodes, dtype=np.int64).ravel()
    capacities = np.asarray(capacities, dtype=np.int64).ravel()
    unit_costs = np.asarray(unit_costs, dtype=np.int64).ravel()
    if not (start_nodes.shape == end_nodes.shape == capacities.shape == unit_costs.shape):
        raise ValueError("Arc arrays must all have the same shape.")
    if hasattr(smcf, "add_arcs_with_capacity_and_unit_cost"):
        arc_ids = smcf.add_arcs_with_capacity_and_unit_cost(start_nodes, end_nodes, capacities, unit_costs)
        return np.asarray(arc_ids, dtype=np.int64)

    arc_ids = np.empty(start_nodes.size, dtype=np.int64)
    add_one = getattr(smcf, "add_arc_with_capacity_and_unit_cost")
    for idx, (u, v, cap, cost) in enumerate(zip(start_nodes, end_nodes, capacities, unit_costs)):
        arc_ids[idx] = int(add_one(int(u), int(v), int(cap), int(cost)))
    return arc_ids

def _ortools_set_nodes_supplies(smcf, supplies: np.ndarray) -> None:
    supplies = np.asarray(supplies, dtype=np.int64).ravel()
    node_ids = np.arange(supplies.size, dtype=np.int64)
    if hasattr(smcf, "set_nodes_supplies"):
        smcf.set_nodes_supplies(node_ids, supplies)
        return

    set_one = getattr(smcf, "set_node_supply")
    for node, supply in zip(node_ids, supplies):
        set_one(int(node), int(supply))

def _ortools_flows(smcf, arc_ids: np.ndarray) -> np.ndarray:
    arc_ids = np.asarray(arc_ids, dtype=np.int64).ravel()
    if hasattr(smcf, "flows"):
        return np.asarray(smcf.flows(arc_ids), dtype=np.int64)
    get_flow = getattr(smcf, "flow")
    return np.fromiter((int(get_flow(int(arc))) for arc in arc_ids), dtype=np.int64, count=arc_ids.size)

def _ortools_optimal_cost(smcf) -> int:
    if hasattr(smcf, "optimal_cost"):
        return int(smcf.optimal_cost())
    if hasattr(smcf, "OptimalCost"):
        return int(smcf.OptimalCost())
    raise AttributeError("Could not find optimal_cost() on the OR-Tools SimpleMinCostFlow object.")

def _cap_as_csr(W: sp.spmatrix | np.ndarray) -> sp.csr_matrix:
    W = sp.csr_matrix(W, dtype=np.float64)
    if W.shape[0] != W.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    W = W.copy()
    W.setdiag(0.0)
    W.eliminate_zeros()
    W.sort_indices()
    return W

def _cap_symmetrize_csr(W: sp.spmatrix | np.ndarray, average: bool = True) -> sp.csr_matrix:
    W = _cap_as_csr(W)
    W = W + W.T.tocsr()
    if average:
        W.data *= 0.5
    W.sum_duplicates()
    W.setdiag(0.0)
    W.eliminate_zeros()
    W.sort_indices()
    return W.tocsr()

def _huber_edge_costs_a_to_b(
    XA: np.ndarray,
    YB: np.ndarray,
    rows_a: np.ndarray,
    cols_b: np.ndarray,
    weights: np.ndarray,
    deltas: np.ndarray,
) -> np.ndarray:
    m = rows_a.shape[0]
    p = XA.shape[1]
    out = np.empty(m, dtype=np.float64)
    for e in prange(m):
        i = rows_a[e]
        j = cols_b[e]
        c = 0.0
        for d in range(p):
            diff = XA[i, d] - YB[j, d]
            ad = diff if diff >= 0.0 else -diff
            delta = deltas[d]
            if ad <= delta:
                c += weights[d] * 0.5 * diff * diff
            else:
                c += weights[d] * (delta * (ad - 0.5 * delta))
        out[e] = c
    return out

def _cap_weights_array(weights: float | np.ndarray | None, p: int) -> np.ndarray:
    if weights is None:
        return np.ones(p, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 0:
        w = np.full(p, float(w), dtype=np.float64)
    if w.shape != (p,):
        raise ValueError(f"weights must be scalar or have shape ({p},).")
    return np.ascontiguousarray(w)

def _cap_delta_array(delta: float | np.ndarray | None, p: int) -> np.ndarray:
    if delta is None:
        return np.full(p, 0.05, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim == 0:
        d = np.full(p, float(d), dtype=np.float64)
    if d.shape != (p,):
        raise ValueError(f"delta must be scalar or have shape ({p},).")
    return np.ascontiguousarray(d)

def build_variable_knn_candidates_a_to_b(
    XA: np.ndarray,
    YB: np.ndarray,
    row_k: np.ndarray,
    eps: float = 0.0,
    p: float = 2.0,
    workers: int | None = None,
    tree: cKDTree | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    XA = np.asarray(XA, dtype=np.float64)
    YB = np.asarray(YB, dtype=np.float64)
    row_k = np.asarray(row_k, dtype=np.int64).ravel()

    if XA.ndim != 2 or YB.ndim != 2:
        raise ValueError("XA and YB must both be 2D arrays.")
    if XA.shape[1] != YB.shape[1]:
        raise ValueError("XA and YB must have the same number of feature columns.")
    nA = int(XA.shape[0])
    nB = int(YB.shape[0])
    if row_k.shape != (nA,):
        raise ValueError(f"row_k must have shape ({nA},), got {row_k.shape}.")
    if np.any(row_k < 1):
        raise ValueError("Each row_k entry must be at least 1.")
    if np.any(row_k > nB):
        raise ValueError("Each row_k entry must be at most nB.")

    if tree is None:
        tree = cKDTree(YB)
    workers_eff = _resolve_tree_workers(workers)

    indptr = np.empty(nA + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(row_k, out=indptr[1:])
    m = int(indptr[-1])

    rows_a = np.empty(m, dtype=np.int64)
    cols_b = np.empty(m, dtype=np.int64)
    dists = np.empty(m, dtype=np.float64)

    unique_k = np.unique(row_k)
    for k_eff in unique_k:
        row_ids = np.flatnonzero(row_k == int(k_eff))
        if row_ids.size == 0:
            continue
        q_dists, q_idx = tree.query(XA[row_ids], k=int(k_eff), eps=eps, p=p, workers=workers_eff)
        if int(k_eff) == 1:
            q_dists = np.asarray(q_dists, dtype=np.float64)[:, None]
            q_idx = np.asarray(q_idx, dtype=np.int64)[:, None]
        else:
            q_dists = np.asarray(q_dists, dtype=np.float64)
            q_idx = np.asarray(q_idx, dtype=np.int64)

        for local_idx, row in enumerate(row_ids):
            s = int(indptr[row])
            e = int(indptr[row + 1])
            rows_a[s:e] = int(row)
            cols_b[s:e] = np.asarray(q_idx[local_idx], dtype=np.int64).reshape(-1)
            dists[s:e] = np.asarray(q_dists[local_idx], dtype=np.float64).reshape(-1)

    return rows_a, cols_b, dists, indptr

def compute_base_costs_a_to_b(
    XA: np.ndarray,
    YB: np.ndarray,
    rows_a: np.ndarray,
    cols_b: np.ndarray,
    weights: float | np.ndarray | None = None,
    delta: float | np.ndarray | None = 0.05,
) -> np.ndarray:
    XA = np.asarray(XA, dtype=np.float64)
    YB = np.asarray(YB, dtype=np.float64)
    if XA.ndim != 2 or YB.ndim != 2:
        raise ValueError("XA and YB must both be 2D arrays.")
    if XA.shape[1] != YB.shape[1]:
        raise ValueError("XA and YB must have the same number of feature columns.")
    p = XA.shape[1]
    w = _cap_weights_array(weights, p)
    d = _cap_delta_array(delta, p)
    rows_a = np.asarray(rows_a, dtype=np.int64)
    cols_b = np.asarray(cols_b, dtype=np.int64)
    return _huber_edge_costs_a_to_b(XA, YB, rows_a, cols_b, w, d)

def compute_mass_proportional_slice_capacities(node_mass: np.ndarray, target_total: int) -> np.ndarray:
    return compute_mass_proportional_duplication_counts(node_mass=node_mass, target_total=target_total)


def _farthest_point_subset(coords: np.ndarray, k: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    n = int(coords.shape[0])
    k = int(k)
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    if coords.ndim != 2 or n == 0:
        raise ValueError("coords must be a nonempty 2D array for farthest-point capacity selection.")
    center = np.nanmean(coords, axis=0)
    d2 = np.sum((coords - center[None, :]) ** 2, axis=1)
    d2[~np.isfinite(d2)] = np.inf
    first = int(np.nanargmin(d2))
    chosen = np.empty(k, dtype=np.int64)
    chosen[0] = first
    min_d2 = np.sum((coords - coords[first][None, :]) ** 2, axis=1)
    min_d2[~np.isfinite(min_d2)] = -np.inf
    min_d2[first] = -np.inf
    for t in range(1, k):
        j = int(np.nanargmax(min_d2))
        chosen[t] = j
        d2j = np.sum((coords - coords[j][None, :]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2j)
        min_d2[chosen[:t + 1]] = -np.inf
    return chosen


def compute_slice_capacities(
    node_mass: np.ndarray,
    target_total: int,
    *,
    coords: np.ndarray | None = None,
    mode: str = "mass_exact",
) -> dict[str, object]:
    """Compute slice-node capacities from a generic caller-selected policy.

    ``mass_exact`` preserves the historical behavior: capacities sum exactly to
    the number of aggregate nodes.  ``unit_upper`` treats every slice node as an
    available one-use candidate and requires only that total capacity be at least
    the number of aggregate nodes.  ``uniform_exact`` and ``spatial_fps_exact``
    are deterministic exact-size alternatives useful for density diagnostics.
    """
    mode = str(mode or "mass_exact").strip().lower().replace("-", "_")
    node_mass = np.asarray(node_mass, dtype=np.float64).ravel()
    n = int(node_mass.size)
    target_total = int(target_total)
    if n <= 0:
        raise ValueError("node_mass must contain at least one slice node.")
    if target_total <= 0:
        raise ValueError("target_total must be positive.")

    if mode in ("mass", "mass_exact", "expression_mass_exact"):
        cap = compute_mass_proportional_slice_capacities(node_mass, target_total=target_total)
        expected = np.asarray(node_mass, dtype=np.float64).copy()
        expected[~np.isfinite(expected)] = 0.0
        expected = np.maximum(expected, 0.0)
        total_mass = float(expected.sum())
        if total_mass <= EPS:
            expected = np.full(n, float(target_total) / float(n), dtype=np.float64)
        else:
            expected = expected / total_mass * float(target_total)
        return {"capacity": cap, "expected": expected, "mode": "mass_exact", "is_upper_bound": False}

    if mode in ("uniform", "uniform_exact"):
        cap = compute_mass_proportional_duplication_counts(np.ones(n, dtype=np.float64), target_total=target_total)
        expected = np.full(n, float(target_total) / float(n), dtype=np.float64)
        return {"capacity": cap, "expected": expected, "mode": "uniform_exact", "is_upper_bound": False}

    if mode in ("unit_upper", "all_slice_nodes", "all_nodes_upper"):
        if n < target_total:
            raise ValueError(
                "unit_upper slice capacity requires at least as many slice nodes as aggregate nodes; "
                f"got n_slice={n}, n_aggregate={target_total}."
            )
        cap = np.ones(n, dtype=np.int64)
        expected = np.full(n, float(target_total) / float(n), dtype=np.float64)
        return {"capacity": cap, "expected": expected, "mode": "unit_upper", "is_upper_bound": True}

    if mode in ("spatial_fps", "spatial_fps_exact", "farthest_point_exact"):
        if coords is None:
            raise ValueError("spatial_fps_exact capacity mode requires slice coordinates.")
        cap = np.zeros(n, dtype=np.int64)
        chosen = _farthest_point_subset(coords, min(target_total, n))
        cap[chosen] = 1
        if target_total > n:
            # Preserve feasibility if the slice is smaller than the aggregate graph.
            cap += compute_mass_proportional_duplication_counts(np.ones(n, dtype=np.float64), target_total=target_total - n)
        expected = np.asarray(cap, dtype=np.float64)
        return {"capacity": cap, "expected": expected, "mode": "spatial_fps_exact", "is_upper_bound": False}

    raise ValueError(
        "Unknown slice capacity mode " + repr(mode) + "; expected one of "
        "mass_exact, uniform_exact, unit_upper, spatial_fps_exact."
    )

def _active_slice_capacity_view(
    YB_features_01: np.ndarray,
    YB_coords: np.ndarray,
    slice_capacities_raw: np.ndarray,
) -> dict[str, np.ndarray]:
    YB_features_01 = np.asarray(YB_features_01, dtype=np.float64)
    YB_coords = np.asarray(YB_coords, dtype=np.float64)
    slice_capacities_raw = np.asarray(slice_capacities_raw, dtype=np.int64).ravel()

    if slice_capacities_raw.shape[0] != YB_features_01.shape[0] or YB_coords.shape[0] != YB_features_01.shape[0]:
        raise ValueError("slice capacities, coordinates, and features must have the same number of rows.")

    active_raw_slice_indices = np.flatnonzero(slice_capacities_raw > 0).astype(np.int64)
    if active_raw_slice_indices.size == 0:
        raise ValueError("No slice nodes have positive capacity.")

    return {
        "active_raw_slice_indices": active_raw_slice_indices,
        "YB_features_active": np.asarray(YB_features_01[active_raw_slice_indices], dtype=np.float64),
        "YB_coords_active": np.asarray(YB_coords[active_raw_slice_indices], dtype=np.float64),
        "slice_capacities_active": np.asarray(slice_capacities_raw[active_raw_slice_indices], dtype=np.int64),
    }

def capacitated_matching_state(
    rows_a: np.ndarray,
    cols_b: np.ndarray,
    nA: int,
    nB: int,
    slice_capacities: np.ndarray,
) -> dict[str, object]:
    rows_a = np.asarray(rows_a, dtype=np.int64).ravel()
    cols_b = np.asarray(cols_b, dtype=np.int64).ravel()
    slice_capacities = np.asarray(slice_capacities, dtype=np.int64).ravel()
    if slice_capacities.shape != (int(nB),):
        raise ValueError(f"slice_capacities must have shape ({int(nB)},), got {slice_capacities.shape}.")
    if np.any(slice_capacities < 0):
        raise ValueError("slice_capacities must be nonnegative.")
    if int(slice_capacities.sum()) < int(nA):
        raise ValueError("slice_capacities must provide at least nA total capacity for a full capacitated matching.")

    source = 0
    offset_a = 1
    offset_b = 1 + int(nA)
    sink = 1 + int(nA) + int(nB)

    rr = np.concatenate(
        [
            np.full(int(nA), source, dtype=np.int64),
            rows_a + offset_a,
            np.arange(int(nB), dtype=np.int64) + offset_b,
        ]
    )
    cc = np.concatenate(
        [
            np.arange(int(nA), dtype=np.int64) + offset_a,
            cols_b + offset_b,
            np.full(int(nB), sink, dtype=np.int64),
        ]
    )
    data = np.concatenate(
        [
            np.ones(int(nA), dtype=np.int64),
            np.ones(rows_a.size, dtype=np.int64),
            slice_capacities.astype(np.int64, copy=False),
        ]
    )

    graph = sp.csr_matrix((data, (rr, cc)), shape=(sink + 1, sink + 1), dtype=np.int64)
    result = maximum_flow(graph, source, sink, method="dinic")
    flow_value = int(result.flow_value)
    full = flow_value == int(nA)

    flow = sp.csr_matrix(result.flow, dtype=np.int64)
    forward_resid = (graph - flow.maximum(0)).tocsr()
    if forward_resid.nnz:
        forward_resid.data = np.maximum(forward_resid.data, 0)
        forward_resid.eliminate_zeros()
    reverse_resid = (-flow.minimum(0)).tocsr()
    residual = (forward_resid + reverse_resid).tocsr()
    if residual.nnz:
        residual.data[:] = 1

    order = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    reachable = np.zeros(residual.shape[0], dtype=bool)
    reachable[np.asarray(order, dtype=np.int64)] = True

    reachable_a = np.flatnonzero(reachable[offset_a : offset_a + int(nA)]).astype(np.int64)
    reachable_b = np.flatnonzero(reachable[offset_b : offset_b + int(nB)]).astype(np.int64)

    return {
        "full": bool(full),
        "flow_value": int(flow_value),
        "reachable_a": reachable_a,
        "reachable_b": reachable_b,
    }

def _augment_candidate_row_limits(
    row_limits: np.ndarray,
    blocked_rows: np.ndarray,
    nB: int,
    soft_k_max: int,
) -> tuple[np.ndarray, bool]:
    row_limits = np.asarray(row_limits, dtype=np.int64).copy()
    blocked_rows = np.asarray(blocked_rows, dtype=np.int64).ravel()
    if blocked_rows.size == 0:
        return row_limits, False

    old = row_limits[blocked_rows]
    grown = np.maximum(old + 32, np.minimum(int(nB), old * 2))

    if int(soft_k_max) > 0:
        within_soft = old < int(soft_k_max)
        grown[within_soft] = np.minimum(grown[within_soft], int(soft_k_max))

    grown = np.minimum(grown, int(nB))
    changed = grown > old
    if np.any(changed):
        row_limits[blocked_rows[changed]] = grown[changed]
        return row_limits, True

    grown = np.minimum(np.maximum(old + 64, np.minimum(int(nB), old * 2)), int(nB))
    changed = grown > old
    if np.any(changed):
        row_limits[blocked_rows[changed]] = grown[changed]
        return row_limits, True

    return row_limits, False

def _quantize_unit_costs_for_ortools(
    costs: np.ndarray,
    total_flow: int,
    *,
    preferred_scale: int = 1_000_000,
    objective_headroom: int = (1 << 60),
) -> tuple[np.ndarray, int, float]:
    costs = np.asarray(costs, dtype=np.float64).ravel()
    if costs.size == 0:
        return np.zeros(0, dtype=np.int64), 1, 0.0
    if not np.all(np.isfinite(costs)):
        raise ValueError("Encountered non-finite transport arc costs.")

    shift = float(min(0.0, np.min(costs)))
    shifted = costs - shift
    cmax = float(np.max(shifted))
    if cmax <= 0.0:
        return np.zeros_like(costs, dtype=np.int64), 1, shift

    max_scale_float = float(objective_headroom) / (max(float(total_flow), 1.0) * cmax)
    max_scale = int(max(1, np.floor(max_scale_float)))
    scale = int(max(1, min(int(preferred_scale), max_scale)))
    unit_costs = np.rint(shifted * float(scale)).astype(np.int64)
    return unit_costs, scale, shift

def solve_capacitated_transport_ortools_unit_costs(
    unit_costs: np.ndarray,
    rows_a: np.ndarray,
    cols_b: np.ndarray,
    indptr_a: np.ndarray,
    slice_capacities: np.ndarray,
    *,
    original_costs: np.ndarray | None = None,
    output_dir: str | None = None,
    solve_label: str = "unit_cost",
) -> dict[str, object]:
    """
    Solve the sparse capacitated transport problem using pre-quantized integer
    arc costs.

    This preserves the exact sparse integer objective used by OR-Tools and is
    the core primitive used by both the ordinary base solve and the ensemble
    replay solves.
    """
    ort_min_cost_flow = _require_ortools_min_cost_flow()

    unit_costs = np.asarray(unit_costs, dtype=np.int64).ravel()
    rows_a = np.asarray(rows_a, dtype=np.int64).ravel()
    cols_b = np.asarray(cols_b, dtype=np.int64).ravel()
    indptr_a = np.asarray(indptr_a, dtype=np.int64).ravel()
    slice_capacities = np.asarray(slice_capacities, dtype=np.int64).ravel()

    nA = int(indptr_a.size - 1)
    nB = int(slice_capacities.size)
    m = int(rows_a.size)

    if nA <= 0:
        raise ValueError("indptr_a must describe at least one A row.")
    if indptr_a.shape != (nA + 1,) or int(indptr_a[0]) != 0 or int(indptr_a[-1]) != m:
        raise ValueError("indptr_a is inconsistent with the number of sparse arcs.")
    if cols_b.shape != rows_a.shape or unit_costs.shape != rows_a.shape:
        raise ValueError("Arc rows, cols, and unit_costs must all have the same shape.")
    if np.any(rows_a < 0) or np.any(rows_a >= nA):
        raise ValueError("rows_a contains indices outside [0, nA).")
    if np.any(cols_b < 0) or np.any(cols_b >= nB):
        raise ValueError("cols_b contains indices outside [0, nB).")
    if np.any(slice_capacities < 0):
        raise ValueError("slice_capacities must be nonnegative.")
    if int(slice_capacities.sum()) < nA:
        raise ValueError("slice capacities must provide at least nA total capacity for a full capacitated transport solve.")
    if np.any(unit_costs < 0):
        raise ValueError("OR-Tools unit costs must be nonnegative after quantization/tie-breaking.")

    if original_costs is not None:
        original_costs = np.asarray(original_costs, dtype=np.float64).ravel()
        if original_costs.shape != rows_a.shape:
            raise ValueError("original_costs must have the same shape as the sparse arc arrays.")

    source = 0
    offset_a = 1
    offset_b = 1 + nA
    sink = 1 + nA + nB
    num_nodes = sink + 1

    start_source = np.full(nA, source, dtype=np.int64)
    end_source = np.arange(nA, dtype=np.int64) + offset_a
    cap_source = np.ones(nA, dtype=np.int64)
    cost_source = np.zeros(nA, dtype=np.int64)

    start_candidate = rows_a + offset_a
    end_candidate = cols_b + offset_b
    cap_candidate = np.ones(m, dtype=np.int64)

    start_sink = np.arange(nB, dtype=np.int64) + offset_b
    end_sink = np.full(nB, sink, dtype=np.int64)
    cap_sink = slice_capacities.astype(np.int64, copy=False)
    cost_sink = np.zeros(nB, dtype=np.int64)

    supplies = np.zeros(num_nodes, dtype=np.int64)
    supplies[source] = int(nA)
    supplies[sink] = -int(nA)

    if output_dir is not None:
        _log_progress(
            output_dir,
            "matching_transport_start",
            f"[matching] Starting OR-Tools SimpleMinCostFlow solve ({solve_label}) with {nA:,} aggregated nodes, {nB:,} active slice nodes, and {m:,} candidate arcs.",
            payload={
                "solve_label": solve_label,
                "nA": int(nA),
                "nB_active": int(nB),
                "candidate_arc_count": int(m),
                "pre_quantized_unit_costs": True,
            },
        )

    t0 = time.time()
    smcf = ort_min_cost_flow.SimpleMinCostFlow()
    _ortools_add_arcs_with_capacity_and_unit_cost(smcf, start_source, end_source, cap_source, cost_source)
    candidate_arc_ids = _ortools_add_arcs_with_capacity_and_unit_cost(
        smcf,
        start_candidate,
        end_candidate,
        cap_candidate,
        unit_costs,
    )
    _ortools_add_arcs_with_capacity_and_unit_cost(smcf, start_sink, end_sink, cap_sink, cost_sink)
    _ortools_set_nodes_supplies(smcf, supplies)
    status = smcf.solve()
    elapsed = time.time() - t0

    if not _ortools_is_optimal(smcf, status):
        raise RuntimeError(
            f"OR-Tools SimpleMinCostFlow failed for solve {solve_label!r}. Solver status: {status}."
        )

    flows = _ortools_flows(smcf, candidate_arc_ids)
    if flows.shape != (m,):
        flows = np.asarray(flows, dtype=np.int64).ravel()
        if flows.shape != (m,):
            raise RuntimeError("Unexpected OR-Tools flow vector shape.")

    pos_mask = flows > 0
    selected_arc_indices = np.flatnonzero(pos_mask).astype(np.int64)
    pos_rows = rows_a[pos_mask]
    pos_cols = cols_b[pos_mask]
    pos_flows = flows[pos_mask]

    if pos_rows.size != nA:
        raise RuntimeError(
            f"Expected exactly {nA} positive-flow candidate arcs, but recovered {pos_rows.size}."
        )
    if not np.all(pos_flows == 1):
        raise RuntimeError("Candidate A->B arcs should all carry unit flow in the transport solution.")

    row_counts = np.bincount(pos_rows, minlength=nA).astype(np.int64)
    if not np.all(row_counts == 1):
        raise RuntimeError("Transport solution does not assign exactly one slice node to each aggregated node.")

    a_to_b = -np.ones(nA, dtype=np.int64)
    a_to_b[pos_rows] = pos_cols
    if np.any(a_to_b < 0):
        raise RuntimeError("Transport solution left at least one aggregated node unmatched.")

    col_counts = np.bincount(pos_cols, minlength=nB).astype(np.int64)
    if np.any(col_counts > slice_capacities):
        raise RuntimeError("Recovered transport assignment exceeds at least one slice-node capacity.")

    objective_integer = int(_ortools_optimal_cost(smcf))
    if original_costs is None:
        objective_float = float(np.sum(unit_costs[selected_arc_indices].astype(np.float64, copy=False)))
    else:
        objective_float = float(np.sum(original_costs[selected_arc_indices]))

    if output_dir is not None:
        _log_progress(
            output_dir,
            "matching_transport_done",
            f"[matching] Completed OR-Tools SimpleMinCostFlow solve ({solve_label}) in {elapsed:.1f}s.",
            payload={
                "solve_label": solve_label,
                "elapsed_seconds": float(elapsed),
                "objective_integer": int(objective_integer),
                "objective_float": float(objective_float),
            },
        )

    return {
        "a_to_b": a_to_b,
        "col_counts": col_counts,
        "objective_integer": int(objective_integer),
        "objective_value": float(objective_float),
        "status": status,
        "flows": flows,
        "selected_arc_indices": selected_arc_indices,
    }


def solve_capacitated_transport_ortools(
    costs: np.ndarray,
    rows_a: np.ndarray,
    cols_b: np.ndarray,
    indptr_a: np.ndarray,
    slice_capacities: np.ndarray,
    *,
    output_dir: str | None = None,
    solve_label: str = "base",
) -> dict[str, object]:
    """
    Backward-compatible float-cost wrapper around the exact sparse integer-cost
    OR-Tools solve.
    """
    costs = np.asarray(costs, dtype=np.float64).ravel()
    indptr_a = np.asarray(indptr_a, dtype=np.int64).ravel()
    nA = int(indptr_a.size - 1)

    unit_costs, cost_scale, cost_shift = _quantize_unit_costs_for_ortools(costs, total_flow=nA)
    if output_dir is not None:
        _log_progress(
            output_dir,
            "matching_transport_quantized",
            f"[matching] Quantized transport costs for {solve_label} with integer scale {int(cost_scale):,}.",
            payload={
                "solve_label": solve_label,
                "cost_scale": int(cost_scale),
                "cost_shift": float(cost_shift),
            },
        )

    out = solve_capacitated_transport_ortools_unit_costs(
        unit_costs,
        rows_a=rows_a,
        cols_b=cols_b,
        indptr_a=indptr_a,
        slice_capacities=slice_capacities,
        original_costs=costs,
        output_dir=output_dir,
        solve_label=solve_label,
    )
    out["cost_scale"] = int(cost_scale)
    out["cost_shift"] = float(cost_shift)
    return out


def _assignment_hash(a_to_slice: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(a_to_slice, dtype=np.int64))
    return hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()


def _compose_lexicographic_unit_costs(
    base_unit_costs: np.ndarray,
    secondary_unit_costs: np.ndarray,
    *,
    total_flow: int,
    objective_headroom: int = (1 << 60),
) -> tuple[np.ndarray, int, int]:
    """
    Compose base and secondary costs so the base sparse integer objective is
    minimized first, and the secondary term only breaks exact ties.
    """
    base = np.asarray(base_unit_costs, dtype=np.int64).ravel()
    sec = np.asarray(secondary_unit_costs, dtype=np.int64).ravel()

    if base.shape != sec.shape:
        raise ValueError("base and secondary costs must have the same shape.")
    if base.size == 0:
        return base.copy(), 1, 0
    if np.any(base < 0):
        raise ValueError("base_unit_costs must be nonnegative.")

    sec = sec - int(np.min(sec))
    requested_sec_max = int(np.max(sec))
    base_max = int(np.max(base))
    total_flow = int(max(1, total_flow))

    sec_max = requested_sec_max
    while sec_max > 0:
        M = total_flow * sec_max + 1
        max_arc_cost = base_max * M + sec_max
        if max_arc_cost * total_flow < int(objective_headroom):
            sec_eff = sec % (sec_max + 1)
            return (base * M + sec_eff).astype(np.int64), int(M), int(sec_max)
        sec_max //= 2

    return base.copy(), 1, 0


def _perturbed_unit_costs(
    base_unit_costs: np.ndarray,
    rng: np.random.Generator,
    perturb_units: int,
) -> np.ndarray:
    """
    Build a small integer-perturbed sparse objective for near-degenerate
    exploration. The returned costs are nonnegative and ready for OR-Tools.
    """
    base = np.asarray(base_unit_costs, dtype=np.int64).ravel()
    perturb_units = int(max(0, perturb_units))
    if perturb_units == 0:
        return base.copy()

    noise = rng.integers(
        -perturb_units,
        perturb_units + 1,
        size=base.size,
        dtype=np.int64,
    )
    out = base + noise
    min_cost = int(np.min(out)) if out.size else 0
    if min_cost < 0:
        # A constant shift per selected candidate arc does not change the sparse
        # optimum because every feasible solution selects exactly nA such arcs.
        out = out - min_cost
    return out.astype(np.int64, copy=False)


_ENSEMBLE_CONTEXT = None
_ENSEMBLE_REFINEMENT_CONTEXT = None


def _ensemble_worker_init(context_path: str, threads_per_worker: int = 1, refinement_context_path: str | None = None) -> None:
    global _ENSEMBLE_CONTEXT, _ENSEMBLE_REFINEMENT_CONTEXT
    _limit_runtime_threadpools(threads_per_worker)
    _ENSEMBLE_CONTEXT = load_matching_context_base(context_path)
    _ENSEMBLE_REFINEMENT_CONTEXT = (
        load_matching_refinement_context(refinement_context_path)
        if refinement_context_path
        else None
    )


def _refine_assignment_from_context(
    *,
    a_to_b_active: np.ndarray,
    base_ctx: dict[str, object],
    refine_ctx: dict[str, object],
    tree_workers: int | None = None,
    solve_label_prefix: str = "ensemble_refine",
) -> dict[str, object]:
    """Replay the same graph-regularized refinement loop for one assignment.

    Each iteration solves an exact sparse min-cost-flow problem on the current
    linearized feature + Dirichlet objective.  No dense A/B cost matrix is built.
    """
    rows = np.asarray(base_ctx["rows"], dtype=np.int64)
    cols = np.asarray(base_ctx["cols"], dtype=np.int64)
    indptr = np.asarray(base_ctx["indptr"], dtype=np.int64)
    slice_capacities_active = np.asarray(base_ctx["slice_capacities_active"], dtype=np.int64)

    W_graph = sp.csr_matrix(refine_ctx["W_graph"], dtype=np.float64)
    XA_features_01 = np.asarray(refine_ctx["XA_features_01"], dtype=np.float64)
    YB_features_active = np.asarray(refine_ctx["YB_features_active"], dtype=np.float64)
    YB_coords_active_graph = np.asarray(refine_ctx["YB_coords_active_graph"], dtype=np.float64)

    requested_refine_iter = int(refine_ctx.get("requested_refine_iter", 0))
    lam_dir_used = float(refine_ctx.get("lam_dir_used", 0.0))
    graph_candidate_k = int(refine_ctx.get("graph_candidate_k", 0))
    if requested_refine_iter <= 0 or lam_dir_used <= 0.0:
        return {
            "a_to_b": np.asarray(a_to_b_active, dtype=np.int64),
            "refinement_applied": False,
            "refinement_iteration_stats": [],
            "objective_integer_final": None,
            "objective_final": None,
        }

    nA = int(indptr.size - 1)
    nB_active = int(slice_capacities_active.size)
    graph_candidate_k = int(min(max(1, graph_candidate_k), max(1, nB_active)))
    prev = np.asarray(a_to_b_active, dtype=np.int64).copy()
    stats: list[dict[str, object]] = []
    objective_final = None
    objective_integer_final = None

    for it in range(requested_refine_iter):
        graph_rows, graph_cols, _graph_indptr, degw, sumy, q = build_graph_refinement_coordinate_candidates(
            W_graph,
            prev,
            YB_coords_active_graph,
            k=graph_candidate_k,
            workers=tree_workers,
        )
        rows_ref = np.concatenate([rows, graph_rows]).astype(np.int64, copy=False)
        cols_ref = np.concatenate([cols, graph_cols]).astype(np.int64, copy=False)
        indptr_ref = _indptr_from_row_indices(rows_ref, nA)

        base_costs_ref = compute_base_costs_a_to_b(
            XA_features_01,
            YB_features_active,
            rows_ref,
            cols_ref,
            weights=None,
            delta=0.05,
        )
        reg = _dirichlet_linearized_penalty_many_to_one(
            YB_coords_active_graph,
            rows_ref,
            cols_ref,
            degw,
            sumy,
            q,
        )
        costs = np.asarray(base_costs_ref + lam_dir_used * reg, dtype=np.float64)
        solve_t = solve_capacitated_transport_ortools(
            costs,
            rows_a=rows_ref,
            cols_b=cols_ref,
            indptr_a=indptr_ref,
            slice_capacities=slice_capacities_active,
            output_dir=None,
            solve_label=f"{solve_label_prefix}_{it + 1}",
        )
        new_a_to_b = np.asarray(solve_t["a_to_b"], dtype=np.int64)
        moved = new_a_to_b != prev
        stats.append({
            "iteration": int(it + 1),
            "n_reassigned": int(np.sum(moved)),
            "fraction_reassigned": float(np.mean(moved)),
            "augmented_candidate_arc_count": int(rows_ref.size),
            "objective_integer": int(solve_t["objective_integer"]),
        })
        prev = new_a_to_b
        objective_final = None if solve_t["objective_value"] is None else float(solve_t["objective_value"])
        objective_integer_final = int(solve_t["objective_integer"])
        if not np.any(moved):
            break

    return {
        "a_to_b": prev,
        "refinement_applied": True,
        "refinement_iteration_stats": stats,
        "objective_integer_final": objective_integer_final,
        "objective_final": objective_final,
    }


def _solve_ensemble_member(job: dict[str, object]) -> dict[str, object]:
    if _ENSEMBLE_CONTEXT is None:
        raise RuntimeError("Ensemble worker context was not initialized.")

    ctx = _ENSEMBLE_CONTEXT
    member_index = int(job["member_index"])
    seed = int(job["seed"])
    mode = str(job["mode"])
    tie_max = int(job["tie_max"])
    perturb_units = int(job["perturb_units"])

    rows = np.asarray(ctx["rows"], dtype=np.int64)
    cols = np.asarray(ctx["cols"], dtype=np.int64)
    indptr = np.asarray(ctx["indptr"], dtype=np.int64)
    base_costs = np.asarray(ctx["base_costs"], dtype=np.float64)
    base_unit_costs = np.asarray(ctx["base_unit_costs"], dtype=np.int64)
    slice_capacities_active = np.asarray(ctx["slice_capacities_active"], dtype=np.int64)
    active_raw_slice_indices = np.asarray(ctx["active_raw_slice_indices"], dtype=np.int64)
    YB_coords_raw = np.asarray(ctx["YB_coords_raw"], dtype=np.float64)

    nA = int(indptr.size - 1)
    rng = np.random.default_rng(seed + 1_000_003 * member_index)

    if mode == "base":
        unit_costs = base_unit_costs.copy()
        lex_multiplier = 1
        secondary_max = 0
    elif mode == "lexicographic":
        secondary = rng.integers(0, max(1, tie_max) + 1, size=base_unit_costs.size, dtype=np.int64)
        unit_costs, lex_multiplier, secondary_max = _compose_lexicographic_unit_costs(
            base_unit_costs,
            secondary,
            total_flow=nA,
        )
    elif mode == "perturb":
        unit_costs = _perturbed_unit_costs(base_unit_costs, rng, perturb_units)
        lex_multiplier = 1
        secondary_max = 0
    else:
        raise ValueError(f"Unknown ensemble mode: {mode!r}")

    solve = solve_capacitated_transport_ortools_unit_costs(
        unit_costs,
        rows_a=rows,
        cols_b=cols,
        indptr_a=indptr,
        slice_capacities=slice_capacities_active,
        original_costs=base_costs,
        output_dir=None,
        solve_label=f"ensemble_{member_index:04d}",
    )

    selected = np.asarray(solve["selected_arc_indices"], dtype=np.int64)
    a_to_b_transport = np.asarray(solve["a_to_b"], dtype=np.int64)
    a_to_b_final = a_to_b_transport
    refinement = {
        "refinement_applied": False,
        "refinement_iteration_stats": [],
        "objective_integer_final": None,
        "objective_final": None,
    }
    if _ENSEMBLE_REFINEMENT_CONTEXT is not None:
        refinement = _refine_assignment_from_context(
            a_to_b_active=a_to_b_transport,
            base_ctx=ctx,
            refine_ctx=_ENSEMBLE_REFINEMENT_CONTEXT,
            tree_workers=None,
            solve_label_prefix=f"ensemble_{member_index:04d}_refine",
        )
        a_to_b_final = np.asarray(refinement["a_to_b"], dtype=np.int64)

    a_to_slice_transport = np.asarray(active_raw_slice_indices[a_to_b_transport], dtype=np.int64)
    a_to_slice_final = np.asarray(active_raw_slice_indices[a_to_b_final], dtype=np.int64)
    coords_transport = np.asarray(YB_coords_raw[a_to_slice_transport], dtype=np.float64)
    coords_final = np.asarray(YB_coords_raw[a_to_slice_final], dtype=np.float64)

    base_integer = int(np.sum(base_unit_costs[selected]))
    base_float = float(np.sum(base_costs[selected]))

    return {
        "member_index": int(member_index),
        "mode": mode,
        "a_to_slice": a_to_slice_final,
        "a_to_slice_transport": a_to_slice_transport,
        "coords": coords_final,
        "coords_transport": coords_transport,
        "hash": _assignment_hash(a_to_slice_final),
        "transport_hash": _assignment_hash(a_to_slice_transport),
        "solver_integer_objective": int(solve["objective_integer"]),
        "base_integer_objective": int(base_integer),
        "base_float_objective": float(base_float),
        "refinement_applied": bool(refinement.get("refinement_applied", False)),
        "refinement_iteration_stats": refinement.get("refinement_iteration_stats", []),
        "refinement_integer_objective": refinement.get("objective_integer_final"),
        "refinement_float_objective": refinement.get("objective_final"),
        "lex_multiplier": int(lex_multiplier),
        "secondary_max": int(secondary_max),
    }


def run_transport_ensemble_from_context(
    *,
    context_path: str,
    output_dir: str,
    ensemble_size: int,
    ensemble_seed: int = 0,
    ensemble_mode: str = "lexicographic",
    ensemble_tie_max: int = 1023,
    ensemble_perturb_units: int = 0,
    ensemble_rel_tol: float = 0.0,
    ensemble_abs_tol: float = 0.0,
    ensemble_n_jobs: int | None = None,
    ensemble_threads_per_worker: int = 1,
    ensemble_mp_start_method: str | None = None,
    refinement_context_path: str | None = None,
) -> dict[str, object]:
    """
    Replay many exact sparse transport solves from the cached matching context.

    In lexicographic mode, every accepted member is an exact optimum of the
    original sparse integer objective. In perturb mode, each member is an exact
    optimum of its own perturbed sparse objective and is filtered by the original
    objective band.
    """
    ensure_dir(output_dir)
    if refinement_context_path is not None and not os.path.isfile(str(refinement_context_path)):
        raise FileNotFoundError("Requested ensemble graph refinement but missing refinement context: " + str(refinement_context_path))
    ensemble_size = int(max(1, ensemble_size))
    ensemble_mode = str(ensemble_mode)
    if ensemble_mode not in {"lexicographic", "perturb"}:
        raise ValueError("ensemble_mode must be either 'lexicographic' or 'perturb'.")

    jobs = [
        {
            "member_index": 0,
            "seed": int(ensemble_seed),
            "mode": "base",
            "tie_max": int(ensemble_tie_max),
            "perturb_units": int(ensemble_perturb_units),
        }
    ]
    for i in range(1, ensemble_size):
        jobs.append(
            {
                "member_index": int(i),
                "seed": int(ensemble_seed),
                "mode": ensemble_mode,
                "tie_max": int(ensemble_tie_max),
                "perturb_units": int(ensemble_perturb_units),
            }
        )

    allocated_cpus = max(1, _slurm_or_affinity_cpus())
    threads_per_worker = int(max(1, ensemble_threads_per_worker))

    # Default to serial replay.  A sparse transport member is not a light Python
    # task: it reloads the full matching context and constructs a full
    # SimpleMinCostFlow graph.  CPU-count defaults multiply memory and can OOM
    # large contexts.
    if ensemble_n_jobs is None or int(ensemble_n_jobs) <= 0:
        requested_workers = 1
    else:
        requested_workers = int(max(1, ensemble_n_jobs))

    requested_workers = min(
        len(jobs),
        requested_workers,
        max(1, allocated_cpus // threads_per_worker),
    )
    max_workers, memory_guard_meta = _cap_ensemble_workers_for_memory(
        context_path=context_path,
        requested_workers=requested_workers,
        output_dir=output_dir,
    )
    max_workers = int(max(1, min(len(jobs), max_workers)))
    mp_start_method = str(ensemble_mp_start_method or _default_ensemble_mp_start_method())

    t0 = time.time()
    _log_progress(
        output_dir,
        "ensemble_start",
        f"[ensemble] Solving {len(jobs):,} sparse transport members with {max_workers:,} worker(s).",
        payload={
            "ensemble_size": int(ensemble_size),
            "ensemble_mode": ensemble_mode,
            "ensemble_n_jobs": int(max_workers),
            "ensemble_threads_per_worker": int(threads_per_worker),
            "allocated_cpus": int(allocated_cpus),
            "ensemble_mp_start_method": str(mp_start_method),
            **memory_guard_meta,
        },
    )

    if max_workers == 1:
        # Serial replay remains in the parent process and keeps the parent thread
        # budget, so optimOps-style single-process performance is not impaired.
        saved_context = globals().get("_ENSEMBLE_CONTEXT")
        saved_refinement_context = globals().get("_ENSEMBLE_REFINEMENT_CONTEXT")
        globals()["_ENSEMBLE_CONTEXT"] = load_matching_context_base(context_path)
        globals()["_ENSEMBLE_REFINEMENT_CONTEXT"] = (
            load_matching_refinement_context(refinement_context_path)
            if refinement_context_path
            else None
        )
        try:
            raw_results = [_solve_ensemble_member(job) for job in jobs]
        finally:
            globals()["_ENSEMBLE_CONTEXT"] = saved_context
            globals()["_ENSEMBLE_REFINEMENT_CONTEXT"] = saved_refinement_context
    else:
        raw_results = []
        saved_env = _save_thread_env()
        _set_thread_env_for_current_process(threads_per_worker)
        try:
            mp_context = multiprocessing.get_context(mp_start_method)
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp_context,
                initializer=_ensemble_worker_init,
                initargs=(context_path, threads_per_worker, refinement_context_path),
            ) as pool:
                futures = [pool.submit(_solve_ensemble_member, job) for job in jobs]
                for fut in as_completed(futures):
                    raw_results.append(fut.result())
        finally:
            _restore_thread_env(saved_env)

    raw_results.sort(key=lambda r: (int(r["base_integer_objective"]), int(r["member_index"])))

    best_base_float = min(float(r["base_float_objective"]) for r in raw_results)
    best_base_integer = min(int(r["base_integer_objective"]) for r in raw_results)

    accepted = []
    seen_hashes = set()
    for result in raw_results:
        h = str(result["hash"])
        if h in seen_hashes:
            continue

        if ensemble_mode == "lexicographic" and float(ensemble_rel_tol) == 0.0 and float(ensemble_abs_tol) == 0.0:
            within_band = int(result["base_integer_objective"]) == int(best_base_integer)
        else:
            allowed = best_base_float + float(ensemble_abs_tol) + float(ensemble_rel_tol) * max(abs(best_base_float), EPS)
            within_band = float(result["base_float_objective"]) <= allowed

        if not within_band:
            continue

        accepted.append(result)
        seen_hashes.add(h)
        if len(accepted) >= ensemble_size:
            break

    if not accepted:
        raise RuntimeError("No ensemble solutions passed the objective-band filter.")

    coords_stack = np.stack([np.asarray(r["coords"], dtype=np.float64) for r in accepted], axis=0)
    coords_transport_stack = np.stack([np.asarray(r["coords_transport"], dtype=np.float64) for r in accepted], axis=0)
    a_to_slice_stack = np.stack([np.asarray(r["a_to_slice"], dtype=np.int64) for r in accepted], axis=0)
    a_to_slice_transport_stack = np.stack([np.asarray(r["a_to_slice_transport"], dtype=np.int64) for r in accepted], axis=0)
    objective_base_float = np.asarray([float(r["base_float_objective"]) for r in accepted], dtype=np.float64)
    objective_base_integer = np.asarray([int(r["base_integer_objective"]) for r in accepted], dtype=np.int64)
    member_indices = np.asarray([int(r["member_index"]) for r in accepted], dtype=np.int64)
    hashes = np.asarray([str(r["hash"]) for r in accepted], dtype="U32")

    ensemble_npz_path = os.path.join(output_dir, ENSEMBLE_COORDS_FILENAME)
    np.savez_compressed(
        ensemble_npz_path,
        coords_final=coords_stack,
        coords_transport=coords_transport_stack,
        a_to_slice=a_to_slice_stack,
        a_to_slice_transport=a_to_slice_transport_stack,
        objective_base_float=objective_base_float,
        objective_base_integer=objective_base_integer,
        member_indices=member_indices,
        solution_hashes=hashes,
        coords_frame=np.asarray(["raw_slice_spatial_xy"], dtype="U32"),
    )

    elapsed = time.time() - t0
    result_meta = {
        "ensemble_npz": ensemble_npz_path,
        "requested_ensemble_size": int(ensemble_size),
        "accepted_ensemble_size": int(len(accepted)),
        "raw_member_count": int(len(raw_results)),
        "unique_raw_member_count": int(len({str(r["hash"]) for r in raw_results})),
        "ensemble_mode": ensemble_mode,
        "ensemble_seed": int(ensemble_seed),
        "ensemble_tie_max": int(ensemble_tie_max),
        "ensemble_perturb_units": int(ensemble_perturb_units),
        "ensemble_rel_tol": float(ensemble_rel_tol),
        "ensemble_abs_tol": float(ensemble_abs_tol),
        "ensemble_n_jobs": int(max_workers),
        "ensemble_threads_per_worker": int(threads_per_worker),
        "refinement_context_npz": None if refinement_context_path is None else str(refinement_context_path),
        "graph_refinement_replayed": bool(refinement_context_path is not None),
        "accepted_refined_member_count": int(sum(bool(r.get("refinement_applied", False)) for r in accepted)),
        "allocated_cpus": int(allocated_cpus),
        "ensemble_mp_start_method": str(mp_start_method),
        "memory_guard": memory_guard_meta,
        "elapsed_seconds": float(elapsed),
        "best_base_float_objective": float(best_base_float),
        "best_base_integer_objective": int(best_base_integer),
    }

    _log_progress(
        output_dir,
        "ensemble_done",
        f"[ensemble] Accepted {len(accepted):,}/{ensemble_size:,} unique sparse transport solutions in {elapsed:.1f}s.",
        payload=result_meta,
    )
    return result_meta


def _matched_neighbor_moments_many_to_one(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    a_to_b: np.ndarray,
    YB_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nA = indptr.shape[0] - 1
    p = YB_coords.shape[1]
    degw = np.zeros(nA, dtype=np.float64)
    sumy = np.zeros((nA, p), dtype=np.float64)
    q = np.zeros(nA, dtype=np.float64)

    for i in range(nA):
        d_acc = 0.0
        q_acc = 0.0
        for t in range(indptr[i], indptr[i + 1]):
            u = indices[t]
            jb = a_to_b[u]
            if jb >= 0:
                w = data[t]
                d_acc += w
                norm2 = 0.0
                for z in range(p):
                    y = YB_coords[jb, z]
                    sumy[i, z] += w * y
                    norm2 += y * y
                q_acc += w * norm2
        degw[i] = d_acc
        q[i] = q_acc
    return degw, sumy, q

def _dirichlet_linearized_penalty_many_to_one(
    YB_coords: np.ndarray,
    rows_a: np.ndarray,
    cols_b: np.ndarray,
    degw: np.ndarray,
    sumy: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    m = rows_a.shape[0]
    p = YB_coords.shape[1]
    out = np.empty(m, dtype=np.float64)
    for e in prange(m):
        i = rows_a[e]
        j = cols_b[e]
        y_norm2 = 0.0
        dot = 0.0
        for z in range(p):
            y = YB_coords[j, z]
            y_norm2 += y * y
            dot += y * sumy[i, z]
        out[e] = degw[i] * y_norm2 - 2.0 * dot + q[i]
    return out

def _exact_dirichlet_energy_many_to_one(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    a_to_b: np.ndarray,
    YB_coords: np.ndarray,
) -> float:
    nA = indptr.shape[0] - 1
    p = YB_coords.shape[1]
    total = 0.0
    for i in range(nA):
        ji = a_to_b[i]
        if ji < 0:
            continue
        for t in range(indptr[i], indptr[i + 1]):
            u = indices[t]
            if u <= i:
                continue
            ju = a_to_b[u]
            if ju < 0:
                continue
            w = data[t]
            dist2 = 0.0
            for z in range(p):
                diff = YB_coords[ji, z] - YB_coords[ju, z]
                dist2 += diff * diff
            total += w * dist2
    return total

def matched_dirichlet_energy_many_to_one(
    W: sp.spmatrix | np.ndarray,
    a_to_b: np.ndarray,
    YB_coords: np.ndarray,
    symmetrize: bool = True,
) -> float:
    W = _cap_as_csr(W)
    if symmetrize:
        W = _cap_symmetrize_csr(W, average=True)
    return float(
        _exact_dirichlet_energy_many_to_one(
            W.indptr.astype(np.int64),
            W.indices.astype(np.int64),
            W.data.astype(np.float64),
            np.asarray(a_to_b, dtype=np.int64),
            np.asarray(YB_coords, dtype=np.float64),
        )
    )

def estimate_adaptive_dirichlet_lambda_many_to_one(
    W: csr_matrix,
    YB_coords: np.ndarray,
    rows_a: np.ndarray,
    cols_b: np.ndarray,
    a_to_b_init: np.ndarray,
    base_costs: np.ndarray,
) -> tuple[float, float, float]:
    W = _cap_symmetrize_csr(W, average=True)
    YB_coords = np.ascontiguousarray(np.asarray(YB_coords, dtype=np.float64))
    rows_a = np.asarray(rows_a, dtype=np.int64)
    cols_b = np.asarray(cols_b, dtype=np.int64)
    a_to_b_init = np.asarray(a_to_b_init, dtype=np.int64)
    base_costs = np.asarray(base_costs, dtype=np.float64)

    degw, sumy, q = _matched_neighbor_moments_many_to_one(
        W.indptr.astype(np.int64),
        W.indices.astype(np.int64),
        W.data.astype(np.float64),
        a_to_b_init,
        YB_coords,
    )
    reg = _dirichlet_linearized_penalty_many_to_one(YB_coords, rows_a, cols_b, degw, sumy, q)

    feature_scale = _finite_positive_median(base_costs)
    reg_scale = _finite_positive_median(reg)
    if reg_scale <= EPS:
        lam = 0.0
    elif feature_scale <= EPS:
        lam = 1.0 / reg_scale
    else:
        lam = feature_scale / reg_scale
    lam = float(np.clip(lam, 0.0, 1e12))
    return lam, feature_scale, reg_scale

def _normalize_graph_refinement_coordinates(
    coords: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    coords = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
    if coords.ndim != 2:
        raise ValueError(f"coords must be 2D, got shape {coords.shape}")
    center = np.nanmean(coords, axis=0)
    centered = coords - center[None, :]
    span = np.ptp(coords, axis=0)
    scale = float(np.linalg.norm(span))
    if not np.isfinite(scale) or scale <= EPS:
        scale = 1.0
    coords_norm = np.ascontiguousarray(centered / scale)
    return coords_norm, {
        "coord_center": center.tolist(),
        "coord_span": span.tolist(),
        "coord_scale": float(scale),
    }

def _normalize_dirichlet_graph_weights(
    W: csr_matrix,
) -> tuple[csr_matrix, dict[str, object]]:
    W = _cap_symmetrize_csr(W, average=True)
    total_undirected_weight = 0.5 * float(W.data.sum()) if W.nnz else 0.0
    if not np.isfinite(total_undirected_weight) or total_undirected_weight <= EPS:
        total_undirected_weight = 1.0
    W_norm = W.copy().astype(np.float64)
    W_norm.data /= float(total_undirected_weight)
    return W_norm, {
        "graph_edge_weight_total_undirected": float(total_undirected_weight),
    }

def _indptr_from_row_indices(rows: np.ndarray, n_rows: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64).ravel()
    counts = np.bincount(rows, minlength=int(n_rows)).astype(np.int64)
    indptr = np.empty(int(n_rows) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return indptr

def build_graph_refinement_coordinate_candidates(
    W_graph: csr_matrix,
    a_to_b: np.ndarray,
    YB_coords_graph: np.ndarray,
    *,
    k: int = 64,
    workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    YB_coords_graph = np.ascontiguousarray(np.asarray(YB_coords_graph, dtype=np.float64))
    a_to_b = np.asarray(a_to_b, dtype=np.int64).ravel()
    nA = int(W_graph.shape[0])
    nB = int(YB_coords_graph.shape[0])
    if a_to_b.shape[0] != nA:
        raise ValueError(f"a_to_b must have length {nA}, got {a_to_b.shape[0]}")
    k_eff = int(min(max(1, int(k)), nB))

    degw, sumy, q = _matched_neighbor_moments_many_to_one(
        W_graph.indptr.astype(np.int64),
        W_graph.indices.astype(np.int64),
        W_graph.data.astype(np.float64),
        a_to_b,
        YB_coords_graph,
    )

    targets = np.asarray(YB_coords_graph[a_to_b], dtype=np.float64).copy()
    mask = degw > EPS
    if np.any(mask):
        targets[mask] = sumy[mask] / degw[mask, None]

    tree = cKDTree(YB_coords_graph)
    _, idx = tree.query(targets, k=k_eff, workers=_resolve_tree_workers(workers))
    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim == 1:
        idx = idx[:, None]

    rows = np.repeat(np.arange(nA, dtype=np.int64), idx.shape[1])
    cols = idx.reshape(-1).astype(np.int64, copy=False)
    indptr = np.arange(0, rows.size + 1, idx.shape[1], dtype=np.int64)
    return rows, cols, indptr, degw, sumy, q

def _aggregate_assigned_features_to_slice_nodes(
    agg_features: np.ndarray,
    a_to_slice: np.ndarray,
    n_slice: int,
) -> tuple[np.ndarray, np.ndarray]:
    agg_features = np.asarray(agg_features, dtype=np.float64)
    a_to_slice = np.asarray(a_to_slice, dtype=np.int64).ravel()
    if agg_features.ndim == 1:
        agg_features = agg_features[:, None]
    p = int(agg_features.shape[1])
    counts = np.bincount(a_to_slice, minlength=int(n_slice)).astype(np.int64)
    sums = np.zeros((int(n_slice), p), dtype=np.float64)
    for d in range(p):
        np.add.at(sums[:, d], a_to_slice, agg_features[:, d])
    means = np.full((int(n_slice), p), np.nan, dtype=np.float64)
    mask = counts > 0
    if np.any(mask):
        means[mask] = sums[mask] / counts[mask, None]
    return means, counts


def write_slice_assigned_aggregated_feature_maps(
    output_dir: str,
    slice_coords: np.ndarray,
    agg_ratio_raw: np.ndarray,
    pair_ids: np.ndarray,
    a_to_slice_base: np.ndarray,
    a_to_slice_final: np.ndarray,
    slice_capacity: np.ndarray,
) -> str:
    slice_coords = np.asarray(slice_coords, dtype=np.float64)
    agg_ratio_raw = np.asarray(agg_ratio_raw, dtype=np.float64)
    pair_ids = np.asarray(pair_ids, dtype=object)
    a_to_slice_base = np.asarray(a_to_slice_base, dtype=np.int64)
    a_to_slice_final = np.asarray(a_to_slice_final, dtype=np.int64)
    slice_capacity = np.asarray(slice_capacity, dtype=np.int64)

    base_mean, base_count = _aggregate_assigned_features_to_slice_nodes(
        agg_ratio_raw,
        a_to_slice_base,
        n_slice=slice_coords.shape[0],
    )
    final_mean, final_count = _aggregate_assigned_features_to_slice_nodes(
        agg_ratio_raw,
        a_to_slice_final,
        n_slice=slice_coords.shape[0],
    )
    delta = final_mean - base_mean

    out_path = os.path.join(output_dir, "slice_assigned_aggregated_feature_maps.npz")
    np.savez_compressed(
        out_path,
        coords=np.asarray(slice_coords, dtype=np.float64),
        feature_mean_base=np.asarray(base_mean, dtype=np.float64),
        feature_mean_final=np.asarray(final_mean, dtype=np.float64),
        feature_delta=np.asarray(delta, dtype=np.float64),
        count_base=np.asarray(base_count, dtype=np.int64),
        count_final=np.asarray(final_count, dtype=np.int64),
        slice_capacity=np.asarray(slice_capacity, dtype=np.int64),
        pair_ids=np.asarray(pair_ids, dtype=object),
    )
    return out_path


def _hist2d_prob(coords: np.ndarray, *, ranges, bins: int = 32) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] == 0:
        h = np.ones(int(bins) * int(bins), dtype=np.float64)
        return h / float(h.sum())
    h, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=int(bins), range=ranges)
    h = h.ravel().astype(np.float64) + 1.0e-12
    return h / float(h.sum())


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    p = p / max(float(np.sum(p)), EPS)
    q = q / max(float(np.sum(q)), EPS)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], 1.0e-300))))
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def write_slice_capacity_spatial_diagnostics(
    output_dir: str,
    *,
    slice_coords: np.ndarray,
    slice_capacity: np.ndarray,
    assigned_coords: np.ndarray | None = None,
    capacity_mode: str = "mass_exact",
) -> str:
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "slice_capacity_spatial_diagnostics.json")
    coords = np.asarray(slice_coords, dtype=np.float64)
    cap = np.asarray(slice_capacity, dtype=np.int64).ravel()
    if coords.ndim != 2 or coords.shape[0] != cap.shape[0] or coords.shape[1] < 2:
        raise ValueError("slice_coords must be (n, >=2) and match slice_capacity length.")
    xy = coords[:, :2]
    active = cap > 0
    lo = np.nanmin(xy, axis=0)
    hi = np.nanmax(xy, axis=0)
    span = np.maximum(hi - lo, 1.0e-12)
    ranges = [[float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]]

    def _span(arr):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return [0.0, 0.0]
        return [float(x) for x in np.ptp(arr[:, :2], axis=0)]

    full_h = _hist2d_prob(xy, ranges=ranges)
    payload = {
        "capacity_mode": str(capacity_mode),
        "n_slice": int(xy.shape[0]),
        "capacity_sum": int(np.sum(cap)),
        "n_capacity_positive": int(np.sum(active)),
        "capacity_positive_fraction": float(np.mean(active)) if active.size else 0.0,
        "slice_xy_span": _span(xy),
        "active_xy_span": _span(xy[active]) if np.any(active) else [0.0, 0.0],
        "active_to_slice_span_ratio": (
            (np.asarray(_span(xy[active]), dtype=np.float64) / span).tolist()
            if np.any(active) else [0.0, 0.0]
        ),
        "active_vs_slice_js_divergence_32": (
            _js_divergence(full_h, _hist2d_prob(xy[active], ranges=ranges)) if np.sum(active) >= 2 else None
        ),
    }
    if assigned_coords is not None:
        assigned = np.asarray(assigned_coords, dtype=np.float64)
        if assigned.ndim == 2 and assigned.shape[0] > 0 and assigned.shape[1] >= 2:
            payload.update({
                "assigned_count": int(assigned.shape[0]),
                "assigned_xy_span": _span(assigned),
                "assigned_to_slice_span_ratio": (np.asarray(_span(assigned), dtype=np.float64) / span).tolist(),
                "assigned_vs_slice_js_divergence_32": _js_divergence(full_h, _hist2d_prob(assigned[:, :2], ranges=ranges)),
            })
    write_json(path, payload)
    return path


def run_sparse_graph_matching_on_ratio_vectors(
    agg_h5ad_path: str,
    XA_features_01: np.ndarray,
    YB_features_01: np.ndarray,
    YB_coords: np.ndarray,
    source_node_mass: np.ndarray,
    output_dir: str,
    k0: int = 16,
    k_max: int = 256,
    lam_dir: float | None = None,
    refine_iter: int = 1,
    tree_workers: int | None = None,
    slice_capacity_mode: str = "mass_exact",
):

    adjacency_path, W_raw = load_aggregated_adjacency_from_transformed_matrix(agg_h5ad_path)
    W_raw = _cap_symmetrize_csr(W_raw, average=True)
    W_graph, graph_norm_meta = _normalize_dirichlet_graph_weights(W_raw)

    XA_features_01 = np.ascontiguousarray(np.asarray(XA_features_01, dtype=np.float64))
    YB_features_01 = np.ascontiguousarray(np.asarray(YB_features_01, dtype=np.float64))
    YB_coords = np.ascontiguousarray(np.asarray(YB_coords, dtype=np.float64))
    source_node_mass = np.asarray(source_node_mass, dtype=np.float64).ravel()

    nA = int(XA_features_01.shape[0])
    nB_raw = int(YB_features_01.shape[0])
    if W_raw.shape[0] != nA:
        raise ValueError(f"Adjacency has {W_raw.shape[0]} rows but XA_features_01 has {nA} rows.")
    if YB_coords.shape[0] != nB_raw:
        raise ValueError(f"YB_coords has {YB_coords.shape[0]} rows but YB_features_01 has {nB_raw} rows.")
    if source_node_mass.shape[0] != nB_raw:
        raise ValueError(f"source_node_mass has {source_node_mass.shape[0]} rows but YB_features_01 has {nB_raw} rows.")

    capacity_info = compute_slice_capacities(
        source_node_mass,
        target_total=nA,
        coords=YB_coords,
        mode=slice_capacity_mode,
    )
    slice_capacities_raw = np.asarray(capacity_info["capacity"], dtype=np.int64)
    slice_expected_count_raw = np.asarray(capacity_info["expected"], dtype=np.float64)
    slice_capacity_mode_eff = str(capacity_info["mode"])
    slice_capacity_is_upper_bound = bool(capacity_info["is_upper_bound"])

    active = _active_slice_capacity_view(YB_features_01, YB_coords, slice_capacities_raw)
    active_raw_slice_indices = np.asarray(active["active_raw_slice_indices"], dtype=np.int64)
    YB_features_active = np.ascontiguousarray(active["YB_features_active"])
    YB_coords_active_raw = np.ascontiguousarray(active["YB_coords_active"])
    slice_capacities_active = np.asarray(active["slice_capacities_active"], dtype=np.int64)
    nB_active = int(slice_capacities_active.size)

    YB_coords_active_graph, coord_norm_meta = _normalize_graph_refinement_coordinates(YB_coords_active_raw)

    _log_progress(
        output_dir,
        "matching_setup",
        f"[matching] Preparing capacity-aware A->B transport on {nA:,} aggregated nodes and {nB_active:,} active slice nodes ({nB_raw - nB_active:,} zero-capacity slice nodes pruned from the matching core).",
        payload={
            "nA": nA,
            "nB_raw": nB_raw,
            "nB_active": nB_active,
            "n_zero_capacity_slice": int(nB_raw - nB_active),
            "slice_capacity_sum": int(np.sum(slice_capacities_raw)),
            "slice_capacity_mode": str(slice_capacity_mode_eff),
            "slice_capacity_is_upper_bound": bool(slice_capacity_is_upper_bound),
        },
    )

    tree = cKDTree(YB_features_active)
    soft_k_max = int(min(max(1, int(k_max)), nB_active))
    row_limits = np.full(nA, int(min(max(1, int(k0)), nB_active)), dtype=np.int64)

    final_rows = final_cols = final_indptr = final_base_costs = None
    final_row_limits = None
    feasibility_iters = 0
    selective_augmentation_used = False

    while True:
        feasibility_iters += 1
        rows_a, cols_b_active, _, indptr_a = build_variable_knn_candidates_a_to_b(
            XA_features_01,
            YB_features_active,
            row_k=row_limits,
            eps=0.0,
            p=2.0,
            workers=tree_workers,
            tree=tree,
        )

        feas = capacitated_matching_state(
            rows_a=rows_a,
            cols_b=cols_b_active,
            nA=nA,
            nB=nB_active,
            slice_capacities=slice_capacities_active,
        )
        _log_progress(
            output_dir,
            "matching_feasibility",
            f"[matching] Feasibility iteration {feasibility_iters}: flow {int(feas['flow_value']):,}/{nA:,} on {rows_a.size:,} candidate arcs; row-k median {float(np.median(row_limits)):.1f}, max {int(np.max(row_limits))}.",
            payload={
                "iteration": int(feasibility_iters),
                "flow_value": int(feas["flow_value"]),
                "nA": nA,
                "candidate_arc_count": int(rows_a.size),
                "row_k_max": int(np.max(row_limits)),
                "row_k_median": float(np.median(row_limits)),
                "blocked_row_count": int(np.asarray(feas["reachable_a"], dtype=np.int64).size),
            },
        )

        if bool(feas["full"]):
            final_rows = rows_a
            final_cols = cols_b_active
            final_indptr = indptr_a
            final_row_limits = row_limits.copy()
            final_base_costs = compute_base_costs_a_to_b(
                XA_features_01,
                YB_features_active,
                rows_a,
                cols_b_active,
                weights=None,
                delta=0.05,
            )
            break

        blocked_rows = np.asarray(feas["reachable_a"], dtype=np.int64)
        row_limits, changed = _augment_candidate_row_limits(
            row_limits=row_limits,
            blocked_rows=blocked_rows,
            nB=nB_active,
            soft_k_max=soft_k_max,
        )
        selective_augmentation_used = True

        if not changed:
            grow_mask = row_limits < int(nB_active)
            if not np.any(grow_mask):
                raise ValueError(
                    "No full capacitated A->B matching exists even after exhausting all active slice-node candidates."
                )
            row_limits[grow_mask] = np.minimum(int(nB_active), np.maximum(row_limits[grow_mask] + 64, row_limits[grow_mask] * 2))
            selective_augmentation_used = True

    base_unit_costs, base_cost_scale, base_cost_shift = _quantize_unit_costs_for_ortools(
        final_base_costs,
        total_flow=nA,
    )
    matching_context_path = write_matching_context_base(
        output_dir,
        rows=final_rows,
        cols=final_cols,
        indptr=final_indptr,
        base_costs=final_base_costs,
        base_unit_costs=base_unit_costs,
        base_cost_scale=base_cost_scale,
        base_cost_shift=base_cost_shift,
        slice_capacities_active=slice_capacities_active,
        active_raw_slice_indices=active_raw_slice_indices,
        YB_coords_raw=YB_coords,
        YB_coords_active_raw=YB_coords_active_raw,
        row_limits=final_row_limits,
    )
    _log_progress(
        output_dir,
        "matching_context_written",
        f"[matching] Wrote reusable sparse matching context to {matching_context_path}.",
        payload={
            "matching_context_npz": matching_context_path,
            "base_cost_scale": int(base_cost_scale),
            "base_cost_shift": float(base_cost_shift),
        },
    )

    solve0 = solve_capacitated_transport_ortools_unit_costs(
        base_unit_costs,
        rows_a=final_rows,
        cols_b=final_cols,
        indptr_a=final_indptr,
        slice_capacities=slice_capacities_active,
        original_costs=final_base_costs,
        output_dir=output_dir,
        solve_label="base",
    )
    solve0["cost_scale"] = int(base_cost_scale)
    solve0["cost_shift"] = float(base_cost_shift)
    a_to_b_active_base = np.asarray(solve0["a_to_b"], dtype=np.int64)
    a_to_b_active = a_to_b_active_base.copy()
    objective_initial = None if solve0["objective_value"] is None else float(solve0["objective_value"])
    objective_integer_initial = int(solve0["objective_integer"])
    cost_scale_used = int(solve0["cost_scale"])
    cost_shift_used = float(solve0["cost_shift"])

    requested_refine_iter = int(refine_iter)
    feature_cost_median = None
    dirichlet_penalty_median = None
    dirichlet_energy_before = None
    dirichlet_energy_after = None
    objective_final = objective_initial
    objective_integer_final = objective_integer_initial
    graph_regularization_used = False
    lam_dir_base = 0.0
    lam_dir_multiplier = 1.0 if lam_dir is None else float(lam_dir)
    lam_dir_used = 0.0
    refinement_iteration_stats: list[dict[str, object]] = []
    refinement_augmented_arc_count = None

    if requested_refine_iter > 0:
        lam_dir_base, feature_cost_median, dirichlet_penalty_median = estimate_adaptive_dirichlet_lambda_many_to_one(
            W=W_graph,
            YB_coords=YB_coords_active_graph,
            rows_a=final_rows,
            cols_b=final_cols,
            a_to_b_init=a_to_b_active,
            base_costs=final_base_costs,
        )
        lam_dir_used = float(max(0.0, lam_dir_multiplier * lam_dir_base))
        if lam_dir is not None and float(lam_dir) == 0.0:
            lam_dir_used = 0.0

        if float(lam_dir_used) > 0.0:
            dirichlet_energy_before = matched_dirichlet_energy_many_to_one(
                W_graph,
                a_to_b=a_to_b_active,
                YB_coords=YB_coords_active_graph,
                symmetrize=False,
            )
            prev = a_to_b_active.copy()
            graph_candidate_k = int(min(max(32, int(np.ceil(np.sqrt(max(nB_active, 1))))), nB_active))
            for it in range(requested_refine_iter):
                _log_progress(
                    output_dir,
                    "matching_refine_iter",
                    f"[matching] Graph-refinement reweighting iteration {it + 1}/{requested_refine_iter} (lambda multiplier {lam_dir_multiplier:.3g}, effective lambda {lam_dir_used:.3g}).",
                    payload={
                        "iteration": int(it + 1),
                        "requested_refine_iter": int(requested_refine_iter),
                        "lam_dir_multiplier": float(lam_dir_multiplier),
                        "lam_dir_base": float(lam_dir_base),
                        "lam_dir_used": float(lam_dir_used),
                        "graph_candidate_k": int(graph_candidate_k),
                    },
                )
                graph_rows, graph_cols, _graph_indptr, degw, sumy, q = build_graph_refinement_coordinate_candidates(
                    W_graph,
                    prev,
                    YB_coords_active_graph,
                    k=graph_candidate_k,
                    workers=tree_workers,
                )
                rows_ref = np.concatenate([final_rows, graph_rows]).astype(np.int64, copy=False)
                cols_ref = np.concatenate([final_cols, graph_cols]).astype(np.int64, copy=False)
                indptr_ref = _indptr_from_row_indices(rows_ref, nA)
                refinement_augmented_arc_count = int(rows_ref.size)

                base_costs_ref = compute_base_costs_a_to_b(
                    XA_features_01,
                    YB_features_active,
                    rows_ref,
                    cols_ref,
                    weights=None,
                    delta=0.05,
                )
                reg = _dirichlet_linearized_penalty_many_to_one(YB_coords_active_graph, rows_ref, cols_ref, degw, sumy, q)
                costs = np.asarray(base_costs_ref + float(lam_dir_used) * reg, dtype=np.float64)
                solve_t = solve_capacitated_transport_ortools(
                    costs,
                    rows_a=rows_ref,
                    cols_b=cols_ref,
                    indptr_a=indptr_ref,
                    slice_capacities=slice_capacities_active,
                    output_dir=output_dir,
                    solve_label=f"refine_{it + 1}",
                )
                new_a_to_b = np.asarray(solve_t["a_to_b"], dtype=np.int64)
                objective_final = None if solve_t["objective_value"] is None else float(solve_t["objective_value"])
                objective_integer_final = int(solve_t["objective_integer"])
                cost_scale_used = int(solve_t["cost_scale"])
                cost_shift_used = float(solve_t["cost_shift"])

                moved = new_a_to_b != prev
                move_dist = np.linalg.norm(
                    YB_coords_active_raw[new_a_to_b] - YB_coords_active_raw[prev],
                    axis=1,
                )
                refinement_iteration_stats.append({
                    "iteration": int(it + 1),
                    "n_reassigned": int(np.sum(moved)),
                    "fraction_reassigned": float(np.mean(moved)),
                    "median_move_distance": float(np.median(move_dist[moved])) if np.any(moved) else 0.0,
                    "max_move_distance": float(np.max(move_dist[moved])) if np.any(moved) else 0.0,
                    "augmented_candidate_arc_count": int(refinement_augmented_arc_count),
                })
                prev = new_a_to_b
                if not np.any(moved):
                    break
            a_to_b_active = prev
            dirichlet_energy_after = matched_dirichlet_energy_many_to_one(
                W_graph,
                a_to_b=a_to_b_active,
                YB_coords=YB_coords_active_graph,
                symmetrize=False,
            )
            graph_regularization_used = True

    a_to_slice_base = np.asarray(active_raw_slice_indices[a_to_b_active_base], dtype=np.int64)
    a_to_slice = np.asarray(active_raw_slice_indices[a_to_b_active], dtype=np.int64)

    aggregated_to_slice_mapping_csr = _one_hot_csr_from_pairs(
        rows=np.arange(nA, dtype=np.int64),
        cols=a_to_slice,
        shape=(nA, nB_raw),
    )
    aggregated_to_slice_mapping_path = os.path.join(output_dir, "aggregated_to_slice_match_csr.npz")
    sp.save_npz(aggregated_to_slice_mapping_path, aggregated_to_slice_mapping_csr)

    slice_to_aggregated_mapping_csr = _one_hot_csr_from_pairs(
        rows=a_to_slice,
        cols=np.arange(nA, dtype=np.int64),
        shape=(nB_raw, nA),
    )
    slice_to_aggregated_mapping_path = os.path.join(output_dir, "slice_to_aggregated_match_csr.npz")
    sp.save_npz(slice_to_aggregated_mapping_path, slice_to_aggregated_mapping_csr)

    capacity_path = os.path.join(output_dir, "slice_capacity_targets.npz")
    np.savez_compressed(
        capacity_path,
        slice_total_mass=np.asarray(source_node_mass, dtype=np.float64),
        slice_capacity=np.asarray(slice_capacities_raw, dtype=np.int64),
        slice_expected_count=np.asarray(slice_expected_count_raw, dtype=np.float64),
        active_raw_slice_indices=np.asarray(active_raw_slice_indices, dtype=np.int64),
        active_slice_capacity=np.asarray(slice_capacities_active, dtype=np.int64),
        slice_capacity_mode=np.asarray([slice_capacity_mode_eff], dtype=object),
        slice_capacity_is_upper_bound=np.asarray([bool(slice_capacity_is_upper_bound)], dtype=np.uint8),
    )

    mapped_slice_coords_base = np.asarray(YB_coords[a_to_slice_base], dtype=np.float64)
    mapped_slice_coords = np.asarray(YB_coords[a_to_slice], dtype=np.float64)
    slice_capacity_diagnostics_path = write_slice_capacity_spatial_diagnostics(
        output_dir,
        slice_coords=YB_coords,
        slice_capacity=slice_capacities_raw,
        assigned_coords=mapped_slice_coords,
        capacity_mode=slice_capacity_mode_eff,
    )
    matched_mask = np.ones(nA, dtype=bool)
    moved_mask = a_to_slice != a_to_slice_base
    move_distance = np.linalg.norm(mapped_slice_coords - mapped_slice_coords_base, axis=1)
    coord_scale_raw = float(coord_norm_meta["coord_scale"])
    if not np.isfinite(coord_scale_raw) or coord_scale_raw <= EPS:
        coord_scale_raw = 1.0
    move_distance_normalized = move_distance / coord_scale_raw

    mapped_slice_coords_path = os.path.join(output_dir, "aggregated_nodes_slice_mapped_coords.npz")
    np.savez_compressed(
        mapped_slice_coords_path,
        coords_base=np.asarray(mapped_slice_coords_base, dtype=np.float64),
        coords_final=np.asarray(mapped_slice_coords, dtype=np.float64),
        # Keep the exact raw-slice row witnesses that generated each coordinate
        # block.  This makes downstream frame checks deterministic without
        # rebuilding features, candidate arcs, or the min-cost-flow graph.
        a_to_slice_base=np.asarray(a_to_slice_base, dtype=np.int64),
        a_to_slice_final=np.asarray(a_to_slice, dtype=np.int64),
        a_to_slice=np.asarray(a_to_slice, dtype=np.int64),
        coords_frame=np.asarray(["raw_slice_spatial_xy"], dtype="U32"),
        moved_mask=np.asarray(moved_mask, dtype=np.uint8),
        move_distance=np.asarray(move_distance, dtype=np.float64),
        move_distance_normalized=np.asarray(move_distance_normalized, dtype=np.float64),
    )

    graph_candidate_k_final = int(min(max(32, int(np.ceil(np.sqrt(max(nB_active, 1))))), nB_active))
    matching_refinement_context_path = None
    if requested_refine_iter > 0 and lam_dir_used > 0.0:
        matching_refinement_context_path = write_matching_refinement_context(
            output_dir,
            W_graph=W_graph,
            XA_features_01=XA_features_01,
            YB_features_active=YB_features_active,
            YB_coords_active_graph=YB_coords_active_graph,
            requested_refine_iter=requested_refine_iter,
            lam_dir_used=lam_dir_used,
            graph_candidate_k=graph_candidate_k_final,
        )

    row_limits_final = np.asarray(final_row_limits, dtype=np.int64)
    effective_row_k_max = int(np.max(row_limits_final))
    effective_row_k_median = float(np.median(row_limits_final))
    effective_edge_count = int(final_rows.size)
    n_reassigned_total = int(np.sum(moved_mask))
    fraction_reassigned_total = float(np.mean(moved_mask))
    median_move_distance = float(np.median(move_distance[moved_mask])) if np.any(moved_mask) else 0.0
    median_move_distance_normalized = float(np.median(move_distance_normalized[moved_mask])) if np.any(moved_mask) else 0.0
    max_move_distance = float(np.max(move_distance[moved_mask])) if np.any(moved_mask) else 0.0

    return {
        "adjacency_path": adjacency_path,
        "slice_capacity_path": capacity_path,
        "slice_capacity_spatial_diagnostics_path": slice_capacity_diagnostics_path,
        "slice_to_aggregated_mapping_path": slice_to_aggregated_mapping_path,
        "slice_to_aggregated_mapping_csr": slice_to_aggregated_mapping_csr,
        "aggregated_to_slice_mapping_path": aggregated_to_slice_mapping_path,
        "aggregated_to_slice_mapping_csr": aggregated_to_slice_mapping_csr,
        "mapped_slice_coords_path": mapped_slice_coords_path,
        "matching_context_path": matching_context_path,
        "matching_refinement_context_path": matching_refinement_context_path,
        "mapped_slice_coords": mapped_slice_coords,
        "matched_mask": matched_mask,
        "a_to_slice": np.asarray(a_to_slice, dtype=np.int64),
        "a_to_slice_base": np.asarray(a_to_slice_base, dtype=np.int64),
        "slice_capacity": np.asarray(slice_capacities_raw, dtype=np.int64),
        "slice_capacity_active": np.asarray(slice_capacities_active, dtype=np.int64),
        "slice_expected_count": np.asarray(slice_expected_count_raw, dtype=np.float64),
        "slice_capacity_mode": str(slice_capacity_mode_eff),
        "slice_capacity_is_upper_bound": bool(slice_capacity_is_upper_bound),
        "source_node_mass": np.asarray(source_node_mass, dtype=np.float64),
        "k": int(effective_row_k_max),
        "rows": np.asarray(final_rows, dtype=np.int64),
        "cols": np.asarray(final_cols, dtype=np.int64),
        "indptr": np.asarray(final_indptr, dtype=np.int64),
        "base_costs": np.asarray(final_base_costs, dtype=np.float64),
        "lam_dir_multiplier": float(lam_dir_multiplier),
        "lam_dir_base": float(lam_dir_base),
        "lam_dir_used": float(lam_dir_used),
        "refine_iter_used": int(requested_refine_iter),
        "graph_regularization_used": bool(graph_regularization_used),
        "feature_cost_median": None if feature_cost_median is None else float(feature_cost_median),
        "dirichlet_penalty_median": None if dirichlet_penalty_median is None else float(dirichlet_penalty_median),
        "dirichlet_energy_before": None if dirichlet_energy_before is None else float(dirichlet_energy_before),
        "dirichlet_energy_after": None if dirichlet_energy_after is None else float(dirichlet_energy_after),
        "objective_initial": None if objective_initial is None else float(objective_initial),
        "objective_final": None if objective_final is None else float(objective_final),
        "objective_integer_initial": int(objective_integer_initial),
        "objective_integer_final": int(objective_integer_final),
        "transport_solver": "ortools.graph.python.min_cost_flow.SimpleMinCostFlow",
        "transport_cost_scale": int(cost_scale_used),
        "transport_cost_shift": float(cost_shift_used),
        "row_limits": row_limits_final,
        "effective_row_k_max": int(effective_row_k_max),
        "effective_row_k_median": float(effective_row_k_median),
        "effective_edge_count": int(effective_edge_count),
        "feasibility_iterations": int(feasibility_iters),
        "selective_augmentation_used": bool(selective_augmentation_used),
        "soft_k_max": int(soft_k_max),
        "soft_k_max_exceeded": bool(effective_row_k_max > soft_k_max),
        "active_raw_slice_indices": np.asarray(active_raw_slice_indices, dtype=np.int64),
        "n_active_slice_nodes": int(nB_active),
        "n_zero_capacity_slice_nodes": int(nB_raw - nB_active),
        "graph_candidate_k": int(graph_candidate_k_final),
        "refinement_augmented_arc_count": None if refinement_augmented_arc_count is None else int(refinement_augmented_arc_count),
        "n_reassigned_by_graph_refinement": int(n_reassigned_total),
        "fraction_reassigned_by_graph_refinement": float(fraction_reassigned_total),
        "median_move_distance": float(median_move_distance),
        "median_move_distance_normalized": float(median_move_distance_normalized),
        "max_move_distance": float(max_move_distance),
        "refinement_iteration_stats": refinement_iteration_stats,
        "graph_coord_normalization": coord_norm_meta,
        "graph_weight_normalization": graph_norm_meta,
    }


def run_pipeline(args):
    ensure_dir(args.output)
    _log_progress(args.output, "startup", "Pipeline started.")

    _require_ortools_min_cost_flow()

    slice_abs = _norm_path(args.slice)
    agg_abs = _norm_path(args.agg)
    request_signature = _build_request_signature(
        slice_h5ad_path=slice_abs,
        agg_h5ad_path=agg_abs,
        time_value=args.time_value,
        no_time_filter=args.no_time_filter,
        num_pole_pairs=args.num_pole_pairs,
        genes_per_pole=args.genes_per_pole,
        abundance_threshold=args.abundance_threshold,
        min_feature_count=args.min_feature_count,
        slice_smooth_k=args.slice_smooth_k,
        agg_smooth_k=args.agg_smooth_k,
        rank_neutral=args.rank_neutral,
        match_k0=args.match_k0,
        match_k_max=args.match_k_max,
        match_lam_dir=args.match_lam_dir,
        match_refine_iter=args.match_refine_iter,
        pole_pairs_json=getattr(args, "pole_pairs_json", None),
        slice_capacity_mode=getattr(args, "slice_capacity_mode", "mass_exact"),
    )

    _log_progress(args.output, "load_source", f"Loading source slice AnnData from {args.slice}.")
    source_adata = load_source_adata(
        args.slice,
        time_value=args.time_value,
        no_time_filter=args.no_time_filter,
    )
    source_coords = get_spatial_coords_from_adata(source_adata)
    source_node_mass = get_node_total_mass(source_adata)
    shared_source_smoother, _ = build_knn_smoothing_operator(source_coords, GENE_POLE_SMOOTH_K)
    _log_progress(
        args.output,
        "load_source_done",
        f"Loaded source slice with {source_coords.shape[0]:,} nodes.",
        payload={"n_slice": int(source_coords.shape[0])},
    )

    _log_progress(args.output, "load_agg", f"Loading aggregated h5ad from {args.agg}.")
    agg_dataset = load_aggregated_gse_h5ad(args.agg)
    _log_progress(
        args.output,
        "load_agg_done",
        f"Loaded aggregated dataset with {int(agg_dataset['n_supernodes']):,} nodes.",
        payload={"n_agg": int(agg_dataset['n_supernodes'])},
    )

    shared_genes = collect_shared_genes(
        agg_dataset,
        source_adata,
        min_feature_count=args.min_feature_count,
    )
    _log_progress(
        args.output,
        "shared_genes",
        f"Collected {len(shared_genes):,} shared genes; discovering pole pairs.",
        payload={"shared_gene_count": int(len(shared_genes))},
    )

    pole_pairs_json = getattr(args, "pole_pairs_json", None)
    if pole_pairs_json:
        pole_pairs = load_explicit_pole_pairs_json(
            pole_pairs_json,
            shared_genes,
            num_pairs=args.num_pole_pairs,
            genes_per_pole=args.genes_per_pole,
        )
        pole_pair_selection_mode = "explicit_json"
        write_pole_pairs_outputs(args.output, pole_pairs)
        _log_progress(
            args.output,
            "pole_pairs_done",
            f"Loaded {len(pole_pairs):,} explicit pole pairs from JSON; computing slice ratio fields.",
            payload={
                "num_pole_pairs_selected": int(len(pole_pairs)),
                "pole_pair_selection_mode": pole_pair_selection_mode,
                "pole_pairs_json": _norm_path(pole_pairs_json),
            },
        )
    else:
        pole_pair_selection_mode = "spatial_discovery"
        pole_pairs = identify_typeAB_gene_pairs(
            source_adata,
            shared_genes,
            args.output,
            num_pairs=args.num_pole_pairs,
            num_top_genes=args.genes_per_pole,
            abundance_threshold=args.abundance_threshold,
            coords=source_coords,
            smoothing_operator=shared_source_smoother,
        )
        _log_progress(
            args.output,
            "pole_pairs_done",
            f"Selected {len(pole_pairs):,} pole pairs; computing slice ratio fields.",
            payload={
                "num_pole_pairs_selected": int(len(pole_pairs)),
                "pole_pair_selection_mode": pole_pair_selection_mode,
            },
        )

    source_field = compute_source_ratio_fields_multi(
        source_adata,
        pole_pairs,
        coords=source_coords,
        smooth_k=args.slice_smooth_k,
        smoothing_operator=shared_source_smoother if int(args.slice_smooth_k) == int(GENE_POLE_SMOOTH_K) else None,
    )
    source_ratio_rank = rank_transform_ratio_fields(
        source_field["ratio"],
        source_field["support"],
        neutral=args.rank_neutral,
    )
    source_feature_01 = rescale_rank_features_to_unit_interval(
        source_ratio_rank,
        source_field["support"],
        neutral=args.rank_neutral,
    )
    _log_progress(args.output, "slice_features_done", "Computed slice ratio features.")

    agg_coords, typeA_counts, typeB_counts = build_pair_count_matrices_from_aggregated(agg_dataset, pole_pairs)
    agg_ratio_raw, agg_support, agg_knn_meta = perform_knn_analysis_with_support_multi(
        agg_coords,
        typeA_counts,
        typeB_counts,
        k=args.agg_smooth_k,
    )
    agg_ratio_rank = rank_transform_ratio_fields(agg_ratio_raw, agg_support, neutral=args.rank_neutral)
    agg_feature_01 = rescale_rank_features_to_unit_interval(
        agg_ratio_rank,
        agg_support,
        neutral=args.rank_neutral,
    )
    _log_progress(args.output, "agg_features_done", "Computed aggregated ratio features; writing pre-match feature checkpoints.")

    source_npz = os.path.join(args.output, "slice_smoothed_ratio_fields.npz")
    np.savez_compressed(
        source_npz,
        coords=np.asarray(source_field["coords"], dtype=np.float64),
        ratio=np.asarray(source_field["ratio"], dtype=np.float64),
        ratio_rank=np.asarray(source_ratio_rank, dtype=np.float64),
        ratio_feature_01=np.asarray(source_feature_01, dtype=np.float64),
        support=np.asarray(source_field["support"], dtype=np.float64),
        typeA_signal=np.asarray(source_field["typeA_signal"], dtype=np.float64),
        typeB_signal=np.asarray(source_field["typeB_signal"], dtype=np.float64),
        node_total_mass=np.asarray(source_node_mass, dtype=np.float64),
        pair_ids=np.asarray(source_field["pair_ids"], dtype=object),
    )

    agg_npz = os.path.join(args.output, "aggregated_gse_ranked_ratio_vectors.npz")
    np.savez_compressed(
        agg_npz,
        coords=np.asarray(agg_coords, dtype=np.float64),
        ratio_raw=np.asarray(agg_ratio_raw, dtype=np.float64),
        ratio_rank=np.asarray(agg_ratio_rank, dtype=np.float64),
        ratio_feature_01=np.asarray(agg_feature_01, dtype=np.float64),
        support=np.asarray(agg_support, dtype=np.float64),
        typeA_counts=np.asarray(typeA_counts, dtype=np.float64),
        typeB_counts=np.asarray(typeB_counts, dtype=np.float64),
        pair_ids=np.asarray(source_field["pair_ids"], dtype=object),
    )
    _log_progress(
        args.output,
        "pre_match_checkpoints_written",
        "Wrote slice_smoothed_ratio_fields.npz and aggregated_gse_ranked_ratio_vectors.npz; starting OR-Tools capacitated transport matching.",
        payload={"slice_output_npz": source_npz, "aggregated_output_npz": agg_npz},
    )

    match_result = run_sparse_graph_matching_on_ratio_vectors(
        agg_h5ad_path=args.agg,
        XA_features_01=agg_feature_01,
        YB_features_01=source_feature_01,
        YB_coords=source_coords,
        source_node_mass=source_node_mass,
        output_dir=args.output,
        k0=args.match_k0,
        k_max=args.match_k_max,
        lam_dir=args.match_lam_dir,
        refine_iter=args.match_refine_iter,
        tree_workers=getattr(args, "tree_workers", None),
        slice_capacity_mode=getattr(args, "slice_capacity_mode", "mass_exact"),
    )

    # The returned match_result intentionally contains the large sparse context
    # arrays for programmatic callers.  run_pipeline itself no longer needs them
    # after writing the base outputs, and keeping them resident while launching
    # ensemble replay can double the working set.
    for _heavy_key in ("rows", "cols", "indptr", "base_costs"):
        match_result.pop(_heavy_key, None)
    try:
        import gc as _gc
        _gc.collect()
    except Exception:
        pass

    ensemble_result = None
    if int(getattr(args, "ensemble_size", 1)) > 1:
        ensemble_result = run_transport_ensemble_from_context(
            context_path=match_result["matching_context_path"],
            output_dir=args.output,
            ensemble_size=int(getattr(args, "ensemble_size", 1)),
            ensemble_seed=int(getattr(args, "ensemble_seed", 0)),
            ensemble_mode=str(getattr(args, "ensemble_mode", "lexicographic")),
            ensemble_tie_max=int(getattr(args, "ensemble_tie_max", 1023)),
            ensemble_perturb_units=int(getattr(args, "ensemble_perturb_units", 0)),
            ensemble_rel_tol=float(getattr(args, "ensemble_rel_tol", 0.0)),
            ensemble_abs_tol=float(getattr(args, "ensemble_abs_tol", 0.0)),
            ensemble_n_jobs=getattr(args, "ensemble_n_jobs", None),
            ensemble_threads_per_worker=int(getattr(args, "ensemble_threads_per_worker", 1)),
            ensemble_mp_start_method=getattr(args, "ensemble_mp_start_method", None),
            refinement_context_path=match_result.get("matching_refinement_context_path"),
        )
        ensemble_signature = _build_ensemble_signature(
            base_request_signature=request_signature,
            ensemble_size=int(getattr(args, "ensemble_size", 1)),
            ensemble_seed=int(getattr(args, "ensemble_seed", 0)),
            ensemble_mode=str(getattr(args, "ensemble_mode", "lexicographic")),
            ensemble_tie_max=int(getattr(args, "ensemble_tie_max", 1023)),
            ensemble_perturb_units=int(getattr(args, "ensemble_perturb_units", 0)),
            ensemble_rel_tol=float(getattr(args, "ensemble_rel_tol", 0.0)),
            ensemble_abs_tol=float(getattr(args, "ensemble_abs_tol", 0.0)),
        )
        ensemble_signature["refinement_context_npz"] = (
            None if match_result.get("matching_refinement_context_path") is None
            else os.path.basename(str(match_result.get("matching_refinement_context_path")))
        )
        write_json(
            os.path.join(args.output, ENSEMBLE_METADATA_FILENAME),
            {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "ensemble_signature": ensemble_signature,
                "ensemble_result": ensemble_result,
            },
        )

    slice_feature_maps_path = write_slice_assigned_aggregated_feature_maps(
        output_dir=args.output,
        slice_coords=source_coords,
        agg_ratio_raw=agg_ratio_raw,
        pair_ids=np.asarray(source_field["pair_ids"], dtype=object),
        a_to_slice_base=np.asarray(match_result["a_to_slice_base"], dtype=np.int64),
        a_to_slice_final=np.asarray(match_result["a_to_slice"], dtype=np.int64),
        slice_capacity=np.asarray(match_result["slice_capacity"], dtype=np.int64),
    )

    slice_capacity = np.asarray(match_result["slice_capacity"], dtype=np.int64)
    matched_mask = np.asarray(match_result["matched_mask"], dtype=bool)

    metadata = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "mode": PIPELINE_MODE,
        "source_adata": slice_abs,
        "aggregated_h5ad": agg_abs,
        "request_signature": request_signature,
        "time_value": str(args.time_value).strip().lower(),
        "no_time_filter": bool(args.no_time_filter),
        "num_pole_pairs_selected": int(len(pole_pairs)),
        "num_pole_pairs_requested": int(args.num_pole_pairs),
        "genes_per_pole": int(args.genes_per_pole),
        "pole_pair_selection_mode": str(pole_pair_selection_mode),
        "pole_pairs_json": None if not getattr(args, "pole_pairs_json", None) else _norm_path(getattr(args, "pole_pairs_json")),
        "abundance_threshold": int(args.abundance_threshold),
        "min_feature_count": int(args.min_feature_count),
        "slice_smooth_k": int(args.slice_smooth_k),
        "agg_smooth_k": int(args.agg_smooth_k),
        "rank_transform": {
            "applied_to": "slice and aggregated ratio field columns",
            "rule": (
                "normalized ranks on support>0 entries, then direct per-column rescaling of those "
                "ranked values to [0, 1]; support==0 entries remain at the neutral value"
            ),
            "neutral_value": float(args.rank_neutral),
        },
        "matching": {
            "solver": match_result["transport_solver"],
            "aggregated_adjacency_path": match_result["adjacency_path"],
            "set_A": "aggregated supernodes in raw h5ad order",
            "set_B": "raw slice observations in raw h5ad order",
            "feature_space": "rank-transformed, directly rescaled multi-pair ratio vectors in [0, 1]",
            "b_coordinates_for_graph_refinement": "centered and scale-normalized raw slice spatial_x/spatial_y coordinates (raw coordinates retained in outputs)",
            "slice_capacity_targets_npz": match_result["slice_capacity_path"],
            "slice_capacity_spatial_diagnostics_json": match_result.get("slice_capacity_spatial_diagnostics_path"),
            "slice_capacity_mode": str(match_result.get("slice_capacity_mode", getattr(args, "slice_capacity_mode", "mass_exact"))),
            "slice_capacity_is_upper_bound": bool(match_result.get("slice_capacity_is_upper_bound", False)),
            "slice_to_aggregated_mapping_csr_npz": match_result["slice_to_aggregated_mapping_path"],
            "aggregated_to_slice_mapping_csr_npz": match_result["aggregated_to_slice_mapping_path"],
            "aggregated_slice_mapped_coordinates_npz": match_result["mapped_slice_coords_path"],
            "matching_context_npz": match_result["matching_context_path"],
            "matching_refinement_context_npz": match_result.get("matching_refinement_context_path"),
            "slice_assigned_aggregated_feature_maps_npz": slice_feature_maps_path,
            "ensemble": ensemble_result,
            "raw_slice_node_count": int(source_coords.shape[0]),
            "active_slice_node_count": int(match_result["n_active_slice_nodes"]),
            "zero_capacity_slice_node_count": int(match_result["n_zero_capacity_slice_nodes"]),
            "slice_capacity_summary": {
                "min": int(np.min(slice_capacity)),
                "median": float(np.median(slice_capacity)),
                "max": int(np.max(slice_capacity)),
                "sum": int(np.sum(slice_capacity)),
            },
            "requested_k0": int(args.match_k0),
            "requested_k_max": int(args.match_k_max),
            "final_effective_row_k_max": int(match_result["effective_row_k_max"]),
            "final_effective_row_k_median": float(match_result["effective_row_k_median"]),
            "final_candidate_arc_count": int(match_result["effective_edge_count"]),
            "feasibility_iterations": int(match_result["feasibility_iterations"]),
            "selective_candidate_augmentation_used": bool(match_result["selective_augmentation_used"]),
            "soft_k_max_exceeded": bool(match_result["soft_k_max_exceeded"]),
            "transport_cost_integer_scale": int(match_result["transport_cost_scale"]),
            "transport_cost_shift": float(match_result["transport_cost_shift"]),
            "objective_initial": None if match_result["objective_initial"] is None else float(match_result["objective_initial"]),
            "objective_final": None if match_result["objective_final"] is None else float(match_result["objective_final"]),
            "objective_integer_initial": int(match_result["objective_integer_initial"]),
            "objective_integer_final": int(match_result["objective_integer_final"]),
            "requested_lam_dir_multiplier": None if args.match_lam_dir is None else float(args.match_lam_dir),
            "lam_dir_multiplier": float(match_result["lam_dir_multiplier"]),
            "lam_dir_base": float(match_result["lam_dir_base"]),
            "lam_dir_used": float(match_result["lam_dir_used"]),
            "refine_iter": int(match_result["refine_iter_used"]),
            "graph_regularization_used": bool(match_result["graph_regularization_used"]),
            "feature_cost_median": None if match_result["feature_cost_median"] is None else float(match_result["feature_cost_median"]),
            "dirichlet_penalty_median": None if match_result["dirichlet_penalty_median"] is None else float(match_result["dirichlet_penalty_median"]),
            "dirichlet_energy_before": None if match_result["dirichlet_energy_before"] is None else float(match_result["dirichlet_energy_before"]),
            "dirichlet_energy_after": None if match_result["dirichlet_energy_after"] is None else float(match_result["dirichlet_energy_after"]),
            "optimized_matching_direction": "aggregated graph nodes (A) -> raw slice observations (B) with deterministic capacity policy selected by --slice-capacity-mode",
            "matched_aggregated_node_count": int(np.sum(matched_mask)),
            "unmatched_aggregated_node_count": int(matched_mask.size - np.sum(matched_mask)),
            "graph_candidate_k": int(match_result["graph_candidate_k"]),
            "refinement_augmented_arc_count": None if match_result["refinement_augmented_arc_count"] is None else int(match_result["refinement_augmented_arc_count"]),
            "n_reassigned_by_graph_refinement": int(match_result["n_reassigned_by_graph_refinement"]),
            "fraction_reassigned_by_graph_refinement": float(match_result["fraction_reassigned_by_graph_refinement"]),
            "median_move_distance": float(match_result["median_move_distance"]),
            "median_move_distance_normalized": float(match_result["median_move_distance_normalized"]),
            "max_move_distance": float(match_result["max_move_distance"]),
            "graph_coord_normalization": match_result["graph_coord_normalization"],
            "graph_weight_normalization": match_result["graph_weight_normalization"],
            "refinement_iteration_stats": match_result["refinement_iteration_stats"],
        },
        "shared_gene_count": int(len(shared_genes)),
        "shared_genes": shared_genes,
        "pole_pairs": pole_pairs,
        "slice_output_npz": source_npz,
        "aggregated_output_npz": agg_npz,
        "agg_knn_meta": agg_knn_meta,
        "notes": [
            "Aggregated h5ad counts were used directly without min_reads thresholding or re-binarization.",
            "All aggregated nodes were assigned directly to raw slice nodes with an exact sparse capacity-aware min-cost-flow solve using OR-Tools SimpleMinCostFlow.",
            "Raw slice capacities are determined by --slice-capacity-mode; mass_exact preserves deterministic mass-proportional exact counts, while upper-bound modes need not saturate every slice node.",
            "Slice nodes with zero target capacity are pruned from the internal matching core only; outputs remain indexed in the full raw slice order.",
            "The base solve is sparse, feature-only, and exact on a capacity-aware adaptively augmented candidate graph.",
            "Graph refinement uses the aggregated adjacency after normalizing edge-weight scale, uses centered/scale-normalized raw slice coordinates for the Dirichlet term, interprets --match-lam-dir as a multiplier on an adaptively balanced base lambda, and augments the refinement candidate graph with graph-driven coordinate-neighborhood arcs so increasing graph weight can visibly change the assignment.",
            "slice_assigned_aggregated_feature_maps.npz stores per-slice mean of the raw aggregated ratio fields before and after graph refinement so it is directly comparable to slice_smoothed_ratio_fields.npz['ratio'].",
            "aggregated_to_slice_match_csr.npz has one stored entry per aggregated node row; slice_to_aggregated_match_csr.npz row sums are bounded by the raw slice capacity targets.",
            "aggregated_nodes_slice_mapped_coords.npz stores finite slice coordinates for every aggregated node in raw aggregated order.",
            "Transport arc costs are quantized to integer unit costs internally because OR-Tools min-cost flow requires integer arc costs; the scale factor is recorded in run_metadata.json.",
            "matching_context_base.npz stores the feasible sparse arc graph, capacities, raw coordinate lookup, and the base integer arc costs so exact sparse ensemble solves can be replayed without rebuilding feature/candidate intermediates.",
            "In lexicographic ensemble mode, random secondary costs are composed with the base integer costs so accepted members remain exact optima of the original sparse integer transport objective; perturb mode solves exact perturbed sparse objectives and filters by the original objective band.",
        ],
    }
    write_json(os.path.join(args.output, "run_metadata.json"), metadata)
    _log_progress(args.output, "done", "Pipeline completed successfully.")


def get_aligned_coords(
    ZF_FLAG: str,
    agg_h5ad_path: str,
    slice_h5ad_path: str,
    output_dir: str | None = None,
    *,
    force_recompute: bool = False,
    no_time_filter: bool = False,
    num_pole_pairs: int = NUM_POLE_PAIRS,
    genes_per_pole: int = GENES_PER_POLE,
    abundance_threshold: int = ABUNDANCE_THRESHOLD,
    min_feature_count: int = MIN_FEATURE_COUNT,
    slice_smooth_k: int = SOURCE_RATIO_SMOOTH_K,
    agg_smooth_k: int = AGG_RATIO_SMOOTH_K,
    rank_neutral: float = 0.5,
    match_k0: int = 16,
    match_k_max: int = 256,
    match_lam_dir: float | None = None,
    match_refine_iter: int = 2,
    tree_workers: int | None = None,
    pole_pairs_json: str | None = None,
    slice_capacity_mode: str = "mass_exact",
) -> np.ndarray:
    """
    Run (or reuse) the alignment pipeline and return final aligned slice
    coordinates for each aggregated node, in raw aggregated-node order.

    Returns
    -------
    agg_xy : ndarray, shape (n_agg, 2)
        Final aligned slice coordinates for each aggregated node.
    """
    allowed_flags = {"12hpf", "18hpf", "24hpf"}
    ZF_FLAG = str(ZF_FLAG).strip().lower()
    if ZF_FLAG not in allowed_flags:
        raise ValueError(f"ZF_FLAG must be one of {sorted(allowed_flags)}, got {ZF_FLAG!r}.")

    agg_h5ad_path = _norm_path(agg_h5ad_path)
    slice_h5ad_path = _norm_path(slice_h5ad_path)

    if not os.path.isfile(agg_h5ad_path):
        raise FileNotFoundError(f"Aggregated h5ad not found: {agg_h5ad_path}")
    if not os.path.isfile(slice_h5ad_path):
        raise FileNotFoundError(f"Slice h5ad not found: {slice_h5ad_path}")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(agg_h5ad_path), f"match_result_{ZF_FLAG}")
    output_dir = _norm_path(output_dir)
    ensure_dir(output_dir)

    mapped_npz_path = os.path.join(output_dir, "aggregated_nodes_slice_mapped_coords.npz")
    source_npz_path = os.path.join(output_dir, "slice_smoothed_ratio_fields.npz")
    maps_npz_path = os.path.join(output_dir, "slice_assigned_aggregated_feature_maps.npz")
    matching_context_npz_path = os.path.join(output_dir, MATCHING_CONTEXT_FILENAME)
    metadata_json_path = os.path.join(output_dir, "run_metadata.json")

    request_signature = _build_request_signature(
        slice_h5ad_path=slice_h5ad_path,
        agg_h5ad_path=agg_h5ad_path,
        time_value=ZF_FLAG,
        no_time_filter=no_time_filter,
        num_pole_pairs=num_pole_pairs,
        genes_per_pole=genes_per_pole,
        abundance_threshold=abundance_threshold,
        min_feature_count=min_feature_count,
        slice_smooth_k=slice_smooth_k,
        agg_smooth_k=agg_smooth_k,
        rank_neutral=rank_neutral,
        match_k0=match_k0,
        match_k_max=match_k_max,
        match_lam_dir=match_lam_dir,
        match_refine_iter=match_refine_iter,
        pole_pairs_json=pole_pairs_json,
        slice_capacity_mode=slice_capacity_mode,
    )

    def _existing_run_matches_request() -> bool:
        if not all(os.path.isfile(p) for p in (
            mapped_npz_path,
            source_npz_path,
            maps_npz_path,
            matching_context_npz_path,
            metadata_json_path,
        )):
            return False
        try:
            with open(metadata_json_path, "r") as f:
                meta = json.load(f)
        except Exception:
            return False

        if int(meta.get("schema_version", -1)) != OUTPUT_SCHEMA_VERSION:
            return False
        if str(meta.get("mode", "")) != PIPELINE_MODE:
            return False
        if meta.get("request_signature") != request_signature:
            return False

        return (
            _npz_has_keys(
                mapped_npz_path,
                {
                    "coords_base",
                    "coords_final",
                    "a_to_slice_base",
                    "a_to_slice_final",
                    "moved_mask",
                    "move_distance",
                    "move_distance_normalized",
                },
            )
            and _npz_has_keys(
                maps_npz_path,
                {"coords", "feature_mean_base", "feature_mean_final", "feature_delta", "count_final", "slice_capacity"},
            )
            and _npz_has_keys(
                source_npz_path,
                {"coords", "ratio", "support"},
            )
            and _npz_has_keys(
                matching_context_npz_path,
                {
                    "rows",
                    "cols",
                    "indptr",
                    "base_costs",
                    "base_unit_costs",
                    "slice_capacities_active",
                    "active_raw_slice_indices",
                    "YB_coords_raw",
                },
            )
        )

    if force_recompute or not _existing_run_matches_request():
        args = argparse.Namespace(
            slice=slice_h5ad_path,
            agg=agg_h5ad_path,
            output=output_dir,
            time_value=ZF_FLAG,
            no_time_filter=bool(no_time_filter),
            num_pole_pairs=int(num_pole_pairs),
            genes_per_pole=int(genes_per_pole),
            abundance_threshold=int(abundance_threshold),
            min_feature_count=int(min_feature_count),
            slice_smooth_k=int(slice_smooth_k),
            agg_smooth_k=int(agg_smooth_k),
            rank_neutral=float(rank_neutral),
            match_k0=int(match_k0),
            match_k_max=int(match_k_max),
            match_lam_dir=None if match_lam_dir is None else float(match_lam_dir),
            match_refine_iter=int(match_refine_iter),
            tree_workers=None if tree_workers is None else int(tree_workers),
            pole_pairs_json=pole_pairs_json,
            slice_capacity_mode=str(slice_capacity_mode),
        )
        run_pipeline(args)

    if not _existing_run_matches_request():
        raise RuntimeError(
            "Alignment outputs are missing or incompatible after running the pipeline in "
            f"{output_dir!r}."
        )

    with np.load(mapped_npz_path, allow_pickle=False) as mapped:
        if "coords_final" not in mapped.files:
            raise KeyError(
                f"{mapped_npz_path} does not contain required key 'coords_final'. "
                f"Available keys: {list(mapped.files)}"
            )
        agg_xy = np.asarray(mapped["coords_final"], dtype=np.float64)

    if agg_xy.ndim != 2 or agg_xy.shape[1] != 2:
        raise ValueError(f"Expected aligned coordinates with shape (n_agg, 2), got {agg_xy.shape}")

    if not np.all(np.isfinite(agg_xy)):
        bad = int(np.size(agg_xy) - np.isfinite(agg_xy).sum())
        raise ValueError(f"Aligned coordinates contain {bad} non-finite values.")

    return agg_xy



def get_aligned_coords_ensemble(
    ZF_FLAG: str,
    agg_h5ad_path: str,
    slice_h5ad_path: str,
    output_dir: str | None = None,
    *,
    force_recompute: bool = False,
    force_ensemble_recompute: bool = False,
    ensemble_size: int = 16,
    ensemble_seed: int = 0,
    ensemble_mode: str = "lexicographic",
    ensemble_tie_max: int = 1023,
    ensemble_perturb_units: int = 0,
    ensemble_rel_tol: float = 0.0,
    ensemble_abs_tol: float = 0.0,
    ensemble_n_jobs: int | None = None,
    ensemble_threads_per_worker: int = 1,
    ensemble_mp_start_method: str | None = None,
    return_payload: bool = False,
    **base_kwargs,
) -> np.ndarray | dict[str, object]:
    """
    Run (or reuse) the base alignment pipeline and return an ensemble of sparse
    transport alignments.

    Returns
    -------
    coords : ndarray, shape (k, n_agg, 2)
        One aligned raw slice coordinate array per accepted ensemble member.

    Notes
    -----
    The ensemble is generated from the cached base sparse transport context. In
    lexicographic mode, accepted members are exact optima of the original sparse
    integer transport objective. In perturb mode, members are exact optima of
    their own perturbed sparse objectives and are filtered by the original base
    objective band.
    """
    # Ensure that the ordinary single-solution outputs and the sparse matching
    # context exist. get_aligned_coords() intentionally keeps its original return
    # type and behavior.
    _ = get_aligned_coords(
        ZF_FLAG=ZF_FLAG,
        agg_h5ad_path=agg_h5ad_path,
        slice_h5ad_path=slice_h5ad_path,
        output_dir=output_dir,
        force_recompute=force_recompute,
        **base_kwargs,
    )

    allowed_flags = {"12hpf", "18hpf", "24hpf"}
    ZF_FLAG = str(ZF_FLAG).strip().lower()
    if ZF_FLAG not in allowed_flags:
        raise ValueError(f"ZF_FLAG must be one of {sorted(allowed_flags)}, got {ZF_FLAG!r}.")

    agg_h5ad_path = _norm_path(agg_h5ad_path)
    slice_h5ad_path = _norm_path(slice_h5ad_path)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(agg_h5ad_path), f"match_result_{ZF_FLAG}")
    output_dir = _norm_path(output_dir)

    metadata_json_path = os.path.join(output_dir, "run_metadata.json")
    context_path = os.path.join(output_dir, MATCHING_CONTEXT_FILENAME)
    ensemble_npz_path = os.path.join(output_dir, ENSEMBLE_COORDS_FILENAME)
    ensemble_meta_path = os.path.join(output_dir, ENSEMBLE_METADATA_FILENAME)

    if not os.path.isfile(metadata_json_path):
        raise FileNotFoundError(f"Missing run metadata after base alignment: {metadata_json_path}")
    if not os.path.isfile(context_path):
        _ = get_aligned_coords(
            ZF_FLAG=ZF_FLAG,
            agg_h5ad_path=agg_h5ad_path,
            slice_h5ad_path=slice_h5ad_path,
            output_dir=output_dir,
            force_recompute=True,
            **base_kwargs,
        )

    with open(metadata_json_path, "r") as f:
        base_meta = json.load(f)

    if int(base_meta.get("schema_version", -1)) != OUTPUT_SCHEMA_VERSION:
        raise RuntimeError("Base alignment metadata has an incompatible schema version.")
    if str(base_meta.get("mode", "")) != PIPELINE_MODE:
        raise RuntimeError("Base alignment metadata has an incompatible pipeline mode.")

    refinement_context_path = None
    base_matching_meta = base_meta.get("matching", {}) if isinstance(base_meta.get("matching", {}), dict) else {}
    candidate_ref_ctx = base_matching_meta.get("matching_refinement_context_npz")
    if candidate_ref_ctx:
        candidate_ref_ctx = str(candidate_ref_ctx)
        if not os.path.isabs(candidate_ref_ctx):
            candidate_ref_ctx = os.path.join(output_dir, candidate_ref_ctx)
        if os.path.isfile(candidate_ref_ctx):
            refinement_context_path = candidate_ref_ctx

    ensemble_signature = _build_ensemble_signature(
        base_request_signature=base_meta["request_signature"],
        ensemble_size=int(ensemble_size),
        ensemble_seed=int(ensemble_seed),
        ensemble_mode=str(ensemble_mode),
        ensemble_tie_max=int(ensemble_tie_max),
        ensemble_perturb_units=int(ensemble_perturb_units),
        ensemble_rel_tol=float(ensemble_rel_tol),
        ensemble_abs_tol=float(ensemble_abs_tol),
    )
    ensemble_signature["refinement_context_npz"] = None if refinement_context_path is None else os.path.basename(refinement_context_path)

    def _ensemble_cache_ok() -> bool:
        if force_recompute or force_ensemble_recompute:
            return False
        if not os.path.isfile(ensemble_npz_path) or not os.path.isfile(ensemble_meta_path):
            return False
        try:
            with open(ensemble_meta_path, "r") as f:
                meta = json.load(f)
            if int(meta.get("schema_version", -1)) != OUTPUT_SCHEMA_VERSION:
                return False
            if meta.get("ensemble_signature") != ensemble_signature:
                return False
            return _npz_has_keys(
                ensemble_npz_path,
                {"coords_final", "coords_transport", "a_to_slice", "a_to_slice_transport", "objective_base_float", "objective_base_integer", "coords_frame"},
            )
        except Exception:
            return False

    if not _ensemble_cache_ok():
        result_meta = run_transport_ensemble_from_context(
            context_path=context_path,
            output_dir=output_dir,
            ensemble_size=int(ensemble_size),
            ensemble_seed=int(ensemble_seed),
            ensemble_mode=str(ensemble_mode),
            ensemble_tie_max=int(ensemble_tie_max),
            ensemble_perturb_units=int(ensemble_perturb_units),
            ensemble_rel_tol=float(ensemble_rel_tol),
            ensemble_abs_tol=float(ensemble_abs_tol),
            ensemble_n_jobs=ensemble_n_jobs,
            ensemble_threads_per_worker=int(ensemble_threads_per_worker),
            ensemble_mp_start_method=ensemble_mp_start_method,
            refinement_context_path=refinement_context_path,
        )
        write_json(
            ensemble_meta_path,
            {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "ensemble_signature": ensemble_signature,
                "ensemble_result": result_meta,
            },
        )

    with np.load(ensemble_npz_path, allow_pickle=False) as z:
        coords = np.asarray(z["coords_final"], dtype=np.float64)
        coords_transport = np.asarray(z["coords_transport"], dtype=np.float64) if "coords_transport" in z.files else None
        a_to_slice = np.asarray(z["a_to_slice"], dtype=np.int64) if "a_to_slice" in z.files else None
        a_to_slice_transport = np.asarray(z["a_to_slice_transport"], dtype=np.int64) if "a_to_slice_transport" in z.files else None
        objective_base_float = np.asarray(z["objective_base_float"], dtype=np.float64) if "objective_base_float" in z.files else None
        objective_base_integer = np.asarray(z["objective_base_integer"], dtype=np.int64) if "objective_base_integer" in z.files else None
        member_indices = np.asarray(z["member_indices"], dtype=np.int64) if "member_indices" in z.files else None

    if coords.ndim != 3 or coords.shape[2] != 2:
        raise ValueError(f"Expected ensemble coords with shape (k, n_agg, 2), got {coords.shape}.")
    if not np.all(np.isfinite(coords)):
        bad = int(np.size(coords) - np.isfinite(coords).sum())
        raise ValueError(f"Ensemble coordinates contain {bad} non-finite values.")

    if return_payload:
        ensemble_meta = None
        if os.path.isfile(ensemble_meta_path):
            try:
                with open(ensemble_meta_path, "r") as f:
                    ensemble_meta = json.load(f)
            except Exception:
                ensemble_meta = None
        return {
            "coords": coords,
            "coords_transport": coords_transport,
            "a_to_slice": a_to_slice,
            "a_to_slice_transport": a_to_slice_transport,
            "objective_base_float": objective_base_float,
            "objective_base_integer": objective_base_integer,
            "member_indices": member_indices,
            "output_dir": output_dir,
            "ensemble_npz_path": ensemble_npz_path,
            "ensemble_meta_path": ensemble_meta_path,
            "metadata": ensemble_meta,
            "base_metadata": base_meta,
        }

    return coords

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute slice and aggregated ratio vectors, then solve an exact sparse capacitated transport problem "
            "from aggregated graph nodes to raw slice nodes using OR-Tools' SimpleMinCostFlow, with optional "
            "graph-regularized Dirichlet reweighting on the aggregated adjacency."
        )
    )
    parser.add_argument("--slice", required=True, help="2D slice h5ad with spatial_x/spatial_y in obs")
    parser.add_argument(
        "--agg",
        required=True,
        help="Aggregated h5ad with X counts and GSE_1/GSE_2 (optionally GSE_3) or obsm['X_gse']",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--time-value", default=DEFAULT_TIME_VALUE)
    parser.add_argument("--no-time-filter", action="store_true")
    parser.add_argument("--num-pole-pairs", type=int, default=NUM_POLE_PAIRS)
    parser.add_argument("--genes-per-pole", type=int, default=GENES_PER_POLE)
    parser.add_argument(
        "--pole-pairs-json",
        default=None,
        help="Optional generic JSON file with explicit pole-pair definitions. If omitted, pole pairs are discovered from the slice.",
    )
    parser.add_argument("--abundance-threshold", type=int, default=ABUNDANCE_THRESHOLD)
    parser.add_argument("--min-feature-count", type=int, default=MIN_FEATURE_COUNT)
    parser.add_argument("--slice-smooth-k", type=int, default=SOURCE_RATIO_SMOOTH_K)
    parser.add_argument("--agg-smooth-k", type=int, default=AGG_RATIO_SMOOTH_K)
    parser.add_argument("--rank-neutral", type=float, default=0.5)
    parser.add_argument(
        "--slice-capacity-mode",
        choices=["mass_exact", "uniform_exact", "unit_upper", "spatial_fps_exact"],
        default="mass_exact",
        help=(
            "Generic slice capacity policy. mass_exact preserves historical expression-mass exact capacities; "
            "unit_upper makes every slice node an available one-use candidate when n_slice >= n_agg."
        ),
    )
    parser.add_argument("--match-k0", type=int, default=16)
    parser.add_argument("--match-k-max", type=int, default=256)
    parser.add_argument(
        "--match-lam-dir",
        type=float,
        default=None,
        help=(
            "Multiplier on the adaptively balanced graph-refinement strength. "
            "The code first estimates a dimensionless base lambda after normalizing the aggregated edge scale and slice coordinate scale, "
            "then multiplies that base value by --match-lam-dir. Omit to use 1.0; set to 0 to disable graph refinement."
        ),
    )
    parser.add_argument(
        "--match-refine-iter",
        type=int,
        default=2,
        help="Number of graph-refinement outer reweighting iterations. Default 2 so W can change the assignment beyond a single reweighting pass while remaining practical.",
    )
    parser.add_argument(
        "--tree-workers",
        type=int,
        default=0,
        help="cKDTree worker count. 0 means use the current process thread cap instead of workers=-1/all CPUs.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="Number of sparse transport ensemble members to request. Default 1 disables ensemble solving.",
    )
    parser.add_argument("--ensemble-seed", type=int, default=0)
    parser.add_argument(
        "--ensemble-mode",
        choices=["lexicographic", "perturb"],
        default="lexicographic",
        help=(
            "lexicographic preserves exact optimality of the base sparse integer objective and randomizes only tie-breaking; "
            "perturb solves exact perturbed sparse objectives and filters by the original objective band."
        ),
    )
    parser.add_argument("--ensemble-tie-max", type=int, default=1023)
    parser.add_argument("--ensemble-perturb-units", type=int, default=0)
    parser.add_argument("--ensemble-rel-tol", type=float, default=0.0)
    parser.add_argument("--ensemble-abs-tol", type=float, default=0.0)
    parser.add_argument(
        "--ensemble-n-jobs",
        type=int,
        default=1,
        help=(
            "Parallel worker processes for ensemble solves. Default 1 is memory-safe for large "
            "OR-Tools transport contexts; increase only when the job memory has been sized for "
            "one full min-cost-flow graph per worker."
        ),
    )
    parser.add_argument(
        "--ensemble-threads-per-worker",
        type=int,
        default=1,
        help="Native thread budget inside each ensemble worker process. Default 1 avoids process x thread oversubscription.",
    )
    parser.add_argument(
        "--ensemble-mp-start-method",
        choices=multiprocessing.get_all_start_methods(),
        default=None,
        help="Multiprocessing start method for ensemble workers. Default uses REGZF_ENSEMBLE_MP_START_METHOD, then forkserver/spawn when available.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build synthetic Voronoi-cell spatial graphs and benchmark downstream analysis.

The script can either consume a numeric positions CSV

    id,label,x[,y[,z...]]

or generate synthetic 3D positions inside Voronoi cells. It then constructs a
weighted bipartite graph, writes simulation ground truth, and optionally drives
benchmark sweeps for external clustering / embedding pipelines.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import warnings

import importlib
import importlib.util
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple, Callable, Any

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, coo_matrix, save_npz
from scipy import special
from scipy.linalg import orthogonal_procrustes
from sklearn.neighbors import NearestNeighbors

try:
    # Limits BLAS/OpenMP thread pools when we already parallelize with Python threads.
    # Safe no-op if threadpoolctl is not installed.
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

from concurrent.futures import ThreadPoolExecutor, as_completed


def _ckdtree_query(tree: cKDTree,
                  x: np.ndarray,
                  *,
                  k: int = 1,
                  workers: int | None = None,
                  **kwargs):
    """Compat wrapper for cKDTree.query with optional multithreading.

    SciPy versions >=1.6 support a `workers=` kwarg. Older versions raise TypeError.
    """
    if workers is None:
        return tree.query(x, k=k, **kwargs)
    try:
        return tree.query(x, k=k, workers=int(workers), **kwargs)
    except TypeError:  # pragma: no cover
        return tree.query(x, k=k, **kwargs)


def _ckdtree_query_ball_point(tree: cKDTree,
                             x: np.ndarray,
                             r: float,
                             *,
                             workers: int | None = None,
                             **kwargs):
    """Compat wrapper for cKDTree.query_ball_point with optional multithreading."""
    if workers is None:
        return tree.query_ball_point(x, r, **kwargs)
    try:
        return tree.query_ball_point(x, r, workers=int(workers), **kwargs)
    except TypeError:  # pragma: no cover
        return tree.query_ball_point(x, r, **kwargs)


def _resolve_n_threads(threads: int | None) -> int:
    """Resolve a user-requested thread count against available CPUs."""
    cpu = int(os.cpu_count() or 1)
    if threads is None:
        return 1
    try:
        t = int(threads)
    except Exception:
        return 1
    if t <= 0:
        t = cpu
    return max(1, min(t, cpu))


def _ensure_rng(seed: int | None):
    return np.random.default_rng(seed)


def _ensure_trailing_sep(p: str) -> str:
    """Ensure a filesystem path ends with the OS path separator (optimOps expects this)."""
    p = str(p)
    if not p.endswith(os.sep):
        return p + os.sep
    return p


def _random_other_endpoint_same_partition(
    originals: np.ndarray,
    *,
    lo: int,
    hi: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a random *other* node from the same partition for each endpoint.

    The sampled node is guaranteed to lie in ``[lo, hi)`` and to differ from the
    corresponding entry in ``originals``. This lets us inject false edges while
    preserving the graph's bipartite structure.
    """
    orig = np.asarray(originals, dtype=np.int64)
    if orig.ndim != 1:
        raise ValueError("originals must be a 1D array")
    if orig.size == 0:
        return orig.copy()

    width = int(hi) - int(lo)
    if width < 2:
        raise ValueError("Cannot sample a random *other* node from a partition with fewer than 2 nodes.")

    local_orig = orig - int(lo)
    if np.any(local_orig < 0) or np.any(local_orig >= width):
        raise ValueError("original endpoints were outside the requested partition bounds")

    draw = rng.integers(0, width - 1, size=orig.shape[0])
    draw = draw + (draw >= local_orig)
    return (int(lo) + draw).astype(np.int64, copy=False)


def inject_false_edges_bipartite(
    graph: csr_matrix,
    *,
    n0: int,
    false_edge_frac: float = 0.0,
    seed: int | None = None,
) -> csr_matrix:
    """Rewire a fraction of the graph's *weighted* edges while preserving bipartiteness.

    `graph` stores edge multiplicities in its CSR data array.  We therefore interpret
    ``false_edge_frac`` against ``graph.data.sum()`` (the total number of individual
    edges), not against ``graph.nnz``.  Each rewired edge occurrence keeps one endpoint
    fixed and replaces the other endpoint with a random *other* node from the same
    bipartite side.
    """
    frac = float(false_edge_frac)
    if frac <= 0.0:
        return graph
    if frac > 1.0:
        raise ValueError("false_edge_frac must lie in [0, 1]")

    if not isinstance(graph, csr_matrix):
        graph = graph.tocsr()
    if graph.shape[0] != graph.shape[1]:
        raise ValueError(f"graph must be square (got {graph.shape})")

    n = int(graph.shape[0])
    n0 = int(n0)
    n1 = int(n - n0)
    if n0 <= 0 or n1 <= 0:
        raise ValueError("inject_false_edges_bipartite requires both bipartite partitions to be non-empty")

    coo = graph.tocoo(copy=True)
    if coo.nnz == 0:
        return graph

    counts = np.asarray(coo.data)
    if counts.ndim != 1:
        counts = counts.reshape(-1)
    if not np.all(np.isfinite(counts)):
        raise ValueError("graph contains non-finite edge weights")
    counts = np.rint(counts).astype(np.int64, copy=False)
    if np.any(counts < 0):
        raise ValueError("graph contains negative edge counts")

    total_edges = int(np.sum(counts))
    if total_edges <= 0:
        return graph

    n_false = int(round(frac * float(total_edges)))
    if n_false <= 0:
        return graph
    n_false = min(n_false, total_edges)

    rng = _ensure_rng(seed)

    # Sample individual edge occurrences without replacement from the weighted edge multiset.
    occurrence_pick = np.sort(rng.choice(total_edges, size=n_false, replace=False))
    csum = np.cumsum(counts, dtype=np.int64)
    picked_edge_idx = np.searchsorted(csum, occurrence_pick, side="right")
    rewired_per_entry = np.bincount(picked_edge_idx, minlength=counts.size).astype(np.int64, copy=False)

    remaining = counts - rewired_per_entry
    keep_mask = remaining > 0

    keep_rows = coo.row[keep_mask].astype(np.int64, copy=False)
    keep_cols = coo.col[keep_mask].astype(np.int64, copy=False)
    keep_data = remaining[keep_mask].astype(np.int64, copy=False)

    false_rows = coo.row[picked_edge_idx].astype(np.int64, copy=False)
    false_cols = coo.col[picked_edge_idx].astype(np.int64, copy=False)

    can_rewire_row = n0 >= 2
    can_rewire_col = n1 >= 2
    if not (can_rewire_row or can_rewire_col):
        return graph
    if can_rewire_row and can_rewire_col:
        rewire_row = rng.random(n_false) < 0.5
    elif can_rewire_row:
        rewire_row = np.ones(n_false, dtype=bool)
    else:
        rewire_row = np.zeros(n_false, dtype=bool)

    if np.any(rewire_row):
        false_rows[rewire_row] = _random_other_endpoint_same_partition(
            false_rows[rewire_row], lo=0, hi=n0, rng=rng
        )
    if np.any(~rewire_row):
        false_cols[~rewire_row] = _random_other_endpoint_same_partition(
            false_cols[~rewire_row], lo=n0, hi=n, rng=rng
        )

    row_all = np.concatenate([keep_rows, false_rows.astype(np.int64, copy=False)])
    col_all = np.concatenate([keep_cols, false_cols.astype(np.int64, copy=False)])
    data_all = np.concatenate([
        keep_data,
        np.ones(n_false, dtype=np.int64),
    ])

    out = coo_matrix((data_all, (row_all, col_all)), shape=graph.shape).tocsr()
    out.sum_duplicates()
    out.eliminate_zeros()
    return out


def _sample_node_fusion_map(
    n_side: int,
    frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a subset of nodes to random untargeted same-partition donors."""
    n_side = int(n_side)
    mapping = np.arange(n_side, dtype=np.int64)
    empty = np.empty((0,), dtype=np.int64)
    frac = float(frac)
    if n_side <= 1 or frac <= 0.0:
        return mapping, empty, empty

    n_target = int(round(frac * float(n_side)))
    n_target = min(max(n_target, 0), n_side - 1)
    if n_target <= 0:
        return mapping, empty, empty

    targets = np.sort(rng.choice(n_side, size=n_target, replace=False).astype(np.int64, copy=False))
    donor_pool = np.setdiff1d(np.arange(n_side, dtype=np.int64), targets, assume_unique=True)
    donors = rng.choice(donor_pool, size=n_target, replace=True).astype(np.int64, copy=False)
    mapping[targets] = donors
    return mapping, targets, donors


def fuse_false_nodes_bipartite(
    graph: csr_matrix,
    *,
    n0: int,
    false_edge_frac: float = 0.0,
    seed: int | None = None,
) -> csr_matrix:
    """Fuse a fraction of nodes to random same-partition donors.

    The graph stores only the upper-right bipartite block (label-0 rows to label-1 columns).
    We therefore implement node fusion by duplicating rows and/or columns of that block:
      - a fused label-0 node copies another row node's adjacency row;
      - a fused label-1 node copies another column node's adjacency column.
    """
    frac = float(false_edge_frac)
    if frac <= 0.0:
        return graph
    if frac > 1.0:
        raise ValueError("false_edge_frac must lie in [0, 1]")

    if not isinstance(graph, csr_matrix):
        graph = graph.tocsr()
    if graph.shape[0] != graph.shape[1]:
        raise ValueError(f"graph must be square (got {graph.shape})")

    n = int(graph.shape[0])
    n0 = int(n0)
    n1 = int(n - n0)
    if n0 <= 0 or n1 <= 0:
        raise ValueError("fuse_false_nodes_bipartite requires both bipartite partitions to be non-empty")

    block = graph[:n0, n0:].tocsr(copy=True)
    if block.nnz == 0:
        return graph

    rng = _ensure_rng(seed)
    row_map, _, _ = _sample_node_fusion_map(n0, frac, rng)
    col_map, _, _ = _sample_node_fusion_map(n1, frac, rng)

    if np.array_equal(row_map, np.arange(n0, dtype=np.int64)) and np.array_equal(col_map, np.arange(n1, dtype=np.int64)):
        return graph

    fused_block = block[row_map, :][:, col_map].tocoo(copy=False)
    out = coo_matrix(
        (fused_block.data, (fused_block.row, fused_block.col + n0)),
        shape=graph.shape,
    ).tocsr()
    out.sum_duplicates()
    out.eliminate_zeros()
    return out


# Sentinel label used to mark nodes that were *not processed* by a method.
#
# Important: this MUST be distinct from -1, which is a legitimate "noise" label
# returned by common clustering algorithms (e.g. HDBSCAN). optimOps inputs are
# written on the simulated graph's largest connected component, so final labels
# are mapped back to the full simulator graph and non-LCC nodes are marked with
# this unprocessed sentinel before NMI is computed.
_UNPROCESSED_LABEL = np.int32(-2)

_BENCHMARK_METRIC_KEY = "NMI"
_BENCHMARK_METRIC_LABEL = "Normalized Mutual Information (NMI)"
_BENCHMARK_METHOD_STAT_KEY = "method_nmi"
_BENCHMARK_UMAP_HDBSCAN_STAT_KEY = "umap_hdbscan_md0.99_nmi"
_BENCHMARK_DECOHERENCE_RADIUS_STAT_KEY = "gse_decoherence_radius"
_BENCHMARK_DECOHERENCE_RADIUS_KEY = "decoherence_radius"
_BENCHMARK_DECOHERENCE_RADIUS_LABEL = "Decoherence radius"
_REGZF_ALIGNMENT_STAT_KEY = "register_zf_alignment"
_REGZF_ALIGNMENT_METRIC_LABEL = "register_zf alignment metric"
_BENCHMARK_LAYOUT_VERSION = 11
# Keep these defaults aligned with the attached optimOps.py.
# The current backend writes a 2-column cluster_labels.npy:
#   col 0 = HDBSCAN on the final embedding
#   col 1 = Infomap on the transformed matrix derived from the final embedding
_OPTIMOPS_FINAL_CLUSTER_LAYOUT_VERSION = 2
_OPTIMOPS_FINAL_CLUSTER_METHODS = (
    "hdbscan",
    "infomap",
)
_UMAP_HDBSCAN_CACHE_VERSION = 3
_UMAP_HDBSCAN_METHOD_KEY = "umap_hdbscan_md0.99"
_UMAP_HDBSCAN_METHOD_LABEL = "Graph UMAP→HDBSCAN (clusterplot settings, min_dist=0.99)"
_UMAP_DECOHERENCE_METHOD_KEY = "umap_embedding_decoherence"
_UMAP_DECOHERENCE_METHOD_LABEL = "UMAP: decoherence radius"
_UMAP_BENCHMARK_PCA_DIM = 15
_UMAP_BENCHMARK_N_NEIGHBORS = 30
_UMAP_BENCHMARK_MIN_DIST = 0.99
_UMAP_BENCHMARK_METRIC = "cosine"
_UMAP_HDBSCAN_MIN_CLUSTER_SIZE = 10
_UMAP_HDBSCAN_MIN_SAMPLES = 10
# These are hard-coded inside the attached optimOps.py and are *not* user-tunable
# through run_GSE() in the current backend.
_OPTIMOPS_FINAL_MIN_CLUSTER_SIZE = 50
_OPTIMOPS_FINAL_HDBSCAN_MIN_SAMPLES = 50
_CYLINDRICAL_HOLE_RADIUS_FRAC = 0.20


def _optimops_default_cluster_method_names(n_cols: int) -> tuple[str, ...]:
    """Best-effort method names for the current attached optimOps cluster_labels layout."""
    n_cols = int(max(0, n_cols))
    if n_cols <= 0:
        return tuple()
    if n_cols == 1:
        return ("hdbscan",)
    if n_cols == 2:
        return ("hdbscan", "infomap")
    names = ["hdbscan", "infomap"]
    names.extend([f"cluster_{i}" for i in range(2, n_cols)])
    return tuple(names[:n_cols])


def _optimops_cluster_method_names_from_meta(meta: Any | None, n_cols: int) -> tuple[str, ...]:
    """Read or infer cluster method names from current optimOps cluster metadata.

    The attached optimOps backend writes layout_version=2 metadata with one
    HDBSCAN column followed by one or more Infomap variants.  Older metadata may
    already contain explicit method names; prefer those when present.
    """
    n_cols = int(max(0, n_cols))
    if n_cols <= 0:
        return tuple()
    if isinstance(meta, dict):
        for key in ("cluster_labels_methods", "cluster_methods"):
            vals = meta.get(key, None)
            if isinstance(vals, list) and len(vals) >= n_cols:
                return tuple(str(v) for v in vals[:n_cols])

        variants = meta.get("infomap_variants", None)
        if isinstance(variants, list) and n_cols >= 2:
            names = ["hdbscan"]
            for i, variant in enumerate(variants):
                vname = None
                if isinstance(variant, dict):
                    vname = str(variant.get("name", "")).strip()
                if not vname or vname.lower() == "default":
                    names.append("infomap" if i == 0 else f"infomap_{i}")
                else:
                    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in vname)
                    names.append("infomap" if safe.lower() == "infomap" else f"infomap_{safe}")
            if len(names) >= n_cols:
                return tuple(names[:n_cols])

    return _optimops_default_cluster_method_names(n_cols)


def _warn_about_unsupported_optimops_flags(args: argparse.Namespace, *, backend_mod: Any | None = None) -> None:
    """Reserved hook for backend-specific warning messages."""
    return None


from contextlib import contextmanager


@contextmanager
def _temporary_env_overrides(overrides: dict[str, str | int | None]):
    """Temporarily set environment variables around one backend call."""
    old: dict[str, str | None] = {}
    try:
        for key, val in (overrides or {}).items():
            if val is None:
                continue
            sval = str(val).strip()
            if not sval:
                continue
            old[key] = os.environ.get(key)
            os.environ[key] = sval
        yield
    finally:
        for key, val in old.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def _prime_optimops_thread_env_from_args(args: argparse.Namespace) -> None:
    """Set thread-budget environment before importing optimOps.py.

    optimOps.py imports threads_bootstrap.NTHREADS at module import time; passing
    --threads to the simulator alone does not reach that import-time constant.
    """
    try:
        t = _resolve_n_threads(getattr(args, "threads", None))
    except Exception:
        t = 1
    if t <= 0:
        return
    for key in (
        "SLURM_CPUS_PER_TASK",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = str(int(t))
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")


def write_optimops_inputs(
    outdir: Path,
    graph: csr_matrix,
    *,
    partition_labels: np.ndarray,
    orig_ids: np.ndarray | None = None,
    link_name: str = "link_assoc_reindexed.npz",
    index_key_name: str = "index_key.npy",
    keep_largest_component: bool = True,
    keep_name: str = "keep_nodes_global.npy",
) -> tuple[Path, Path]:
    """Write optimOps-compatible inputs alongside the simulator outputs.

    optimOps.py expects two files in the dataset directory:
      - link_assoc_reindexed.npz : CSR sparse matrix (usually the one-way bipartite block)
      - index_key.npy           : integer array of shape (N, 3) with columns:
            [partition_label (0/1), original_id, reindexed_id]

    Parameters
    ----------
    outdir:
        Directory to write files into.
    graph:
        CSR adjacency matrix of shape (N,N).
    partition_labels:
        (N,) array in {0,1} matching the node ordering in `graph`.
    orig_ids:
        Optional (N,) array of "original" ids (per optimOps index_key semantics).
        If omitted or invalid, a per-partition 0..n-1 scheme is used.

    Returns
    -------
    (link_path, index_key_path)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not isinstance(graph, csr_matrix):
        graph = graph.tocsr()
    n = int(graph.shape[0])
    if graph.shape[0] != graph.shape[1]:
        raise ValueError(f"graph must be square (got {graph.shape})")

    lab = np.asarray(partition_labels)
    if lab.ndim != 1 or lab.shape[0] != n:
        raise ValueError(f"partition_labels must be shape (N,) matching graph (N={n}); got {lab.shape}")
    if not set(np.unique(lab)).issubset({0, 1}):
        raise ValueError("partition_labels must contain only 0/1")

    # ------------------------------------------------------------------
    # IMPORTANT: For optimOps we always write ONLY the largest connected component
    # of the *simulated* graph (undirected connectivity). This matches the
    # typical optimOps preprocessing step (reindex_input_files) and guarantees
    # embedding/clustering results have a consistent connected support.
    #
    # We also write keep_nodes_global.npy to map optimOps node indices back to
    # the simulator-global node indices (the ordering used by node_positions_scaled.npy
    # and ground_truth_cells.csv).
    # ------------------------------------------------------------------
    keep = np.arange(n, dtype=np.int64)
    if bool(keep_largest_component) and n > 0:
        try:
            from scipy.sparse.csgraph import connected_components
            # Connectivity is computed on the undirected version.
            A = (graph + graph.T).tocsr()
            A.sum_duplicates()
            A.eliminate_zeros()
            n_comp, comp = connected_components(A, directed=False, return_labels=True)
            if int(n_comp) > 1:
                sizes = np.bincount(comp.astype(np.int64, copy=False), minlength=int(n_comp))
                best = int(np.argmax(sizes))
                keep = np.flatnonzero(comp == best).astype(np.int64, copy=False)
                keep.sort()
            else:
                keep = np.arange(n, dtype=np.int64)
        except Exception:
            keep = np.arange(n, dtype=np.int64)

        # Persist mapping (even if keep is identity; simplifies downstream alignment).
        try:
            np.save(outdir / str(keep_name), keep.astype(np.int64, copy=False))
        except Exception:
            pass

    # Subset graph + labels/orig_ids to the kept nodes.
    if keep is not None and keep.size != n:
        graph = graph[keep, :][:, keep].tocsr()
        lab = lab[keep]
        if orig_ids is not None:
            try:
                orig_ids = np.asarray(orig_ids)[keep]
            except Exception:
                orig_ids = None
        n = int(graph.shape[0])

    # Save adjacency under optimOps' expected name.
    # optimOps' loaders assume float weights in some paths; store float64 for safety.
    link_path = outdir / str(link_name)
    graph_to_save = graph
    try:
        if graph_to_save.dtype != np.float64:
            graph_to_save = graph_to_save.astype(np.float64)
    except Exception:
        graph_to_save = graph
    save_npz(link_path, graph_to_save)

    # Build index_key: [partition_label, original_id, reindexed_id]
    idx = np.arange(n, dtype=np.int64)
    if orig_ids is not None:
        oid = np.asarray(orig_ids)
        if oid.ndim != 1 or oid.shape[0] != n:
            oid = None
        else:
            try:
                oid = oid.astype(np.int64, copy=False)
            except Exception:
                oid = None
    else:
        oid = None

    if oid is None:
        # Default: contiguous per-partition ids (mirrors optimOps.reindex_input_files output).
        oid = np.empty(n, dtype=np.int64)
        m0 = (lab == 0)
        m1 = ~m0
        oid[m0] = np.arange(int(np.sum(m0)), dtype=np.int64)
        oid[m1] = np.arange(int(np.sum(m1)), dtype=np.int64)

    index_key = np.column_stack([lab.astype(np.int64, copy=False), oid, idx]).astype(np.int32, copy=False)
    ik_path = outdir / str(index_key_name)
    np.save(ik_path, index_key)


    return link_path, ik_path


# ───────────────────── Synthetic register_zf fixtures + alignment plots ─────────────────────

_REGZF_FIXTURE_PARENT_H5AD = "final.h5ad"
_REGZF_FIXTURE_SLICE_H5AD = "register_zf_slice.h5ad"


def _optimops_register_zf_flag(args: argparse.Namespace) -> str:
    """Return the requested register_zf time flag."""
    return str(getattr(args, "optimops_register_zf", "") or "").strip().lower()


def _sync_optimops_register_zf_aliases(args: argparse.Namespace) -> None:
    """Normalize register_zf argparse values to the current optimOps surface."""
    setattr(args, "optimops_register_zf", _optimops_register_zf_flag(args))


def _optimops_register_zf_requested(args: argparse.Namespace) -> bool:
    return bool(_optimops_register_zf_flag(args))


def _optimops_zf_slice_path_for_run(run_dir: Path | str, args: argparse.Namespace) -> Path:
    supplied = str(getattr(args, "optimops_slice_path", "") or "").strip()
    if supplied:
        return Path(supplied).expanduser().resolve()
    return Path(run_dir).resolve() / _REGZF_FIXTURE_SLICE_H5AD


def _coerce_h5ad_dataframe_for_legacy_anndata(df: pd.DataFrame) -> pd.DataFrame:
    """Make obs/var frames safe for old and new AnnData h5ad writers.

    AnnData's HDF5 writer refuses a DataFrame whose index name is also a
    column name unless the index and column values are exactly identical.  The
    synthetic register_zf parent used index.name == "node_id" while also
    keeping an integer obs["node_id"] column, which newer anndata versions
    reject.  Rename the index in that conflict case; the column remains the
    useful numeric node id and the index remains a plain string observation id.
    """
    out = df.copy()
    index_name = None if out.index.name is None else str(out.index.name)
    if index_name is not None and index_name in out.columns:
        # Avoid AnnData's ambiguous index/column serialization path entirely.
        # Keeping both named "node_id" is not necessary for downstream code; the
        # actual numeric node id remains available as obs["node_id"].
        base = f"{index_name}_index"
        candidate = base
        ii = 1
        while candidate in out.columns:
            candidate = f"{base}_{ii}"
            ii += 1
        index_name = candidate

    out.index = pd.Index(
        np.asarray([str(v) for v in out.index.to_list()], dtype=object),
        name=index_name,
    )
    for col in list(out.columns):
        ser = out[col]
        dtype_name = str(getattr(ser.dtype, "name", ser.dtype)).lower()
        if dtype_name == "string" or dtype_name.startswith("string["):
            out[col] = ser.astype(object).where(~ser.isna(), None)
        elif ser.dtype == object:
            out[col] = ser.where(~pd.isna(ser), None)
    return out


def _write_h5ad_compat(adata, h5ad_path: Path | str) -> None:
    """Atomically write an AnnData object while avoiding nullable-string failures."""
    try:
        import anndata as _anndata
        if hasattr(_anndata, "settings") and hasattr(_anndata.settings, "allow_write_nullable_strings"):
            _anndata.settings.allow_write_nullable_strings = True
    except Exception:
        pass

    try:
        adata.obs = _coerce_h5ad_dataframe_for_legacy_anndata(adata.obs)
    except Exception:
        pass
    try:
        adata.var = _coerce_h5ad_dataframe_for_legacy_anndata(adata.var)
    except Exception:
        pass

    h5ad_path = Path(h5ad_path)
    h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = h5ad_path.with_name(h5ad_path.name + ".tmp")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            adata.write_h5ad(str(tmp_path), compression="gzip", compression_opts=4)
        except TypeError:
            adata.write_h5ad(str(tmp_path))
        os.replace(str(tmp_path), str(h5ad_path))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _h5ad_looks_readable(path: Path | str) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        import anndata as ad
        a = ad.read_h5ad(str(path), backed="r")
        try:
            return int(a.n_obs) > 0 and int(a.n_vars) > 0
        finally:
            try:
                a.file.close()
            except Exception:
                pass
    except Exception:
        # If anndata is unavailable in a lightweight environment, fall back to a
        # conservative nonempty-file check.  The real optimOps/register_zf run will
        # import anndata and fail loudly if the file is corrupt.
        return True


def _unit_interval_xy(coords_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(coords_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"coords_xy must be shape (n, >=2); got {xy.shape}")
    xy = xy[:, :2]
    lo = np.nanmin(xy, axis=0)
    hi = np.nanmax(xy, axis=0)
    span = np.maximum(hi - lo, 1e-9)
    return (xy - lo[None, :]) / span[None, :], lo, span


def _synthetic_zf_marker_specs(*, num_pairs: int = 3, genes_per_pole: int = 3) -> tuple[list[str], list[dict[str, Any]]]:
    num_pairs = int(max(1, num_pairs))
    genes_per_pole = int(max(1, genes_per_pole))
    base_dirs = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [2.0, 1.0], [1.0, 2.0]],
        dtype=float,
    )
    genes: list[str] = []
    specs: list[dict[str, Any]] = []
    for pair in range(num_pairs):
        direction = base_dirs[pair % base_dirs.shape[0]].astype(float)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        for side, sign in (("a", 1.0), ("b", -1.0)):
            center = np.asarray([0.5, 0.5], dtype=float) + sign * 0.38 * direction
            center = np.clip(center, 0.06, 0.94)
            for g in range(genes_per_pole):
                name = f"regzf_pair{pair:02d}_{side}_{g:02d}"
                genes.append(name)
                specs.append({"gene": name, "pair": int(pair), "side": side.upper(), "center": center})
    return genes, specs



def _synthetic_zf_expression_csr(
    coords_xy: np.ndarray,
    marker_specs: list[dict[str, Any]],
    *,
    seed: int | None,
    marker_strength: float = 24.0,
    sigma_frac: float = 0.16,
    background_rate: float = 0.0,
) -> csr_matrix:
    xy01, _lo, _span = _unit_interval_xy(coords_xy)
    n = int(xy01.shape[0])
    g = int(len(marker_specs))
    rng = np.random.default_rng(seed)
    X = np.zeros((n, g), dtype=np.int16)
    sigma2 = float(max(sigma_frac, 1e-6)) ** 2
    for j, spec in enumerate(marker_specs):
        c = np.asarray(spec["center"], dtype=float).reshape(1, 2)
        d2 = np.sum((xy01 - c) ** 2, axis=1)
        field = np.exp(-0.5 * d2 / sigma2)
        lam = float(background_rate) + float(marker_strength) * field
        vals = rng.poisson(lam).astype(np.int16, copy=False)
        if n > 0:
            core = field >= np.quantile(field, 0.90)
            vals[core] = np.maximum(vals[core], 3).astype(np.int16, copy=False)
        X[:, j] = vals
    return csr_matrix(X)


def write_synthetic_register_zf_fixtures(
    outdir: Path,
    *,
    sim_pos: np.ndarray,
    partition_labels: np.ndarray,
    cell_ids: np.ndarray | None = None,
    zf_flag: str = "18hpf",
    slice_h5ad_path: Path | str | None = None,
    write_slice_h5ad: bool = True,
    slice_n: int = 0,
    num_pole_pairs: int = 3,
    genes_per_pole: int = 3,
    seed: int | None = None,
) -> tuple[Path, Path | None]:
    """Write synthetic final.h5ad and raw slice h5ad for optimOps/register_zf tests."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        import anndata as ad
    except Exception as exc:
        raise ImportError("Synthetic register_zf fixtures require anndata.") from exc

    sim_pos = np.asarray(sim_pos, dtype=float)
    labels = np.asarray(partition_labels, dtype=np.int32).ravel()
    if sim_pos.ndim != 2 or sim_pos.shape[0] != labels.shape[0] or sim_pos.shape[1] < 2:
        raise ValueError("sim_pos must be (N, >=2) and match partition_labels.")

    keep_path = outdir / "keep_nodes_global.npy"
    if keep_path.exists():
        keep = np.asarray(np.load(str(keep_path))).astype(np.int64, copy=False).ravel()
    else:
        keep = np.arange(sim_pos.shape[0], dtype=np.int64)
    if keep.size == 0:
        raise ValueError("Cannot write synthetic register_zf h5ad for an empty optimOps support.")

    marker_genes, marker_specs = _synthetic_zf_marker_specs(
        num_pairs=int(num_pole_pairs), genes_per_pole=int(genes_per_pole)
    )
    var = pd.DataFrame(index=pd.Index(marker_genes, name="gene"))
    var["feature_type"] = "protein_coding"
    agg_xy = sim_pos[keep, :2]
    X_agg = _synthetic_zf_expression_csr(
        agg_xy,
        marker_specs,
        seed=(None if seed is None else int(seed) + 1701),
        marker_strength=24.0,
        sigma_frac=0.17,
        background_rate=0.0,
    )
    obs = pd.DataFrame(index=pd.Index([str(i) for i in range(keep.size)], name="obs_id"))
    obs["node_id"] = np.arange(keep.size, dtype=np.int64)
    obs["sim_global_node_id"] = keep.astype(np.int64, copy=False)
    obs["partition_label"] = labels[keep].astype(np.int32, copy=False)
    obs["spatial_x"] = agg_xy[:, 0]
    obs["spatial_y"] = agg_xy[:, 1]
    if cell_ids is not None:
        cell_ids_arr = np.asarray(cell_ids).ravel()
        if cell_ids_arr.shape[0] == labels.shape[0]:
            obs["cell_id"] = cell_ids_arr[keep].astype(np.int32, copy=False)
    parent = ad.AnnData(X=X_agg, obs=obs, var=var.copy())
    agg_h5ad_path = outdir / _REGZF_FIXTURE_PARENT_H5AD
    _write_h5ad_compat(parent, agg_h5ad_path)

    written_slice_path: Path | None = None
    if bool(write_slice_h5ad):
        slice_h5ad_path = Path(slice_h5ad_path) if slice_h5ad_path is not None else (outdir / _REGZF_FIXTURE_SLICE_H5AD)
        slice_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(None if seed is None else int(seed) + 1702)
        n_slice = int(slice_n)
        if n_slice <= 0 or n_slice == sim_pos.shape[0]:
            slice_xy = np.asarray(sim_pos[:, :2], dtype=float)
            slice_global = np.arange(sim_pos.shape[0], dtype=np.int64)
        else:
            replace = n_slice > sim_pos.shape[0]
            slice_global = rng.choice(sim_pos.shape[0], size=n_slice, replace=replace).astype(np.int64, copy=False)
            slice_xy = np.asarray(sim_pos[slice_global, :2], dtype=float)
            span = np.maximum(np.ptp(sim_pos[:, :2], axis=0), 1e-9)
            slice_xy = slice_xy + rng.normal(scale=0.01 * span[None, :], size=slice_xy.shape)
        X_slice = _synthetic_zf_expression_csr(
            slice_xy,
            marker_specs,
            seed=(None if seed is None else int(seed) + 1703),
            marker_strength=20.0,
            sigma_frac=0.16,
            background_rate=0.005,
        )
        obs_s = pd.DataFrame(index=pd.Index([str(i) for i in range(slice_xy.shape[0])], name="slice_node_id"))
        obs_s["spatial_x"] = slice_xy[:, 0]
        obs_s["spatial_y"] = slice_xy[:, 1]
        obs_s["time"] = str(zf_flag).strip().lower()
        obs_s["sim_global_node_id"] = slice_global.astype(np.int64, copy=False)
        # The downstream registration code reads this as a generic sampling/capacity
        # weight column; the simulator chooses a uniform slice density for this fixture.
        obs_s["slice_capacity_weight"] = 1.0
        src = ad.AnnData(X=X_slice, obs=obs_s, var=var.copy())
        _write_h5ad_compat(src, slice_h5ad_path)
        written_slice_path = slice_h5ad_path


    meta = {
        "agg_h5ad_path": str(agg_h5ad_path),
        "slice_h5ad_path": None if written_slice_path is None else str(written_slice_path),
        "zf_flag": str(zf_flag).strip().lower(),
        "n_parent_nodes": int(keep.size),
        "n_total_sim_nodes": int(sim_pos.shape[0]),
        "n_genes": int(len(marker_genes)),
        "num_pole_pairs": int(num_pole_pairs),
        "genes_per_pole": int(genes_per_pole),
    }
    with open(outdir / "synthetic_register_zf_fixture_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return agg_h5ad_path, written_slice_path


# ───────────────────── Positions generation (3D Voronoi interior) ─────────────────────

def positions_to_csv_array(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Convert (points, labels) into the numeric CSV array format:
        columns = id,label,x,y,z
    """
    P = np.asarray(points, dtype=float)
    y = np.asarray(labels, dtype=int)
    if P.ndim != 2:
        raise ValueError("points must be shape (n,3)")
    if y.ndim != 1 or y.shape[0] != P.shape[0]:
        raise ValueError("labels must be shape (n,) and match points")
    n, d = P.shape
    if d != 3:
        raise ValueError("Only 3D positions are supported (d must be 3).")
    ids = np.arange(n, dtype=np.int64).astype(float)
    out = np.zeros((n, 2 + d), dtype=float)
    out[:, 0] = ids
    out[:, 1] = y.astype(float)
    out[:, 2:] = P
    return out


def _points_outside_cylindrical_hole(points: np.ndarray,
                                    *,
                                    bbox_lo: np.ndarray,
                                    bbox_hi: np.ndarray,
                                    radius_frac: float = _CYLINDRICAL_HOLE_RADIUS_FRAC) -> np.ndarray:
    """Return a mask for points that lie outside a central cylindrical exclusion zone.

    The cylinder is centered in the x-y plane of the bounding box and extends through
    the full z-span of the box. This only affects node placement; Voronoi centers and
    the diffusion field remain unchanged.
    """
    X = np.asarray(points, dtype=float)
    if X.ndim != 2 or X.shape[1] < 3:
        raise ValueError("points must be shape (n,3) for cylindrical hole masking")

    lo = np.asarray(bbox_lo, dtype=float).reshape(3)
    hi = np.asarray(bbox_hi, dtype=float).reshape(3)
    center_xy = 0.5 * (lo[:2] + hi[:2])
    span_xy = np.maximum(hi[:2] - lo[:2], 1e-12)
    radius = float(radius_frac) * float(np.min(span_xy))
    rr = np.sum((X[:, :2] - center_xy[None, :]) ** 2, axis=1)
    return rr > (radius * radius)


def generate_3d_voronoi_positions(
    n0: int,
    n1: int,
    *,
    n_cells: int = 300,
    cell_jitter: float = 0.25,
    bbox_lo: tuple[float, float, float] | None = None,
    bbox_hi: tuple[float, float, float] | None = None,
    boundary_margin_frac: float = 0.20,
    hole: bool = False,
    hole_radius_frac: float = _CYLINDRICAL_HOLE_RADIUS_FRAC,
    threads: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 3D Voronoi "cell packing" and sample node positions uniformly
    inside the cells away from the Voronoi boundaries. When ``hole`` is true,
    node placement excludes a centered cylinder running through the full z-span
    of the bounding box, while Voronoi centers and the diffusion field are left
    unchanged.

    Returns:
      centers : (n_cells, 3)
      points  : (n0+n1, 3)
      labels  : (n0+n1,) in {0,1}
    """
    n0 = int(n0)
    n1 = int(n1)
    n_cells = int(n_cells)
    if n0 <= 0 or n1 <= 0:
        raise ValueError("n0 and n1 must both be positive.")
    if n_cells <= 1:
        raise ValueError("n_cells must be >= 2.")
    boundary_margin_frac = float(boundary_margin_frac)
    if boundary_margin_frac < 0.0:
        raise ValueError("boundary_margin_frac must be >= 0.")
    hole = bool(hole)
    hole_radius_frac = float(hole_radius_frac)
    if hole_radius_frac <= 0.0:
        raise ValueError("hole_radius_frac must be > 0 when hole generation is enabled.")

    rng = _ensure_rng(seed)

    # Use multithreaded KDTree queries during point generation when available.
    n_threads = _resolve_n_threads(threads)

    lo = np.array(bbox_lo if bbox_lo is not None else (0.0, 0.0, 0.0), dtype=float)
    hi = np.array(bbox_hi if bbox_hi is not None else (10.0, 10.0, 10.0), dtype=float)
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError("bbox_lo and bbox_hi must be length-3.")
    span = np.maximum(hi - lo, 1e-9)

    # --- Jittered lattice centers
    vol = float(np.prod(span))
    s = float((vol / float(n_cells)) ** (1.0 / 3.0))  # target spacing
    counts = np.maximum(1, np.ceil(span / max(s, 1e-12)).astype(int))
    step = span / counts

    axes = [lo[k] + (np.arange(counts[k]) + 0.5) * step[k] for k in range(3)]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    if mesh.shape[0] >= n_cells:
        choose = rng.choice(mesh.shape[0], size=n_cells, replace=False)
        centers = mesh[choose].copy()
    else:
        extra = rng.uniform(lo, hi, size=(n_cells - mesh.shape[0], 3))
        centers = np.concatenate([mesh, extra], axis=0)

    if cell_jitter > 0:
        centers = centers + rng.normal(scale=float(cell_jitter), size=centers.shape) * step
        centers = np.clip(centers, lo, hi)

    tree = cKDTree(centers)

    # Cell-specific boundary margin = fraction of half nearest-neighbor center distance
    dnn, _ = _ckdtree_query(tree, centers, k=2, workers=n_threads)
    if dnn.ndim != 2 or dnn.shape[1] < 2:
        raise RuntimeError("Failed to compute center nearest-neighbor distances.")
    margin = boundary_margin_frac * 0.5 * dnn[:, 1]  # (n_cells,)

    # --- Rejection sample points away from boundaries
    n_total = n0 + n1
    accepted = []
    got = 0
    max_iter = 2000
    it = 0

    while got < n_total and it < max_iter:
        it += 1
        remaining = n_total - got
        batch = int(max(4096, min(200000, remaining * 40)))
        cands = rng.uniform(lo, hi, size=(batch, 3))

        dists, idxs = _ckdtree_query(tree, cands, k=2, workers=n_threads)
        d1 = dists[:, 0]
        d2 = dists[:, 1]
        i1 = idxs[:, 0]
        i2 = idxs[:, 1]

        sep = np.linalg.norm(centers[i2] - centers[i1], axis=1)
        sep = np.maximum(sep, 1e-12)
        delta = ((d2 * d2) - (d1 * d1)) / (2.0 * sep)
        delta = np.maximum(delta, 0.0)

        keep = delta > margin[i1]
        if hole and np.any(keep):
            keep &= _points_outside_cylindrical_hole(
                cands,
                bbox_lo=lo,
                bbox_hi=hi,
                radius_frac=hole_radius_frac,
            )
        if np.any(keep):
            pts = cands[keep]
            accepted.append(pts)
            got += int(pts.shape[0])

        if it % 50 == 0 and got == 0 and boundary_margin_frac > 0.0:
            hole_note = " or disabling --hole" if hole else ""
            raise RuntimeError(
                "No points accepted so far; boundary_margin_frac may be too large for the chosen n_cells/bbox. "
                f"Try a smaller --boundary-margin-frac, fewer cells{hole_note}."
            )

    if got < n_total:
        hole_note = " or disabling --hole" if hole else ""
        raise RuntimeError(
            f"Failed to sample {n_total} interior points within {max_iter} iterations "
            f"(accepted={got}). Consider reducing --boundary-margin-frac, increasing the bbox{hole_note}."
        )

    points = np.concatenate(accepted, axis=0)[:n_total].copy()

    # Identical distributions across point types: shuffle then assign labels by index.
    rng.shuffle(points)
    labels = np.concatenate([np.zeros(n0, dtype=np.int32), np.ones(n1, dtype=np.int32)], axis=0)

    return centers, points, labels


def generate_jittered_lattice_cell_centers(points: np.ndarray,
                                          n_cells: int,
                                          jitter_frac: float = 0.25,
                                          seed: int | None = None) -> np.ndarray:
    """
    Generate ~quasi-periodic "cell centers" in 2D/3D via a jittered lattice over the point cloud bbox.
    """
    X = np.asarray(points, dtype=float)
    if X.ndim != 2:
        raise ValueError("points must be a 2D array (n,d)")
    n, d = X.shape
    if d not in (2, 3):
        raise ValueError(f"Only d=2 or d=3 supported (got d={d})")
    n_cells = int(n_cells)
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")

    rng = _ensure_rng(seed)

    lo = np.min(X, axis=0)
    hi = np.max(X, axis=0)
    span = np.maximum(hi - lo, 1e-9)

    # Target spacing from equal-volume cells.
    vol = float(np.prod(span))
    s = float((vol / n_cells) ** (1.0 / d))

    counts = np.maximum(1, np.ceil(span / max(s, 1e-9)).astype(int))
    step = span / counts

    axes = [lo[k] + (np.arange(counts[k]) + 0.5) * step[k] for k in range(d)]
    mesh = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, d)

    if mesh.shape[0] >= n_cells:
        choose = rng.choice(mesh.shape[0], size=n_cells, replace=False)
        centers = mesh[choose].copy()
    else:
        extra = rng.uniform(lo, hi, size=(n_cells - mesh.shape[0], d))
        centers = np.concatenate([mesh, extra], axis=0)

    if jitter_frac > 0:
        jitter = rng.normal(scale=jitter_frac, size=centers.shape) * step
        centers = centers + jitter
        centers = np.clip(centers, lo, hi)

    return centers


# ───────────────────── Cellular diffusion field + graph (cellular mode) ─────────────────────

class CellularDiffusionField:
    """
    Strictly-positive diffusion coefficient field D(x) in 2D/3D built from a Voronoi tessellation.

    This is kept intact from the original script because it drives the 'cellular' graph builder.
    """
    def __init__(self,
                 centers: np.ndarray,
                 D_in: float = 1.0,
                 D_out: float | None = None,
                 D_min: float = 0.05,
                 D_max: float = 10.0,
                 ecs_width: float = 0.0,
                 qp_modes: int = 6,
                 qp_wavelength: float | None = None,
                 qp_amp: float = 0.7,
                 qp_mode: str = "cell",
                 cell_sigma: float = 0.5,
                 cell_q_corr: float = 0.6,
                 membrane_width: float = 0.25,
                 membrane_strength: float = 2.0,
                 seed: int | None = None):
        self.centers = np.asarray(centers, dtype=float)
        if self.centers.ndim != 2:
            raise ValueError("centers must be (n_cells, d)")
        self.n_cells, self.d = self.centers.shape
        if self.d not in (2, 3):
            raise ValueError("Only d=2 or d=3 supported")
        self.tree = cKDTree(self.centers)

        self.D_in = float(D_in)
        self.D0 = self.D_in  # backwards compat alias
        self.D_out = self.D_in if (D_out is None) else float(D_out)
        if (not np.isfinite(self.D_out)) or (self.D_out <= 0):
            self.D_out = self.D_in

        self.D_min = float(D_min)
        self.D_max = float(D_max)
        self.ecs_width = float(max(0.0, ecs_width))
        if self.D_in <= 0 or self.D_out <= 0 or self.D_min <= 0 or self.D_max <= 0:
            raise ValueError("D_in, D_out, D_min, D_max must be > 0")
        if self.D_min > self.D_max:
            raise ValueError("Require D_min <= D_max")

        self.qp_modes = int(max(0, qp_modes))
        self.qp_amp = float(qp_amp)
        qp_mode = str(qp_mode).lower()
        if qp_mode not in ("global", "cell"):
            raise ValueError("qp_mode must be 'global' or 'cell'")
        self.qp_mode = qp_mode
        self.membrane_width = float(max(0.0, membrane_width))
        self.membrane_strength = float(max(0.0, membrane_strength))

        rng = _ensure_rng(seed)

        # Quasi-periodic mode wavevectors
        if self.qp_modes > 0:
            if qp_wavelength is None or qp_wavelength <= 0:
                dnn, _ = self.tree.query(self.centers, k=2)
                qp_wavelength = float(np.median(dnn[:, 1]))
                if not np.isfinite(qp_wavelength) or qp_wavelength <= 0:
                    qp_wavelength = 1.0
            qp_wavelength = float(qp_wavelength)

            dirs = rng.normal(size=(self.qp_modes, self.d))
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            dirs = dirs / norms

            wl_jit = rng.uniform(0.8, 1.2, size=(self.qp_modes, 1))
            wls = qp_wavelength * wl_jit
            self.kvecs = dirs / wls
            self.phases = rng.uniform(0.0, 2.0 * np.pi, size=self.qp_modes)
            self.mode_amps = (self.qp_amp / np.sqrt(self.qp_modes)) * np.ones(self.qp_modes)
        else:
            self.kvecs = np.zeros((0, self.d), dtype=float)
            self.phases = np.zeros((0,), dtype=float)
            self.mode_amps = np.zeros((0,), dtype=float)

        self.qp_centers = self._qp_raw(self.centers)

        cell_sigma = float(max(0.0, cell_sigma))
        cell_q_corr = float(np.clip(cell_q_corr, 0.0, 1.0))
        if cell_sigma > 0:
            qp_c = self.qp_centers
            mu = float(np.mean(qp_c))
            sd = float(np.std(qp_c))
            if not np.isfinite(sd) or sd <= 1e-12:
                qp_z = np.zeros_like(qp_c)
            else:
                qp_z = (qp_c - mu) / sd
            eps = rng.normal(size=self.n_cells)
            self.cell_eta = cell_sigma * (cell_q_corr * qp_z + np.sqrt(max(0.0, 1.0 - cell_q_corr**2)) * eps)
        else:
            self.cell_eta = np.zeros(self.n_cells, dtype=float)

        self._logD0 = float(np.log(self.D0))

    def assign_cells(self, x: np.ndarray, *, workers: int | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        _, idx = _ckdtree_query(self.tree, x, k=1, workers=workers)
        return np.asarray(idx, dtype=np.int32)

    def _qp_raw(self, x: np.ndarray) -> np.ndarray:
        if self.qp_modes <= 0:
            return np.zeros((np.asarray(x).shape[0],), dtype=float) if np.asarray(x).ndim == 2 else np.array(0.0)
        X = np.asarray(x, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        dots = X @ self.kvecs.T
        arg = (2.0 * np.pi) * dots + self.phases[None, :]
        return (np.cos(arg) * self.mode_amps[None, :]).sum(axis=1)

    def _boundary_distance(self, x: np.ndarray, *, workers: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(x, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        dists, idxs = _ckdtree_query(self.tree, X, k=2, workers=workers)
        d1 = dists[:, 0]
        d2 = dists[:, 1]
        i1 = idxs[:, 0].astype(np.int32)
        i2 = idxs[:, 1].astype(np.int32)
        c1 = self.centers[i1]
        c2 = self.centers[i2]
        sep = np.linalg.norm(c2 - c1, axis=1)
        sep = np.maximum(sep, 1e-12)
        delta = ((d2 * d2) - (d1 * d1)) / (2.0 * sep)
        delta = np.maximum(delta, 0.0)
        return delta, i1

    def __call__(self, x: np.ndarray, *, workers: int | None = None) -> np.ndarray:
        X = np.asarray(x, dtype=float)
        scalar = False
        if X.ndim == 1:
            X = X[None, :]
            scalar = True
        if X.shape[1] != self.d:
            raise ValueError(f"x has dimension {X.shape[1]} but field is d={self.d}")

        delta, cell = self._boundary_distance(X, workers=workers)

        if self.ecs_width > 0.0:
            s_ecs = np.exp(-(delta / self.ecs_width) ** 2)
        else:
            s_ecs = np.zeros_like(delta)

        if (self.ecs_width > 0.0) and (self.D_out != self.D_in):
            log_base = self._logD0 + s_ecs * np.log(self.D_out / self.D_in)
        else:
            log_base = self._logD0

        if self.qp_mode == "global":
            qp = self._qp_raw(X)
        else:
            qp_cell = self.qp_centers[cell]
            if self.ecs_width > 0.0:
                qp_global = self._qp_raw(X)
                qp = (1.0 - s_ecs) * qp_cell + s_ecs * qp_global
            else:
                qp = qp_cell

        cell_term = self.cell_eta[cell]
        if self.ecs_width > 0.0:
            cell_term = (1.0 - s_ecs) * cell_term

        logD = log_base + qp + cell_term

        if self.membrane_width > 0.0 and self.membrane_strength > 0.0:
            b = np.exp(-(delta / self.membrane_width) ** 2)
            logD = logD - (self.membrane_strength * b)

        D = np.exp(logD)
        D = np.clip(D, self.D_min, self.D_max)
        return D[0] if scalar else D


def _eval_invD_and_cell(field: CellularDiffusionField,
                        x: np.ndarray,
                        *,
                        workers: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate invD(x)=1/D(x) and Voronoi cell id for points x using one KDTree query."""
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X[None, :]

    delta, cell = field._boundary_distance(X, workers=workers)

    if field.ecs_width > 0.0:
        s_ecs = np.exp(-(delta / field.ecs_width) ** 2)
    else:
        s_ecs = np.zeros_like(delta)

    if (field.ecs_width > 0.0) and (field.D_out != field.D_in):
        log_base = field._logD0 + s_ecs * np.log(field.D_out / field.D_in)
    else:
        log_base = field._logD0

    if field.qp_mode == "global":
        qp = field._qp_raw(X)
    else:
        qp_cell = field.qp_centers[cell]
        if field.ecs_width > 0.0:
            qp_global = field._qp_raw(X)
            qp = (1.0 - s_ecs) * qp_cell + s_ecs * qp_global
        else:
            qp = qp_cell

    cell_term = field.cell_eta[cell]
    if field.ecs_width > 0.0:
        cell_term = (1.0 - s_ecs) * cell_term

    logD = log_base + qp + cell_term
    if field.membrane_width > 0.0 and field.membrane_strength > 0.0:
        b = np.exp(-(delta / field.membrane_width) ** 2)
        logD = logD - (field.membrane_strength * b)

    D = np.exp(logD)
    D = np.clip(D, field.D_min, field.D_max)
    invD = 1.0 / np.maximum(D, field.D_min)
    return invD, cell.astype(np.int32)


def _path_harmonic_mean_D_boundaryaware_batch(
        field: CellularDiffusionField,
        x0: np.ndarray,
        x1: np.ndarray,
        dist: np.ndarray,
        n_base: int = 5,
        max_crossings: int = 16,
        min_cosang: float = 1e-3,
        cap_frac: float = 0.45,
        P_mem: float = 0.0) -> np.ndarray:
    """
    Boundary-aware harmonic-mean diffusivity for many segments from a common source x0 to targets x1.

    Returns:
      Dbar : (m,) effective diffusivity per target
    """
    x0 = np.asarray(x0, dtype=float).reshape(1, -1)
    X1 = np.asarray(x1, dtype=float)
    if X1.ndim == 1:
        X1 = X1[None, :]

    m = int(X1.shape[0])
    if m == 0:
        return np.zeros((0,), dtype=float)

    dist = np.asarray(dist, dtype=float).reshape(-1)
    P_mem = float(P_mem)
    if (not np.isfinite(P_mem)) or P_mem <= 0.0:
        P_mem = 0.0
    n_base = int(max(2, n_base))
    ts_base = np.linspace(0.0, 1.0, n_base, dtype=float)

    v = X1 - x0  # (m,d)

    inv_base = np.empty((n_base, m), dtype=float)
    cell_base = np.empty((n_base, m), dtype=np.int32)
    for k, t in enumerate(ts_base):
        pts = x0 + t * v
        # Called inside per-source workers; avoid nested parallelism in KDTree.
        invD, cell = _eval_invD_and_cell(field, pts, workers=1)
        inv_base[k, :] = invD
        cell_base[k, :] = cell

    changes = cell_base[1:, :] != cell_base[:-1, :]
    n_cross = changes.sum(axis=0).astype(float)

    if field.membrane_width <= 0.0 or field.membrane_strength <= 0.0:
        dt = ts_base[1:] - ts_base[:-1]
        inv_int = (0.5 * (inv_base[:-1, :] + inv_base[1:, :]) * dt[:, None]).sum(axis=0)
        if P_mem > 0.0:
            dj = np.maximum(dist, 1e-12)
            inv_int = inv_int + (n_cross / (P_mem * dj))
        inv_int = np.maximum(inv_int, 1e-12)
        return 1.0 / inv_int

    ref_t: list[float] = []
    ref_j: list[int] = []
    w = float(field.membrane_width)

    for j in range(m):
        ks = np.nonzero(changes[:, j])[0]
        if ks.size == 0:
            continue
        if ks.size > max_crossings:
            ks = ks[np.linspace(0, ks.size - 1, max_crossings, dtype=int)]

        dj = float(dist[j])
        if not np.isfinite(dj) or dj <= 1e-12:
            continue

        uj = v[j] / dj

        for kk in ks:
            tL = float(ts_base[kk])
            tR = float(ts_base[kk + 1])
            cA = int(cell_base[kk, j])
            cB = int(cell_base[kk + 1, j])
            if cA == cB:
                continue

            # Bisection to localize boundary
            t_left = tL
            t_right = tR
            id_left = cA
            id_right = cB
            for _ in range(8):
                t_mid = 0.5 * (t_left + t_right)
                x_mid = x0[0] + t_mid * v[j]
                id_mid = int(field.assign_cells(x_mid, workers=1)[0])
                if id_mid == id_left:
                    t_left = t_mid
                else:
                    t_right = t_mid
                    id_right = id_mid
            t_cross = 0.5 * (t_left + t_right)

            c1 = field.centers[id_left]
            c2 = field.centers[id_right]
            wvec = c2 - c1

            sep = float(np.linalg.norm(wvec))
            if sep > 1e-12:
                n_hat = wvec / sep
                cosang = abs(float(np.dot(n_hat, uj)))
            else:
                cosang = 1.0
            cosang = max(float(min_cosang), float(cosang))
            dt_band = (w / cosang) / dj
            dt_cap = float(cap_frac) * abs(tR - tL)
            dt = min(float(dt_band), float(dt_cap))
            if dt <= 0.0:
                continue

            for tt in (t_cross - dt, t_cross, t_cross + dt):
                if 0.0 <= tt <= 1.0:
                    ref_t.append(float(tt))
                    ref_j.append(int(j))

    ref_by_j = [[] for _ in range(m)]
    if ref_t:
        t_arr = np.asarray(ref_t, dtype=float)
        j_arr = np.asarray(ref_j, dtype=np.int32)
        pts_ref = x0 + t_arr[:, None] * v[j_arr]
        inv_ref, _ = _eval_invD_and_cell(field, pts_ref, workers=1)
        for tt, jj, invv in zip(t_arr, j_arr, inv_ref):
            ref_by_j[int(jj)].append((float(tt), float(invv)))

    Dbar = np.empty((m,), dtype=float)
    for j in range(m):
        ts = ts_base
        inv = inv_base[:, j]
        extras = ref_by_j[j]
        if extras:
            ts = np.concatenate([ts, np.asarray([e[0] for e in extras], dtype=float)])
            inv = np.concatenate([inv, np.asarray([e[1] for e in extras], dtype=float)])

        order = np.argsort(ts)
        ts = ts[order]
        inv = inv[order]

        # compress duplicates (conservative: keep max inv)
        if ts.size > 1:
            uts = [float(ts[0])]
            uinv = [float(inv[0])]
            for kk in range(1, ts.size):
                if abs(float(ts[kk]) - uts[-1]) < 1e-12:
                    uinv[-1] = max(uinv[-1], float(inv[kk]))
                else:
                    uts.append(float(ts[kk]))
                    uinv.append(float(inv[kk]))
            ts = np.asarray(uts, dtype=float)
            inv = np.asarray(uinv, dtype=float)

        if ts.size < 2:
            inv_int = float(inv[0]) if inv.size else (1.0 / max(field.D0, field.D_min))
        else:
            dt = ts[1:] - ts[:-1]
            inv_int = float(np.sum(0.5 * (inv[:-1] + inv[1:]) * dt))
            inv_int = max(inv_int, 1e-12)

        if P_mem > 0.0:
            dj = float(dist[j])
            if np.isfinite(dj) and dj > 1e-12:
                inv_int += float(n_cross[j]) / (P_mem * dj)

        inv_int = max(inv_int, 1e-12)
        Dbar[j] = 1.0 / inv_int

    Dbar = np.maximum(Dbar, field.D_min)
    return Dbar


def _path_harmonic_mean_D_sample_batch(
        field: CellularDiffusionField,
        x0: np.ndarray,
        x1: np.ndarray,
        n_samples: int = 5,
        P_mem: float = 0.0) -> np.ndarray:
    """
    Fast harmonic-mean diffusivity for many segments from a common source x0 to targets x1.
    """
    x0 = np.asarray(x0, dtype=float).reshape(1, -1)
    X1 = np.asarray(x1, dtype=float)
    if X1.ndim == 1:
        X1 = X1[None, :]
    m = int(X1.shape[0])
    if m == 0:
        return np.zeros((0,), dtype=float)

    P_mem = float(P_mem)
    if (not np.isfinite(P_mem)) or P_mem <= 0.0:
        P_mem = 0.0

    n_samples = int(max(2, n_samples))
    ts = np.linspace(0.0, 1.0, n_samples, dtype=float)
    v = X1 - x0
    dist = np.linalg.norm(v, axis=1)
    dist = np.maximum(dist, 1e-12)

    inv = np.empty((n_samples, m), dtype=float)
    cell = np.empty((n_samples, m), dtype=np.int32)
    for k, t in enumerate(ts):
        pts = x0 + t * v
        # Called inside per-source workers; avoid nested parallelism in KDTree.
        invD, cid = _eval_invD_and_cell(field, pts, workers=1)
        inv[k, :] = invD
        cell[k, :] = cid

    dt = ts[1:] - ts[:-1]
    inv_int = (0.5 * (inv[:-1, :] + inv[1:, :]) * dt[:, None]).sum(axis=0)

    if P_mem > 0.0:
        changes = cell[1:, :] != cell[:-1, :]
        n_cross = changes.sum(axis=0).astype(float)
        inv_int = inv_int + (n_cross / (P_mem * dist))

    inv_int = np.maximum(inv_int, 1e-12)
    Dbar = 1.0 / inv_int
    return np.maximum(Dbar, field.D_min)


def _yukawa_weight(d: np.ndarray,
                   ell: np.ndarray,
                   dim: int,
                   eps: float = 0.05) -> np.ndarray:
    """
    Screened reaction–diffusion kernel weight:
      - 3D: exp(-r/ell)/r
      - 2D: K0(r/ell)
    with small-distance regularization r -> sqrt(r^2 + eps^2).
    """
    d = np.asarray(d, dtype=float)
    ell = np.asarray(ell, dtype=float)
    r = np.sqrt(d * d + float(eps) ** 2)
    ell = np.maximum(ell, 1e-9)

    if dim == 3:
        return np.exp(-r / ell) / r
    elif dim == 2:
        return special.k0(r / ell)
    else:
        raise ValueError("dim must be 2 or 3")


# ─────────────────────────── Cellular builder ───────────────────────────

def build_graph_from_positions_cellular(
    pos_csv: str,
    outdir: Path,
    *,
    rescale: float,
    mperPt: float,
    pi_short: float,
    sigma_s: float,
    short_trunc: float,
    k_capture: float,
    long_trunc: float,
    long_eps: float,
    path_samples: int,
    n_cells: int,
    cell_jitter: float,
    D_in: float,
    D_out: float,
    ecs_width: float,
    ecs_width_frac: float,
    P_mem: float,
    D_min: float,
    D_max: float,
    qp_modes: int,
    qp_wavelength: float | None,
    qp_amp: float,
    cell_sigma: float,
    cell_q_corr: float,
    membrane_width: float,
    membrane_width_frac: float,
    membrane_strength: float,
    qp_mode: str,
    amp_dispersion: float,
    false_edge_frac: float = 0.0,
    false_edge_targets_nodes: bool = False,
    write_synthetic_zf_fixtures: bool = False,
    synthetic_zf_flag: str = "18hpf",
    synthetic_zf_slice_path: str | None = None,
    synthetic_zf_write_slice: bool = True,
    synthetic_zf_slice_n: int = 0,
    synthetic_zf_num_pole_pairs: int = 3,
    synthetic_zf_genes_per_pole: int = 3,
    path_mode: str = "sample",
    max_nbrs_per_source: int = 0,
    threads: int = 0,
    graph_chunk_size: int = 0,
    seed: int | None = None,
    voronoi_centers: np.ndarray | None = None,
    # Isosurface visualization (AO if possible; fallback to marching cubes)
    render_isosurface: bool = True,
    isosurface_grid_res: int = 110,
    isosurface_show_points: bool = False,
    isosurface_point_size: float = 4.0,
    isosurface_panel_size: tuple[int, int] = (1400, 1100),
) -> tuple[Path, int, int]:
    """
    Build graph.npz from positions CSV using the Voronoi cellular diffusion field model.

    NOTE: This version does NOT write umi0.txt/umi1.txt. Ground-truth files are still written.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    rng = _ensure_rng(seed)

    # Threading controls (also used for large KDTree queries during simulation).
    n_threads = _resolve_n_threads(threads)
    blas_ctx = threadpool_limits(limits=1) if (threadpool_limits is not None and n_threads > 1) else nullcontext()

    arr = np.loadtxt(pos_csv, delimiter=',')
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 3:
        raise ValueError('pos CSV must have at least 3 columns: id,label,x[,y...]')

    order = np.lexsort((arr[:, 0], arr[:, 1]))
    arr = arr[order]

    labels = arr[:, 1].astype(int)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError('label column must be 0 or 1 only')
    mask0 = labels == 0
    mask1 = labels == 1
    n0 = int(mask0.sum())
    n1 = int(mask1.sum())
    if n0 == 0 or n1 == 0:
        raise ValueError('both partitions must be nonempty (labels 0 and 1).')

    # Variance-normalized coordinates
    pos = arr[:, 2:]
    d = int(pos.shape[1])
    if d not in (2, 3):
        raise ValueError(f"Only 2D/3D supported for cellular diffusion (got d={d})")
    varsum = float(np.sum(np.var(pos, axis=0)))
    scale = (float(rescale) / np.sqrt(varsum)) if varsum > 0.0 else 1.0
    sim_pos = pos * scale
    pos0 = sim_pos[mask0]
    pos1 = sim_pos[mask1]

    # --- Generate cells + diffusion field
    if voronoi_centers is None:
        centers = generate_jittered_lattice_cell_centers(
            sim_pos,
            n_cells=int(n_cells),
            jitter_frac=float(cell_jitter),
            seed=(None if seed is None else int(seed) + 12345),
        )
    else:
        centers = np.asarray(voronoi_centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != d:
            raise ValueError(f"voronoi_centers must be shape (n_cells,{d}) in the same coordinate system as pos_csv")
        if centers.shape[0] != int(n_cells):
            print(f"[WARN] voronoi_centers has {centers.shape[0]} centers, overriding n_cells={n_cells}.")
            n_cells = int(centers.shape[0])
        centers = centers * scale

    _tmp_tree = cKDTree(centers)
    _dnn, _ = _ckdtree_query(_tmp_tree, centers, k=2, workers=n_threads)
    center_spacing = float(np.median(_dnn[:, 1])) if _dnn.shape[1] > 1 else float(np.median(_dnn))
    if (not np.isfinite(center_spacing)) or center_spacing <= 0.0:
        center_spacing = 1.0

    D_in = float(D_in)
    if (not np.isfinite(D_in)) or D_in <= 0.0:
        raise ValueError("D_in must be > 0")
    D_out = float(D_out)
    if (not np.isfinite(D_out)) or D_out <= 0.0:
        D_out = D_in

    membrane_width = float(membrane_width)
    if membrane_width <= 0.0:
        membrane_width = float(membrane_width_frac) * center_spacing
    membrane_width = float(max(0.0, membrane_width))

    ecs_width = float(ecs_width)
    if ecs_width <= 0.0:
        ecs_width = float(ecs_width_frac) * center_spacing
    ecs_width = float(max(0.0, ecs_width))

    P_mem = float(P_mem)
    if (not np.isfinite(P_mem)) or P_mem <= 0.0:
        P_mem = 0.0

    field = CellularDiffusionField(
        centers=centers,
        D_in=D_in,
        D_out=D_out,
        ecs_width=ecs_width,
        D_min=float(D_min),
        D_max=float(D_max),
        qp_modes=int(qp_modes),
        qp_wavelength=(None if qp_wavelength is None else float(qp_wavelength)),
        qp_amp=float(qp_amp),
        qp_mode=str(qp_mode),
        cell_sigma=float(cell_sigma),
        cell_q_corr=float(cell_q_corr),
        membrane_width=float(membrane_width),
        membrane_strength=float(membrane_strength),
        seed=(None if seed is None else int(seed) + 23456),
    )

    # Large KDTree queries: use internal multithreading where available.
    cell_ids = field.assign_cells(sim_pos, workers=n_threads)
    D_nodes = field(sim_pos, workers=n_threads)
    delta_nodes, _ = field._boundary_distance(sim_pos, workers=n_threads)
    if field.membrane_width > 0.0:
        b_nodes = np.exp(-(delta_nodes / field.membrane_width) ** 2)
    else:
        b_nodes = np.zeros_like(delta_nodes, dtype=float)
    if field.ecs_width > 0.0:
        ecs_factor_nodes = np.exp(-(delta_nodes / field.ecs_width) ** 2)
    else:
        ecs_factor_nodes = np.zeros_like(delta_nodes, dtype=float)

    # --- Per-node activity weights (log-space)
    amps = rng.normal(loc=0.0, scale=float(amp_dispersion), size=(n0 + n1))
    amp0 = amps[:n0]
    amp1 = amps[n0:]
    A0 = np.exp(amp0)
    A1 = np.exp(amp1)

    # --- Total UEIs and mixture split
    L_total = int(max(0, round(float(mperPt) * float(n0 + n1))))
    pi_short = float(np.clip(pi_short, 0.0, 1.0))
    L_short = int(rng.binomial(L_total, pi_short))
    L_long = int(L_total - L_short)

    sigma_s = float(sigma_s)
    if sigma_s <= 0:
        raise ValueError("--sigma-s must be > 0")
    short_trunc = float(max(0.0, short_trunc))
    r_short = short_trunc * sigma_s

    k_capture = float(k_capture)
    if k_capture <= 0:
        raise ValueError("--k-capture must be > 0")
    ell_max = float(np.sqrt(field.D_max / k_capture))
    long_trunc = float(max(0.0, long_trunc))
    r_long = long_trunc * ell_max

    r_max = max(r_short, r_long)

    tree1 = cKDTree(pos1)

    path_mode = str(path_mode).strip().lower()
    if path_mode not in {"sample", "boundaryaware", "endpoint"}:
        raise ValueError(f"--path-mode must be one of sample|boundaryaware|endpoint (got {path_mode!r})")
    max_nbrs_per_source = int(max_nbrs_per_source)
    if max_nbrs_per_source < 0:
        raise ValueError("--max-nbrs-per-source must be >= 0")

    D0_nodes = D_nodes[:n0]
    D1_nodes = D_nodes[n0:]
    cell0_nodes = cell_ids[:n0]
    cell1_nodes = cell_ids[n0:]

    # Threading controls (graph construction)
    chunk = int(graph_chunk_size) if int(graph_chunk_size) > 0 else max(32, n0 // (max(1, n_threads) * 4))
    chunk = int(min(512, max(1, chunk)))

    # --------------------------
    # Pass 1: compute per-source normalizers Zs, Zl
    # --------------------------
    Zs = np.zeros(n0, dtype=float)
    Zl = np.zeros(n0, dtype=float)

    def _pass1_worker(start_i: int, end_i: int):
        zs = np.zeros(end_i - start_i, dtype=float)
        zl = np.zeros(end_i - start_i, dtype=float)

        Xblk = pos0[start_i:end_i]
        if max_nbrs_per_source > 0:
            k = int(max_nbrs_per_source)
            dmat, imat = tree1.query(Xblk, k=k, distance_upper_bound=r_max)
            if k == 1:
                dmat = dmat[:, None]
                imat = imat[:, None]
        else:
            nbrs_list = tree1.query_ball_point(Xblk, r_max)

        for ii, i in enumerate(range(start_i, end_i)):
            xi = Xblk[ii]

            if max_nbrs_per_source > 0:
                nbrs = imat[ii]
                dist = dmat[ii]
                m_ok = np.isfinite(dist) & (nbrs < n1)
                if not np.any(m_ok):
                    continue
                nbrs = nbrs[m_ok].astype(np.int32, copy=False)
                dist = dist[m_ok].astype(float, copy=False)
            else:
                nbrs = nbrs_list[ii]
                if not nbrs:
                    continue
                nbrs = np.asarray(nbrs, dtype=np.int32)
                dx = pos1[nbrs] - xi[None, :]
                dist = np.linalg.norm(dx, axis=1)

            if L_short > 0 and r_short > 0:
                mS = dist <= r_short
                if np.any(mS):
                    dS = dist[mS]
                    jS = nbrs[mS]
                    wS = (A0[i] * A1[jS]) * np.exp(-(dS * dS) / (4.0 * sigma_s * sigma_s))
                    zs[ii] = float(np.sum(wS))

            if L_long > 0 and r_long > 0:
                mL = dist <= r_long
                if np.any(mL):
                    dL = dist[mL]
                    jL = nbrs[mL]
                    xj = pos1[jL]

                    if path_mode == "boundaryaware":
                        Dbar = _path_harmonic_mean_D_boundaryaware_batch(
                            field, xi, xj, dL, n_base=path_samples, P_mem=P_mem)
                    elif path_mode == "sample":
                        Dbar = _path_harmonic_mean_D_sample_batch(
                            field, xi, xj, n_samples=path_samples, P_mem=P_mem)
                    else:
                        inv = 0.5 * ((1.0 / D0_nodes[i]) + (1.0 / D1_nodes[jL]))
                        if P_mem > 0.0:
                            n_cross = (cell0_nodes[i] != cell1_nodes[jL]).astype(float)
                            inv = inv + (n_cross / (P_mem * np.maximum(dL, 1e-12)))
                        inv = np.maximum(inv, 1e-12)
                        Dbar = 1.0 / inv
                        Dbar = np.maximum(Dbar, field.D_min)

                    ell = np.sqrt(Dbar / k_capture)
                    wK = _yukawa_weight(dL, ell, dim=d, eps=float(long_eps))
                    wL = (A0[i] * A1[jL]) * wK
                    wL = np.where(np.isfinite(wL) & (wL > 0), wL, 0.0)
                    zl[ii] = float(np.sum(wL))

        return start_i, zs, zl

    if n_threads > 1 and n0 > chunk:
        with blas_ctx, ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = []
            for s in range(0, n0, chunk):
                futures.append(ex.submit(_pass1_worker, s, min(s + chunk, n0)))
            for fut in as_completed(futures):
                s, zs_chunk, zl_chunk = fut.result()
                Zs[s:s + zs_chunk.size] = zs_chunk
                Zl[s:s + zl_chunk.size] = zl_chunk
    else:
        s, zs_chunk, zl_chunk = _pass1_worker(0, n0)
        Zs[:] = zs_chunk
        Zl[:] = zl_chunk

    if L_short > 0 and not np.any(Zs > 0):
        Zs[:] = 1.0
    if L_long > 0 and not np.any(Zl > 0):
        Zl[:] = 1.0

    ps = Zs / float(np.sum(Zs)) if L_short > 0 else None
    pl = Zl / float(np.sum(Zl)) if L_long > 0 else None

    ms = rng.multinomial(L_short, ps) if L_short > 0 else np.zeros(n0, dtype=int)
    ml = rng.multinomial(L_long,  pl) if L_long > 0 else np.zeros(n0, dtype=int)

    # --------------------------
    # Pass 2: sample targets and accumulate edge counts
    # --------------------------
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []

    n_chunks = int((n0 + chunk - 1) // chunk)
    chunk_seeds = None
    if seed is not None:
        ss = np.random.SeedSequence(int(seed) + 7777)
        chunk_seeds = ss.spawn(n_chunks)

    def _pass2_worker(chunk_id: int, start_i: int, end_i: int):
        rng_local = (np.random.default_rng(chunk_seeds[chunk_id])
                     if chunk_seeds is not None else np.random.default_rng())
        rows_l: list[int] = []
        cols_l: list[int] = []
        data_l: list[int] = []

        Xblk = pos0[start_i:end_i]
        if max_nbrs_per_source > 0:
            k = int(max_nbrs_per_source)
            dmat, imat = tree1.query(Xblk, k=k, distance_upper_bound=r_max)
            if k == 1:
                dmat = dmat[:, None]
                imat = imat[:, None]
        else:
            nbrs_list = tree1.query_ball_point(Xblk, r_max)

        for ii, i in enumerate(range(start_i, end_i)):
            xi = Xblk[ii]

            if max_nbrs_per_source > 0:
                nbrs = imat[ii]
                dist = dmat[ii]
                m_ok = np.isfinite(dist) & (nbrs < n1)
                if not np.any(m_ok):
                    continue
                nbrs = nbrs[m_ok].astype(np.int32, copy=False)
                dist = dist[m_ok].astype(float, copy=False)
            else:
                nbrs = nbrs_list[ii]
                if not nbrs:
                    continue
                nbrs = np.asarray(nbrs, dtype=np.int32)
                dx = pos1[nbrs] - xi[None, :]
                dist = np.linalg.norm(dx, axis=1)

            mi = int(ms[i])
            if mi > 0 and r_short > 0:
                mS = dist <= r_short
                if np.any(mS):
                    dS = dist[mS]
                    jS = nbrs[mS]
                    wS = (A0[i] * A1[jS]) * np.exp(-(dS * dS) / (4.0 * sigma_s * sigma_s))
                    sZ = float(np.sum(wS))
                    if sZ > 0:
                        p = wS / sZ
                        c = rng_local.multinomial(mi, p).astype(np.int64)
                        nz = np.nonzero(c > 0)[0]
                        if nz.size:
                            rows_l.extend([i] * int(nz.size))
                            cols_l.extend((n0 + jS[nz]).tolist())
                            data_l.extend(c[nz].tolist())

            mi = int(ml[i])
            if mi > 0 and r_long > 0:
                mL = dist <= r_long
                if np.any(mL):
                    dL = dist[mL]
                    jL = nbrs[mL]
                    xj = pos1[jL]

                    if path_mode == "boundaryaware":
                        Dbar = _path_harmonic_mean_D_boundaryaware_batch(
                            field, xi, xj, dL, n_base=path_samples, P_mem=P_mem)
                    elif path_mode == "sample":
                        Dbar = _path_harmonic_mean_D_sample_batch(
                            field, xi, xj, n_samples=path_samples, P_mem=P_mem)
                    else:
                        inv = 0.5 * ((1.0 / D0_nodes[i]) + (1.0 / D1_nodes[jL]))
                        if P_mem > 0.0:
                            n_cross = (cell0_nodes[i] != cell1_nodes[jL]).astype(float)
                            inv = inv + (n_cross / (P_mem * np.maximum(dL, 1e-12)))
                        inv = np.maximum(inv, 1e-12)
                        Dbar = 1.0 / inv
                        Dbar = np.maximum(Dbar, field.D_min)

                    ell = np.sqrt(Dbar / k_capture)
                    wK = _yukawa_weight(dL, ell, dim=d, eps=float(long_eps))
                    wL = (A0[i] * A1[jL]) * wK
                    wL = np.where(np.isfinite(wL) & (wL > 0), wL, 0.0)
                    sZ = float(np.sum(wL))
                    if sZ > 0:
                        p = wL / sZ
                        c = rng_local.multinomial(mi, p).astype(np.int64)
                        nz = np.nonzero(c > 0)[0]
                        if nz.size:
                            rows_l.extend([i] * int(nz.size))
                            cols_l.extend((n0 + jL[nz]).tolist())
                            data_l.extend(c[nz].tolist())

        return rows_l, cols_l, data_l

    if n_threads > 1 and n0 > chunk:
        with blas_ctx, ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = []
            chunk_id = 0
            for s in range(0, n0, chunk):
                futures.append(ex.submit(_pass2_worker, chunk_id, s, min(s + chunk, n0)))
                chunk_id += 1
            for fut in as_completed(futures):
                r_l, c_l, d_l = fut.result()
                if r_l:
                    rows.extend(r_l)
                    cols.extend(c_l)
                    data.extend(d_l)
    else:
        r_l, c_l, d_l = _pass2_worker(0, 0, n0)
        rows.extend(r_l)
        cols.extend(c_l)
        data.extend(d_l)

    if rows:
        G = coo_matrix((np.asarray(data, dtype=np.int64),
                        (np.asarray(rows, dtype=np.int32),
                         np.asarray(cols, dtype=np.int32))),
                       shape=(n0 + n1, n0 + n1)).tocsr()
        G.sum_duplicates()
    else:
        G = csr_matrix((n0 + n1, n0 + n1), dtype=np.int64)

    if float(false_edge_frac) > 0.0:
        if bool(false_edge_targets_nodes):
            G = fuse_false_nodes_bipartite(
                G,
                n0=n0,
                false_edge_frac=float(false_edge_frac),
                seed=seed,
            )
        else:
            G = inject_false_edges_bipartite(
                G,
                n0=n0,
                false_edge_frac=float(false_edge_frac),
                seed=seed,
            )

    outdir.mkdir(parents=True, exist_ok=True)

    # Save scaled node positions (graph order) for later Procrustes alignment / visualization.
    try:
        np.save(outdir / "node_positions_scaled.npy", sim_pos.astype(np.float32, copy=False))
    except Exception:
        pass

    graph_path = outdir / 'graph.npz'
    save_npz(graph_path, G)

    # Write optimOps input files alongside the simulator outputs
    try:
        write_optimops_inputs(
            outdir,
            G,
            partition_labels=labels.astype(np.int32, copy=False),
            orig_ids=arr[:, 0],
        )
    except Exception as e:
        print(f"[WARN] Could not write optimOps inputs (link_assoc_reindexed/index_key): {e}")

    # Optional synthetic expression fixtures for optimOps -> register_zf tests.
    if bool(write_synthetic_zf_fixtures):
        try:
            write_synthetic_register_zf_fixtures(
                outdir,
                sim_pos=sim_pos,
                partition_labels=labels.astype(np.int32, copy=False),
                cell_ids=cell_ids,
                zf_flag=str(synthetic_zf_flag),
                slice_h5ad_path=(None if synthetic_zf_slice_path is None else str(synthetic_zf_slice_path)),
                write_slice_h5ad=bool(synthetic_zf_write_slice),
                slice_n=int(synthetic_zf_slice_n),
                num_pole_pairs=int(synthetic_zf_num_pole_pairs),
                genes_per_pole=int(synthetic_zf_genes_per_pole),
                seed=seed,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not write synthetic register_zf fixtures required for optimOps/register_zf testing: {e}"
            ) from e

    # --- Ground truth outputs
    gt_nodes = pd.DataFrame({
        "node_id": np.arange(n0 + n1, dtype=int),
        "partition_label": labels.astype(int),
        "cell_id": cell_ids.astype(int),
        "D_eff": D_nodes.astype(float),
        "delta_to_boundary": delta_nodes.astype(float),
        "ecs_factor": ecs_factor_nodes.astype(float),
        "membrane_factor": b_nodes.astype(float),
        "orig_csv_row": order.astype(int),
        "orig_id": arr[:, 0].astype(int, copy=False) if np.all(np.isfinite(arr[:, 0])) else arr[:, 0],
    })
    gt_nodes_path = outdir / "ground_truth_cells.csv"
    gt_nodes.to_csv(gt_nodes_path, index=False)

    gt_centers = pd.DataFrame(centers, columns=[f"center_{k}" for k in range(d)])
    gt_centers.insert(0, "cell_id", np.arange(centers.shape[0], dtype=int))
    gt_centers_path = outdir / "ground_truth_cell_centers.csv"
    gt_centers.to_csv(gt_centers_path, index=False)

    try:
        _dq = np.quantile(delta_nodes, [0.0, 0.05, 0.5, 0.95, 1.0]).tolist()
    except Exception:
        _dq = None

    meta = {
        "pos_csv": str(pos_csv),
        "d": int(d),
        "rescale": float(rescale),
        "scale_factor_applied": float(scale),
        "n0": int(n0),
        "n1": int(n1),
        "n_cells": int(n_cells),
        "total_ueis": int(L_total),
        "pi_short": float(pi_short),
        "sigma_s": float(sigma_s),
        "short_trunc": float(short_trunc),
        "k_capture": float(k_capture),
        "long_trunc": float(long_trunc),
        "long_eps": float(long_eps),
        "path_samples": int(path_samples),
        "path_mode": str(path_mode),
        "max_nbrs_per_source": int(max_nbrs_per_source),
        "threads_graph": int(n_threads),
        "graph_chunk_sources": int(chunk),
        "delta_nodes_quantiles": _dq,
        "diffusion_field": {
            "D_in": float(D_in),
            "D_out": float(D_out),
            "D_min": float(D_min),
            "D_max": float(D_max),
            "ecs_width_param": float(ecs_width),
            "ecs_width_frac_param": float(ecs_width_frac),
            "ecs_width_frac_of_spacing": (float(ecs_width) / float(center_spacing) if float(center_spacing) > 0 else None),
            "P_mem": (float(P_mem) if P_mem > 0 else None),
            "qp_modes": int(qp_modes),
            "qp_wavelength": (None if qp_wavelength is None else float(qp_wavelength)),
            "qp_amp": float(qp_amp),
            "qp_mode": str(qp_mode),
            "center_spacing_median": float(center_spacing),
            "membrane_width_param": float(membrane_width),
            "membrane_width_frac_param": float(membrane_width_frac),
            "membrane_width_frac_of_spacing": (float(membrane_width) / float(center_spacing) if float(center_spacing) > 0 else None),
            "cell_sigma": float(cell_sigma),
            "cell_q_corr": float(cell_q_corr),
            "membrane_strength": float(membrane_strength),
        },
        "corruption": {
            "false_edge_frac": float(false_edge_frac),
            "false_edge_targets_nodes": bool(false_edge_targets_nodes),
            "mode": ("node_fusion" if bool(false_edge_targets_nodes) else "edge_rewire"),
        },
        "files": {
            "graph_npz": str(graph_path.name),
            "ground_truth_nodes": str(gt_nodes_path.name),
            "ground_truth_centers": str(gt_centers_path.name),
        }
    }
    meta_path = outdir / "ground_truth_diffusion_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    # --- Visualization (best effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if d == 2:
            lo = np.min(sim_pos, axis=0)
            hi = np.max(sim_pos, axis=0)
            nx = 320
            ny = 320
            xs = np.linspace(lo[0], hi[0], nx)
            ys = np.linspace(lo[1], hi[1], ny)
            XX, YY = np.meshgrid(xs, ys, indexing='xy')
            grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
            Dg = field(grid, workers=n_threads).reshape(ny, nx)
            cid = field.assign_cells(grid, workers=n_threads).reshape(ny, nx)
            boundary = np.zeros((ny, nx), dtype=bool)
            boundary[:, 1:] |= (cid[:, 1:] != cid[:, :-1])
            boundary[1:, :] |= (cid[1:, :] != cid[:-1, :])

            delta_g, _ = field._boundary_distance(grid, workers=n_threads)
            if field.membrane_width > 0.0:
                bfac = np.exp(-(delta_g / field.membrane_width) ** 2).reshape(ny, nx)
            else:
                bfac = np.zeros((ny, nx), dtype=float)

            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

            im0 = ax0.imshow(np.log(Dg), origin='lower',
                             extent=(lo[0], hi[0], lo[1], hi[1]),
                             aspect='auto')
            ax0.contour(boundary.astype(float), levels=[0.5], origin='lower',
                        extent=(lo[0], hi[0], lo[1], hi[1]),
                        linewidths=0.6)
            ax0.scatter(pos0[:, 0], pos0[:, 1], s=2, alpha=0.5)
            ax0.scatter(pos1[:, 0], pos1[:, 1], s=2, alpha=0.5)
            ax0.set_title("log D_eff(x) + Voronoi boundaries")
            ax0.set_xlabel("x (scaled)")
            ax0.set_ylabel("y (scaled)")

            im1 = ax1.imshow(bfac, origin='lower',
                             extent=(lo[0], hi[0], lo[1], hi[1]),
                             aspect='auto')
            ax1.contour(boundary.astype(float), levels=[0.5], origin='lower',
                        extent=(lo[0], hi[0], lo[1], hi[1]),
                        linewidths=0.6)
            ax1.set_title(f"membrane factor exp(-(δ/w)^2), w={field.membrane_width:.3g}")
            ax1.set_xlabel("x (scaled)")
            ax1.set_ylabel("y (scaled)")

            fig.colorbar(im0, ax=ax0, shrink=0.85, label="log D_eff")
            fig.colorbar(im1, ax=ax1, shrink=0.85, label="b(x)")
            fig.tight_layout()
            fig.savefig(outdir / "ground_truth_diffusion.png", dpi=200)
            plt.close(fig)

        else:
            lo = np.min(sim_pos, axis=0)
            hi = np.max(sim_pos, axis=0)
            nx = 260
            ny = 260
            xs = np.linspace(lo[0], hi[0], nx)
            ys = np.linspace(lo[1], hi[1], ny)
            # Only the *middle* z cross-section: avoids obscuring the colorbar/legend.
            z = float(0.5 * (lo[2] + hi[2]))
            fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.6))
            XX, YY = np.meshgrid(xs, ys, indexing='xy')
            grid = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, z)], axis=1)
            Dg = field(grid, workers=n_threads).reshape(ny, nx)
            cid = field.assign_cells(grid, workers=n_threads).reshape(ny, nx)
            boundary = np.zeros((ny, nx), dtype=bool)
            boundary[:, 1:] |= (cid[:, 1:] != cid[:, :-1])
            boundary[1:, :] |= (cid[1:, :] != cid[:-1, :])

            im = ax.imshow(np.log(Dg), origin='lower',
                           extent=(lo[0], hi[0], lo[1], hi[1]),
                           aspect='auto')
            ax.contour(boundary.astype(float), levels=[0.5], origin='lower',
                       extent=(lo[0], hi[0], lo[1], hi[1]),
                       linewidths=0.5)
            ax.set_title(f"log D_eff(x) slice at z={z:.2f} (scaled coords)")
            ax.set_xlabel("x (scaled)")
            ax.set_ylabel("y (scaled)")
            fig.colorbar(im, ax=ax, shrink=0.85, label="log D_eff")
            fig.tight_layout()
            fig.savefig(outdir / "ground_truth_diffusion.png", dpi=200)
            plt.close(fig)

    except Exception as e:
        print(f"[WARN] ground-truth diffusion visualization failed: {e}")

    # --- Voronoi boundary isosurface (best effort; SSAO if voronoi_3d_patch.py is available)
    try:
        if bool(render_isosurface) and int(d) == 3:
            _ensure_voronoi_isosurface(
                centers=centers,
                sim_pos=sim_pos,
                labels=labels,
                field=field,
                outpath=outdir / "voronoi_isosurface.png",
                D0=float(D_in),
                D_min=float(D_min),
                D_max=float(D_max),
                qp_modes=int(qp_modes),
                qp_amp=float(qp_amp),
                cell_sigma=float(cell_sigma),
                cell_q_corr=float(cell_q_corr),
                membrane_width=float(getattr(field, "membrane_width", membrane_width)),
                membrane_width_frac=float(membrane_width_frac),
                membrane_strength=float(membrane_strength),
                grid_res=int(isosurface_grid_res),
                seed=(None if seed is None else int(seed)),
                panel_size=tuple(int(x) for x in isosurface_panel_size),
                show_points=bool(isosurface_show_points),
                point_size=float(isosurface_point_size),
            )
    except Exception as e:
        print(f"[WARN] Voronoi isosurface rendering failed: {e}")

    return graph_path, n0, n1





# ───────────────────── optimOps embedding visualization ─────────────────────

def _mpl_set_arial_fonts(*, base_size: float = 12.0) -> str:
    """Matplotlib font setup: download and register Arial if not already available.

    Returns the chosen primary font name.
    """
    import matplotlib
    matplotlib.use("Agg")  # safe for headless runs
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    avail = {f.name for f in fm.fontManager.ttflist}

    if "Arial" not in avail:
        # Try to find Arial in common system paths not yet registered
        import glob
        _search_dirs = [
            os.path.expanduser("~/.local/share/fonts"),
            os.path.expanduser("~/.fonts"),
            "/usr/share/fonts",
            "/usr/local/share/fonts",
        ]
        for _sd in _search_dirs:
            for _ttf in glob.glob(os.path.join(_sd, "**", "[Aa]rial*.ttf"), recursive=True):
                try:
                    fm.fontManager.addfont(_ttf)
                except Exception:
                    pass
        avail = {f.name for f in fm.fontManager.ttflist}

    if "Arial" not in avail:
        # Download Arial.ttf to a user-writable font directory
        _font_cache = os.path.join(
            os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
            "fonts",
        )
        os.makedirs(_font_cache, exist_ok=True)
        _arial_dest = os.path.join(_font_cache, "Arial.ttf")
        if not os.path.isfile(_arial_dest):
            _ARIAL_URLS = [
                "https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf",
                "https://github.com/JotJunior/PHP-Boleto-ZF2/raw/master/public/assets/fonts/arial.ttf",
            ]
            for _url in _ARIAL_URLS:
                try:
                    import urllib.request
                    urllib.request.urlretrieve(_url, _arial_dest)
                    break
                except Exception:
                    continue
        if os.path.isfile(_arial_dest):
            try:
                fm.fontManager.addfont(_arial_dest)
            except Exception:
                pass
            # Also look for bold/italic variants in the same cache dir
            for _ttf in fm.findSystemFonts(fontpaths=[_font_cache]):
                try:
                    fm.fontManager.addfont(_ttf)
                except Exception:
                    pass
        avail = {f.name for f in fm.fontManager.ttflist}

    primary = "Arial" if "Arial" in avail else (
        "Helvetica" if "Helvetica" in avail else (
            "Liberation Sans" if "Liberation Sans" in avail else "DejaVu Sans"
        )
    )

    if primary != "Arial":
        import warnings
        warnings.warn(
            f"Arial font not found on this system; falling back to '{primary}'. "
            "Install msttcorefonts or place Arial.ttf in a system font directory "
            "to get true Arial rendering.",
            stacklevel=2,
        )

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [primary, "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": float(base_size),
        "axes.titlesize": float(base_size) + 3.0,
        "axes.labelsize": float(base_size) + 1.5,
        "xtick.labelsize": float(base_size) - 0.5,
        "ytick.labelsize": float(base_size) - 0.5,
        "legend.fontsize": float(base_size) - 0.5,
        "figure.titlesize": float(base_size) + 3.0,
    })
    return primary


def _load_gse_output_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load optimOps GSEoutput.txt (comma-delimited; first col = node index)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    # Try strict comma format first; fall back to any whitespace if needed.
    try:
        arr = np.loadtxt(str(path), delimiter=",")
    except Exception:
        arr = np.loadtxt(str(path))

    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 2:
        raise ValueError(f"{path.name} had shape {arr.shape}; expected at least 2 columns (idx + coords).")

    idx = arr[:, 0].astype(np.int64, copy=False)
    coords = arr[:, 1:].astype(float, copy=False)
    # Sort by node index so we can align against other per-node arrays.
    order = np.argsort(idx)
    return idx[order], coords[order]


def _similarity_procrustes(src: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Similarity Procrustes alignment mapping `src` -> `ref`.

    Allows rotation, translation, reflection, and uniform rescaling.

    Returns (src_aligned, transform_dict).
    """
    X = np.asarray(src, dtype=float)
    Y = np.asarray(ref, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("src and ref must be 2D arrays")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"src/ref must have same number of points (got {X.shape[0]} vs {Y.shape[0]})")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"src/ref must have same dimensionality (got {X.shape[1]} vs {Y.shape[1]})")

    muX = X.mean(axis=0, keepdims=True)
    muY = Y.mean(axis=0, keepdims=True)
    X0 = X - muX
    Y0 = Y - muY

    ssX = float(np.sum(X0 * X0))
    if not np.isfinite(ssX) or ssX <= 1e-18:
        # Degenerate: just translate to match centroids.
        t = (muY - muX).ravel()
        X_al = X + t[None, :]
        return X_al, {"R": np.eye(X.shape[1]), "s": np.array(1.0), "t": t}

    M = X0.T @ Y0
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt  # includes reflection if it improves fit (allowed)
    Xr = X0 @ R
    s = float(np.sum(Xr * Y0) / ssX)  # optimal scalar
    t = (muY - s * (muX @ R)).ravel()

    X_aligned = (s * (X @ R)) + t[None, :]
    return X_aligned, {"R": R, "s": np.array(s), "t": t}


def _pad_or_truncate_to_3d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] == 3:
        return X
    if X.shape[1] > 3:
        return X[:, :3]
    out = np.zeros((X.shape[0], 3), dtype=float)
    out[:, :X.shape[1]] = X
    return out


def _search_upwards_for_file(start: Path, filename: str, *, max_up: int = 4) -> Path | None:
    """Search for `filename` in start/parents up to max_up levels."""
    p = Path(start).resolve()
    for _ in range(int(max_up) + 1):
        cand = p / filename
        if cand.exists():
            return cand
        if p.parent == p:
            break
        p = p.parent
    return None

def _compose_keep_nodes_global_chain(final_dir: Path, keep: np.ndarray | None, *, max_up: int = 4) -> np.ndarray | None:
    """Compose keep_nodes_global.npy mappings across nested dataset directories.

    This fixes the common two-stage reindexing that happens when:
      1) the builder writes an optimOps dataset on the largest connected component (LCC), producing:
           <run_dir>/keep_nodes_global.npy         : LCC_index -> simulator_global_index
      2) a downstream run may introduce additional nested keep_nodes_global.npy files.

    In that case, the *true* mapping back to simulator-global indices is the composition
    of all compatible parent keep-node maps.

    We implement this generically by walking up parent directories and composing as long as the
    current mapping appears to index into the next keep_nodes_global.npy that we encounter.
    """
    if keep is None:
        return None
    try:
        keep_comp = np.asarray(keep).astype(np.int64, copy=False).ravel()
    except Exception:
        return None
    if keep_comp.ndim != 1 or keep_comp.size == 0:
        return keep_comp

    p = Path(final_dir).resolve()
    # Start from the parent so we don't immediately "compose with ourselves".
    q = p.parent
    steps = 0
    while steps < int(max_up) and q is not None:
        cand = q / "keep_nodes_global.npy"
        if cand.exists():
            try:
                up = np.asarray(np.load(str(cand))).astype(np.int64, copy=False).ravel()
                if up.ndim == 1 and up.size > 0:
                    mx = int(np.max(keep_comp))
                    # Only compose when keep_comp is a valid index array into `up`.
                    if mx >= 0 and up.size > mx:
                        keep_comp = up[keep_comp]
                    else:
                        break
            except Exception:
                break
        if q.parent == q:
            break
        q = q.parent
        steps += 1
    return keep_comp



def _load_positions_scaled_for_dir(final_dir: Path, *, n_expected: int | None) -> tuple[np.ndarray | None, np.ndarray | None]:

    """Best-effort load of scaled node positions and (optional) simulator-global index mapping.

    Returns:
      (pos_scaled, keep_nodes_global)

    Notes
    -----
    * In the simple case (no nested reindexing), keep_nodes_global.npy maps dataset indices
      directly back to the simulator-global node index space.
    * In the coarsen case, optimOps exports a fine dataset whose keep_nodes_global.npy often maps
      fine indices back to the *upstream* dataset (typically the LCC). We therefore compose any
      parent keep_nodes_global.npy mappings that we find so that the returned keep_nodes_global
      always maps to simulator-global indices when possible.

    The returned pos_scaled is always in the dataset's node order (matching link_assoc_reindexed.npz
    and GSEoutput.txt index column).
    """
    final_dir = Path(final_dir)

    # Local mapping provided by this directory (may be "fine->LCC" in coarsen outputs).
    keep_local: np.ndarray | None = None

    keep_path = final_dir / "keep_nodes_global.npy"
    if keep_path.exists():
        try:
            keep_local = np.asarray(np.load(str(keep_path))).astype(np.int64, copy=False).ravel()
        except Exception:
            keep_local = None

    # Compose nested mappings to obtain a simulator-global mapping when possible.
    keep_global = _compose_keep_nodes_global_chain(final_dir, keep_local, max_up=4)
    keep_ret = keep_global if keep_global is not None else keep_local

    def _subset_positions(pos: np.ndarray) -> np.ndarray | None:
        """Return pos subsetted into the dataset's node order when possible."""
        pos = np.asarray(pos, dtype=float)
        if pos.ndim != 2 or pos.shape[0] <= 0:
            return None

        # Prefer simulator-global mapping when it indexes into `pos` (common: pos is full simulator array).
        if keep_ret is not None and getattr(keep_ret, "size", 0) > 0:
            try:
                mx = int(np.max(keep_ret))
                if mx >= 0 and pos.shape[0] > mx:
                    return pos[keep_ret]
            except Exception:
                pass

        # Fall back to local mapping when that indexes into `pos` (common: pos already LCC-sized).
        if keep_local is not None and getattr(keep_local, "size", 0) > 0:
            try:
                mx = int(np.max(keep_local))
                if mx >= 0 and pos.shape[0] > mx:
                    return pos[keep_local]
            except Exception:
                pass

        return None

    # Prefer positions stored alongside the dataset directory.
    for fname in ("node_positions_scaled.npy", "node_positions.npy"):
        p = final_dir / fname
        if p.exists():
            try:
                pos = np.asarray(np.load(str(p))).astype(float, copy=False)
                pos_sub = _subset_positions(pos)
                if pos_sub is not None:
                    if n_expected is None or pos_sub.shape[0] == int(n_expected):
                        return pos_sub, keep_ret
                if n_expected is None or (pos.ndim == 2 and pos.shape[0] == int(n_expected)):
                    return pos, keep_ret

            except Exception:
                pass

    # Search in parent directories.
    for fname in ("node_positions_scaled.npy", "node_positions.npy"):
        p2 = _search_upwards_for_file(final_dir, fname, max_up=4)
        if p2 is not None:
            try:
                pos = np.asarray(np.load(str(p2))).astype(float, copy=False)
                pos_sub = _subset_positions(pos)
                if pos_sub is not None:
                    pos = pos_sub
                if n_expected is None or (pos.ndim == 2 and pos.shape[0] == int(n_expected)):
                    return pos, keep_ret

            except Exception:
                pass

    # Last resort: reconstruct from the positions CSV referenced in ground_truth_diffusion_meta.json
    meta_path = _search_upwards_for_file(final_dir, "ground_truth_diffusion_meta.json", max_up=4)
    if meta_path is not None:
        try:
            with open(meta_path, "r") as fh:
                meta = json.load(fh)
            pos_csv = meta.get("pos_csv", None)
            scale = float(meta.get("scale_factor_applied", 1.0))
            if pos_csv is not None and os.path.exists(str(pos_csv)):
                arr = np.loadtxt(str(pos_csv), delimiter=',')
                if arr.ndim == 1:
                    arr = arr[None, :]
                order = np.lexsort((arr[:, 0], arr[:, 1]))
                arr = arr[order]
                pos_u = arr[:, 2:]
                pos_s = np.asarray(pos_u, dtype=float) * float(scale)
                pos_sub = _subset_positions(pos_s)
                if pos_sub is not None:
                    pos_s = pos_sub
                if n_expected is None or (pos_s.ndim == 2 and pos_s.shape[0] == int(n_expected)):
                    return pos_s, keep_ret



        except Exception:
            pass

    return None, keep_ret


def _load_true_cells_for_dir(final_dir: Path, *, n_expected: int | None, keep_nodes_global: np.ndarray | None) -> np.ndarray | None:
    """Load ground-truth Voronoi cell ids (if present) for ROC/AUC visualizations."""
    gt_path = _search_upwards_for_file(final_dir, "ground_truth_cells.csv", max_up=4)
    if gt_path is None:
        return None
    try:
        gt = pd.read_csv(gt_path)
        if "cell_id" not in gt.columns:
            return None
        cell = gt["cell_id"].to_numpy(dtype=int)
        if keep_nodes_global is not None and cell.shape[0] >= keep_nodes_global.max() + 1:
            cell = cell[keep_nodes_global]
        if n_expected is not None and cell.shape[0] != int(n_expected):
            return None
        return cell
    except Exception:
        return None


def _sample_edges_for_plot(A: csr_matrix,
                           *,
                           max_edges: int,
                           seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled (i,j) edges from a CSR matrix, with i<j and without duplicates."""
    from scipy.sparse import coo_matrix

    if not isinstance(A, csr_matrix):
        A = A.tocsr()
    A = A.copy()
    A.sum_duplicates()
    try:
        A.sort_indices()
    except Exception:
        pass

    coo = coo_matrix(A)
    if coo.nnz == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    # Keep one direction to avoid double-drawing.
    m = coo.row < coo.col
    row = coo.row[m].astype(np.int32, copy=False)
    col = coo.col[m].astype(np.int32, copy=False)
    if row.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    if max_edges <= 0 or row.size <= int(max_edges):
        return row, col

    rng = np.random.default_rng(seed)
    pick = rng.choice(row.size, size=int(max_edges), replace=False)
    return row[pick], col[pick]


def _render_pairwise_roc_from_embedding(
    X: np.ndarray,
    true_cell: np.ndarray,
    outpath: Path,
    *,
    n_pairs: int = 150000,
    seed: int | None = 0,
    title: str = "Pairwise ROC: same-cell vs embedding distance",
) -> None:
    """Compute and plot a ROC curve using embedding distances as scores.

    We sample pairs of nodes:
      - positives: same ground-truth cell
      - negatives: different ground-truth cells
    Score = -||x_i - x_j|| (larger => more likely same cell).
    """
    try:
        from sklearn.metrics import roc_curve, auc
    except Exception as e:
        raise ImportError("scikit-learn is required for ROC plots") from e

    X = np.asarray(X, dtype=float)
    y = np.asarray(true_cell, dtype=int).ravel()
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("X and true_cell must have compatible shapes")

    n = int(X.shape[0])
    if n < 4:
        return

    rng = np.random.default_rng(seed)

    # Index lists per cell
    from collections import defaultdict
    cell_to_idx: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(y.tolist()):
        cell_to_idx[int(c)].append(i)

    pos_cells = [c for c, idxs in cell_to_idx.items() if len(idxs) >= 2]
    all_cells = list(cell_to_idx.keys())
    if not pos_cells or len(all_cells) < 2:
        return

    m = int(max(2000, min(int(n_pairs), n * (n - 1) // 4)))
    m_pos = m // 2
    m_neg = m - m_pos

    # Sample positive pairs
    i_pos = np.empty(m_pos, dtype=np.int32)
    j_pos = np.empty(m_pos, dtype=np.int32)
    for k in range(m_pos):
        c = int(rng.choice(pos_cells))
        idxs = cell_to_idx[c]
        a, b = rng.choice(len(idxs), size=2, replace=False)
        i_pos[k] = int(idxs[a])
        j_pos[k] = int(idxs[b])

    # Sample negative pairs
    i_neg = np.empty(m_neg, dtype=np.int32)
    j_neg = np.empty(m_neg, dtype=np.int32)
    for k in range(m_neg):
        c1, c2 = rng.choice(all_cells, size=2, replace=False)
        i_neg[k] = int(rng.choice(cell_to_idx[int(c1)]))
        j_neg[k] = int(rng.choice(cell_to_idx[int(c2)]))

    I = np.concatenate([i_pos, i_neg])
    J = np.concatenate([j_pos, j_neg])
    y_true = np.concatenate([np.ones(m_pos, dtype=int), np.zeros(m_neg, dtype=int)])

    d = np.linalg.norm(X[I] - X[J], axis=1)
    scores = -d  # higher = closer = more likely same cell

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = float(auc(fpr, tpr))

    # Plot
    _mpl_set_arial_fonts(base_size=12.0)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.6))
    ax.plot(fpr, tpr, linewidth=2.0, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="0.6", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outpath), dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def render_optimops_embedding_scatter_3d(
    final_dir: Path | str,
    *,
    gse_name: str = "GSEoutput.txt",
    labels_name: str = "cluster_labels.npy",
    graph_name: str = "link_assoc_reindexed.npz",
    out_name: str = "GSE_embedding_3d.png",
    label_col: int = 0,
    label_col_name: str | None = None,
    max_edges: int = 35000,
    edge_seed: int | None = 0,
    force: bool = False,
    font_size: float = 12.5,
    show_top_down: bool = False,
) -> Path | None:
    """Render a 3D perspective scatter of optimOps embedding, colored by cluster labels, with faint graph edges.

    The embedding is Procrustes-aligned (similarity transform with reflection) to the simulator's
    scaled node positions when available so the view matches the Voronoi isosurface geometry.

    Returns the written image path, or None if required inputs were missing.
    """
    final_dir = Path(final_dir)
    gse_path = final_dir / str(gse_name)
    lab_path = final_dir / str(labels_name)
    graph_path = final_dir / str(graph_name)
    if not graph_path.exists():
        gp = _search_upwards_for_file(final_dir, str(graph_name), max_up=4)
        if gp is not None:
            graph_path = gp

    if not gse_path.exists() or not lab_path.exists():
        return None

    outpath = final_dir / str(out_name)
    if outpath.exists() and not force:
        return outpath

    idx, coords = _load_gse_output_txt(gse_path)
    X = _pad_or_truncate_to_3d(coords)

    labs_raw = np.load(str(lab_path))
    labs = np.asarray(labs_raw)
    if labs.ndim == 1:
        labs = labs.reshape(-1, 1)
    elif labs.ndim > 2:
        labs = labs.reshape(labs.shape[0], -1)
    labs = labs.astype(np.int32, copy=False)

    n = int(X.shape[0])
    if labs.shape[0] < int(idx.max() + 1):
        raise ValueError(f"{labels_name} has {labs.shape[0]} rows but {gse_name} references idx up to {int(idx.max())}.")
    # Reindex labels into the embedding row order.
    labs_ord = labs[idx, :]

    # Reference positions (scaled) for Procrustes alignment
    ref_pos, keep = _load_positions_scaled_for_dir(final_dir, n_expected=int(idx.max() + 1))
    ref_ord = None
    if ref_pos is not None and ref_pos.shape[0] >= int(idx.max() + 1):
        ref_ord = ref_pos[idx, :]
        # Align embedding to reference in 3D when possible.
        try:
            X_aligned, tf = _similarity_procrustes(X, _pad_or_truncate_to_3d(ref_ord))
            X = X_aligned
            aligned_note = f"Procrustes aligned (s={float(tf['s']):.3g})"
        except Exception:
            aligned_note = "Unaligned (Procrustes failed)"
    else:
        aligned_note = "Unaligned (no reference positions)"

    # Edge sampling (optional)
    edges_i = edges_j = None
    if graph_path.exists():
        try:
            from scipy.sparse import load_npz
            A = load_npz(str(graph_path)).tocsr()
            # Ensure indices exist; if adjacency is larger, subset by idx order.
            if A.shape[0] >= int(idx.max() + 1):
                # Restrict to the embedding node set (idx order) by slicing.
                A_sub = A[idx, :][:, idx]
            else:
                A_sub = A
            edges_i, edges_j = _sample_edges_for_plot(A_sub, max_edges=int(max_edges), seed=edge_seed)
        except Exception:
            edges_i = edges_j = None

    # --- Plot
    primary_font = _mpl_set_arial_fonts(base_size=float(font_size))
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib.lines import Line2D

    # Colors for clusters (use first clustering by default; show up to 12 in legend).
    col = int(label_col)
    if col < 0 or col >= labs_ord.shape[1]:
        col = 0
    lab0 = labs_ord[:, col].astype(int, copy=False)
    if label_col_name is None:
        label_col_name = f"labels{col}"
    uniq = np.unique(lab0)
    uniq = uniq[np.argsort(uniq)]
    has_noise = (-1 in uniq)
    uniq_core = uniq[uniq >= 0]
    n_clust = int(uniq_core.size)

    cmap = plt.get_cmap("tab20" if n_clust <= 20 else "gist_ncar")
    color_map: dict[int, tuple] = {}
    for k, lab in enumerate(uniq_core.tolist()):
        color_map[int(lab)] = cmap(k / max(1, n_clust - 1))
    if has_noise:
        color_map[-1] = (0.55, 0.55, 0.55, 1.0)

    colors = np.asarray([color_map.get(int(l), (0.2, 0.2, 0.2, 1.0)) for l in lab0], dtype=float)

    n_nodes = int(X.shape[0])
    s = 10.0 if n_nodes <= 6000 else (5.0 if n_nodes <= 20000 else 2.0)

    # Pre-compute reference bbox limits.
    if ref_ord is not None:
        P = _pad_or_truncate_to_3d(ref_ord)
        rlo = np.min(P, axis=0)
        rhi = np.max(P, axis=0)
        rpad = 0.03 * np.maximum(rhi - rlo, 1e-9)
        rlo = rlo - rpad
        rhi = rhi + rpad
    else:
        rlo = rhi = None

    def _populate_emb_ax(ax, *, elev: float, azim: float, subtitle: str | None = None,
                         depthshade: bool = True):
        """Fill a 3D axis with edges, nodes, and formatting."""
        try:
            ax.set_proj_type("persp", focal_length=1.0)
        except TypeError:
            try:
                ax.set_proj_type("persp")
            except Exception:
                pass
        ax.view_init(elev=elev, azim=azim)

        # Draw faint edges first (so nodes sit on top).
        if edges_i is not None and edges_i.size > 0:
            segs = np.stack([X[edges_i], X[edges_j]], axis=1)
            same = (lab0[edges_i] == lab0[edges_j])
            edge_cols = np.empty((segs.shape[0], 4), dtype=float)
            edge_cols[:] = (0.25, 0.25, 0.25, 0.07)
            if np.any(same):
                edge_cols[same] = colors[edges_i[same]]
                edge_cols[same, 3] = 0.12
            lc = Line3DCollection(segs, colors=edge_cols, linewidths=0.35)
            ax.add_collection3d(lc)

        # Node scatter
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=s, c=colors, alpha=0.92, depthshade=depthshade)

        if subtitle is not None:
            ax.set_title(subtitle, fontname=primary_font, fontsize=max(9.0, float(font_size) - 2.0))
        ax.set_xlabel("x", labelpad=10)
        ax.set_ylabel("y", labelpad=10)
        ax.set_zlabel("z", labelpad=8)
        if rlo is not None and rhi is not None:
            ax.set_xlim(rlo[0], rhi[0])
            ax.set_ylim(rlo[1], rhi[1])
            ax.set_zlim(rlo[2], rhi[2])
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass
        ax.grid(False)
        ax.set_facecolor("white")

    # --- Figure layout ---
    n_panels = 2 if show_top_down else 1
    fig_w = 9.6 if n_panels == 1 else 17.0
    fig = plt.figure(figsize=(fig_w, 8.4))

    # Diagonal view
    cam_dir = np.array([0.75, 0.58, 0.54], dtype=float)
    cam_dir = cam_dir / max(np.linalg.norm(cam_dir), 1e-12)
    diag_azim = float(np.degrees(np.arctan2(cam_dir[1], cam_dir[0])))
    diag_elev = float(np.degrees(np.arctan2(cam_dir[2], np.sqrt(cam_dir[0] ** 2 + cam_dir[1] ** 2))))

    ax_diag = fig.add_subplot(1, n_panels, 1, projection="3d")
    _populate_emb_ax(ax_diag, elev=diag_elev, azim=diag_azim,
                     subtitle="Diagonal view" if show_top_down else None)

    # Top-down view
    if show_top_down:
        ax_top = fig.add_subplot(1, n_panels, 2, projection="3d")
        _populate_emb_ax(ax_top, elev=90.0, azim=-90.0, subtitle="Top-down view (x-y plane)",
                         depthshade=False)

    # Axes formatting
    fig.suptitle(f"optimOps embedding (3D) colored by {label_col_name}  |  {aligned_note}",
                 fontname=primary_font, fontsize=float(font_size), y=0.98)

    fig.patch.set_facecolor("white")

    # Legend: show up to 12 largest clusters (+ noise)
    from collections import Counter
    counts = Counter(lab0.tolist())
    legend_labs = [lab for lab, _ in counts.most_common(12) if lab >= 0]
    if has_noise:
        legend_labs.append(-1)

    handles = []
    for lab in legend_labs:
        name = "noise" if lab == -1 else f"cluster {lab}"
        handles.append(Line2D([0], [0], marker="o", color="none",
                              markerfacecolor=color_map.get(int(lab), (0.2, 0.2, 0.2, 1.0)),
                              markeredgecolor="none", markersize=8,
                              label=f"{name} (n={counts.get(int(lab), 0)})"))
    if edges_i is not None and edges_i.size > 0:
        handles.append(Line2D([0], [0], color=(0.25, 0.25, 0.25, 0.35), lw=2.0, label="edges (sampled)"))

    # Place legend outside so it is never obscured by points.
    ax_diag.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   borderaxespad=0.0, frameon=True)

    fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outpath), dpi=240, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return outpath



def render_ground_truth_scatter_3d(
    final_dir: Path | str,
    *,
    gse_name: str = "GSEoutput.txt",
    graph_name: str = "link_assoc_reindexed.npz",
    out_name: str = "GSE_ground_truth_nodes_3d.png",
    max_edges: int = 35000,
    edge_seed: int | None = 0,
    force: bool = False,
    font_size: float = 12.5,
    cmap_name: str = "tab20",
    show_top_down: bool = False,
) -> Path | None:
    """Render ground-truth node positions colored by ground-truth Voronoi cell id.

    This plot is intentionally *independent* of any clustering / segmentation method:
      - coordinates: simulator reference space (scaled node positions)
      - colors: ground-truth Voronoi cell memberships (from ground_truth_cells.csv)
      - edges: sampled from link_assoc_reindexed.npz for connectivity context

    Returns the written image path, or None if required inputs were missing.
    """
    final_dir = Path(final_dir)
    gse_path = final_dir / str(gse_name)
    if not gse_path.exists():
        return None

    outpath = final_dir / str(out_name)
    if outpath.exists() and not force:
        return outpath

    # Use optimOps ordering so the picture matches embedding plots 1:1.
    idx, _coords = _load_gse_output_txt(gse_path)
    n_expected = int(idx.max() + 1)

    # Ground-truth positions (scaled) and Voronoi cell ids (properly LCC-aligned if needed).
    ref_pos, keep = _load_positions_scaled_for_dir(final_dir, n_expected=n_expected)
    if ref_pos is None or ref_pos.shape[0] < n_expected:
        return None

    true_cell = _load_true_cells_for_dir(final_dir, n_expected=n_expected, keep_nodes_global=keep)
    if true_cell is None or true_cell.shape[0] < n_expected:
        return None

    X = _pad_or_truncate_to_3d(ref_pos[idx, :])
    cell = np.asarray(true_cell[idx], dtype=int)

    # Edge sampling (optional).
    graph_path = final_dir / str(graph_name)
    if not graph_path.exists():
        gp = _search_upwards_for_file(final_dir, str(graph_name), max_up=4)
        if gp is not None:
            graph_path = gp

    edges_i = edges_j = None
    if graph_path.exists():
        try:
            from scipy.sparse import load_npz
            A = load_npz(str(graph_path)).tocsr()
            A_sub = A[idx, :][:, idx] if A.shape[0] >= n_expected else A
            edges_i, edges_j = _sample_edges_for_plot(A_sub, max_edges=int(max_edges), seed=edge_seed)
        except Exception:
            edges_i = edges_j = None

    title = (
        f"Ground-truth node positions (scaled) colored by Voronoi cell id  |  "
        f"n_cells={len(np.unique(cell))}"
    )
    ref_bbox = _pad_or_truncate_to_3d(ref_pos)

    _plot_cells_scatter_3d_common(
        X,
        cell,
        title=title,
        outpath=outpath,
        xlabel="x (scaled)",
        ylabel="y (scaled)",
        zlabel="z (scaled)",
        ref_bbox=ref_bbox,
        edges_i=edges_i,
        edges_j=edges_j,
        font_size=float(font_size),
        cmap_name=str(cmap_name),
        show_top_down=bool(show_top_down),
    )
    return outpath


def _plot_cells_scatter_3d_common(
    X: np.ndarray,
    cell: np.ndarray,
    *,
    title: str,
    outpath: Path,
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "z",
    ref_bbox: np.ndarray | None = None,
    edges_i: np.ndarray | None = None,
    edges_j: np.ndarray | None = None,
    font_size: float = 12.5,
    cmap_name: str = "tab20",
    show_top_down: bool = False,
) -> Path:
    """Shared 3D scatter renderer for plots colored by *ground-truth* cell id.

    Notes on the palette:
      - We treat cell ids as categorical labels (not a continuous scalar field).
      - The previous continuous colormaps (e.g. turbo/hsv) can make adjacent cell ids hard to
        visually separate; here we generate a categorical palette with stronger separation.
    """
    X3 = _pad_or_truncate_to_3d(np.asarray(X, dtype=float))
    cell = np.asarray(cell, dtype=int)

    primary_font = _mpl_set_arial_fonts(base_size=float(font_size))
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D

    def _categorical_palette(n: int, *, scheme: str = "tab20", seed: int = 0) -> list[tuple[float, float, float, float]]:
        """Return n visually separated RGBA colors for categorical labels."""
        if n <= 0:
            return []

        # Preferred: concatenate tab20/tab20b/tab20c (60 fairly distinct colors).
        scheme_l = str(scheme).lower().strip()
        if scheme_l in {"tab20", "categorical", "cells", "distinct"}:
            cols: list[tuple[float, float, float, float]] = []
            for nm in ("tab20", "tab20b", "tab20c"):
                try:
                    cm = plt.get_cmap(nm)
                    cols.extend([cm(i) for i in range(getattr(cm, "N", 20))])
                except Exception:
                    continue

            if len(cols) >= n:
                # Reorder indices to spread adjacent hues a bit (stable bit-reversal-like ordering).
                m = len(cols)
                order = np.arange(m, dtype=int)
                # simple "spread" permutation: take even indices then odd indices
                order = np.concatenate([order[::2], order[1::2]])
                return [cols[int(i)] for i in order[:n]]

            # Need more than 60 colors: extend with a golden-ratio HSV sequence.
            import colorsys
            rng = np.random.default_rng(int(seed))
            h0 = float(rng.random())
            gr = 0.618033988749895
            for i in range(n - len(cols)):
                h = (h0 + (len(cols) + i) * gr) % 1.0
                r, g, b = colorsys.hsv_to_rgb(h, 0.68, 0.95)
                cols.append((float(r), float(g), float(b), 1.0))
            return cols[:n]

        # Fallback: sample any matplotlib colormap, but use it as a discrete palette.
        try:
            cm = plt.get_cmap(str(scheme))
        except Exception:
            return _categorical_palette(n, scheme="tab20", seed=seed)

        if n == 1:
            return [cm(0.5)]
        xs = np.linspace(0.0, 1.0, n, endpoint=False)
        return [cm(float(x)) for x in xs]

    # --- Categorical label → color mapping (support optional noise label -1)
    uniq = np.unique(cell)
    uniq = uniq[np.argsort(uniq)]
    has_noise = (uniq.size > 0 and int(uniq[0]) == -1)
    core = uniq[uniq != -1]

    core_cols = _categorical_palette(int(core.size), scheme=str(cmap_name), seed=0)
    labels: list[int] = []
    cols: list[tuple[float, float, float, float]] = []
    if has_noise:
        labels.append(-1)
        cols.append((0.55, 0.55, 0.55, 1.0))
    labels.extend([int(x) for x in core.tolist()])
    cols.extend(core_cols)

    lab_to_idx = {int(lab): i for i, lab in enumerate(labels)}
    idx = np.asarray([lab_to_idx.get(int(v), 0) for v in cell], dtype=int)

    cmap = ListedColormap(cols, name="gt_cells")
    norm = BoundaryNorm(np.arange(-0.5, len(cols) + 0.5, 1.0), len(cols))
    colors = cmap(idx)

    # --- Shared helpers for populating a single 3D axis ---
    n_nodes = int(X3.shape[0])
    s = 10.0 if n_nodes <= 6000 else (5.0 if n_nodes <= 20000 else 2.0)

    # Pre-compute reference bounding box limits.
    P = _pad_or_truncate_to_3d(np.asarray(ref_bbox, dtype=float)) if ref_bbox is not None else X3
    lo = np.min(P, axis=0)
    hi = np.max(P, axis=0)
    pad = 0.03 * np.maximum(hi - lo, 1e-9)
    lo = lo - pad
    hi = hi + pad

    def _populate_ax(ax, *, elev: float, azim: float, subtitle: str | None = None,
                     depthshade: bool = True):
        """Fill a 3D axis with edges, nodes, and formatting."""
        try:
            ax.set_proj_type("persp", focal_length=1.0)
        except TypeError:
            try:
                ax.set_proj_type("persp")
            except Exception:
                pass
        ax.view_init(elev=elev, azim=azim)

        # Draw edges first.
        if edges_i is not None and edges_j is not None and edges_i.size > 0:
            segs = np.stack([X3[edges_i], X3[edges_j]], axis=1)
            same = (idx[edges_i] == idx[edges_j])
            edge_cols = np.empty((segs.shape[0], 4), dtype=float)
            edge_cols[:] = (0.25, 0.25, 0.25, 0.07)
            if np.any(same):
                edge_cols[same] = colors[edges_i[same]]
                edge_cols[same, 3] = 0.12
            lc = Line3DCollection(segs, colors=edge_cols, linewidths=0.35)
            ax.add_collection3d(lc)

        # Nodes
        ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], s=s, c=colors, alpha=0.92, depthshade=depthshade)

        if subtitle is not None:
            ax.set_title(subtitle, fontname=primary_font, fontsize=max(9.0, float(font_size) - 2.0))
        ax.set_xlabel(xlabel, labelpad=10)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_zlabel(zlabel, labelpad=8)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass
        ax.grid(False)
        ax.set_facecolor("white")

    # --- Figure layout ---
    n_panels = 2 if show_top_down else 1
    fig_w = 9.6 if n_panels == 1 else 17.0
    fig = plt.figure(figsize=(fig_w, 8.4))

    # Diagonal / side view (always present)
    cam_dir = np.array([0.75, 0.58, 0.54], dtype=float)
    cam_dir = cam_dir / max(np.linalg.norm(cam_dir), 1e-12)
    diag_azim = float(np.degrees(np.arctan2(cam_dir[1], cam_dir[0])))
    diag_elev = float(np.degrees(np.arctan2(cam_dir[2], np.sqrt(cam_dir[0] ** 2 + cam_dir[1] ** 2))))

    ax_diag = fig.add_subplot(1, n_panels, 1, projection="3d")
    _populate_ax(ax_diag, elev=diag_elev, azim=diag_azim,
                 subtitle="Diagonal view" if show_top_down else None)

    # Top-down view (elev=90 looks straight down the z-axis → x-y plane)
    if show_top_down:
        ax_top = fig.add_subplot(1, n_panels, 2, projection="3d")
        _populate_ax(ax_top, elev=90.0, azim=-90.0, subtitle="Top-down view (x-y plane)",
                     depthshade=False)

    fig.suptitle(title, fontname=primary_font, fontsize=float(font_size), y=0.98)
    fig.patch.set_facecolor("white")

    # Discrete colorbar (acts as the membership legend).
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb_right = 0.80 if n_panels == 1 else 0.88
    fig.subplots_adjust(right=cb_right)
    cax = fig.add_axes([cb_right + 0.03, 0.22, 0.02, 0.58])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Voronoi cell id")

    # Keep tick labels readable when there are many cells.
    n_lab = len(labels)
    if n_lab <= 15:
        tick_idx = np.arange(n_lab, dtype=int)
    else:
        tick_idx = np.unique(np.linspace(0, n_lab - 1, 12, dtype=int))
    cb.set_ticks(tick_idx)
    cb.set_ticklabels([str(labels[int(i)]) for i in tick_idx])
    cb.ax.tick_params(labelsize=max(8.0, float(font_size) - 3.0))

    # Tiny legend for edges (attach to the first axis).
    handles = []
    if edges_i is not None and edges_j is not None and edges_i.size > 0:
        handles.append(Line2D([0], [0], color=(0.25, 0.25, 0.25, 0.35), lw=2.0, label="edges (sampled)"))
        ax_diag.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.02, 0.02),
                       borderaxespad=0.0, frameon=True)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outpath), dpi=240, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return outpath


def render_embedding_ground_truth_scatter_3d(
    final_dir: Path | str,
    *,
    gse_name: str = "GSEoutput.txt",
    graph_name: str = "link_assoc_reindexed.npz",
    out_name: str = "GSE_embedding_ground_truth_nodes_3d.png",
    max_edges: int = 35000,
    edge_seed: int | None = 0,
    force: bool = False,
    font_size: float = 12.5,
    cmap_name: str = "tab20",
    show_top_down: bool = False,
) -> Path | None:
    """Render a Procrustes-aligned optimOps embedding colored by *ground-truth* cell id."""
    final_dir = Path(final_dir)
    gse_path = final_dir / str(gse_name)
    if not gse_path.exists():
        return None

    outpath = final_dir / str(out_name)
    if outpath.exists() and not force:
        return outpath

    idx, coords = _load_gse_output_txt(gse_path)
    n_expected = int(idx.max() + 1)
    X = _pad_or_truncate_to_3d(coords)

    # Reference positions for alignment + plotting limits.
    ref_pos, keep = _load_positions_scaled_for_dir(final_dir, n_expected=n_expected)
    aligned_note = "Unaligned (no reference positions)"
    if ref_pos is not None and ref_pos.shape[0] >= n_expected:
        ref_ord = _pad_or_truncate_to_3d(ref_pos[idx, :])
        try:
            X_aligned, tf = _similarity_procrustes(X, ref_ord)
            X = X_aligned
            aligned_note = f"Procrustes aligned (s={float(tf['s']):.3g})"
        except Exception:
            aligned_note = "Unaligned (Procrustes failed)"

    # Ground-truth Voronoi cell ids.
    true_cell = _load_true_cells_for_dir(final_dir, n_expected=n_expected, keep_nodes_global=keep)
    if true_cell is None or true_cell.shape[0] < n_expected:
        return None
    cell = np.asarray(true_cell[idx], dtype=int)

    # Edge sampling.
    graph_path = final_dir / str(graph_name)
    if not graph_path.exists():
        gp = _search_upwards_for_file(final_dir, str(graph_name), max_up=4)
        if gp is not None:
            graph_path = gp

    edges_i = edges_j = None
    if graph_path.exists():
        try:
            from scipy.sparse import load_npz
            A = load_npz(str(graph_path)).tocsr()
            if A.shape[0] >= n_expected:
                A_sub = A[idx, :][:, idx]
            else:
                A_sub = A
            edges_i, edges_j = _sample_edges_for_plot(A_sub, max_edges=int(max_edges), seed=edge_seed)
        except Exception:
            edges_i = edges_j = None

    title = f"optimOps embedding (3D) colored by ground-truth Voronoi cell id  |  {aligned_note}"
    ref_bbox = _pad_or_truncate_to_3d(ref_pos) if ref_pos is not None else None
    _plot_cells_scatter_3d_common(
        X,
        cell,
        title=title,
        outpath=outpath,
        xlabel="x",
        ylabel="y",
        zlabel="z",
        ref_bbox=ref_bbox,
        edges_i=edges_i,
        edges_j=edges_j,
        font_size=float(font_size),
        cmap_name=str(cmap_name),
        show_top_down=bool(show_top_down),
    )
    return outpath


def _graph_umap_cache_stem(*,
                           pca_dim: int,
                           n_neighbors: int,
                           min_dist: float) -> str:
    """Stable stem used for graph-only UMAP caches and derived artifacts."""
    return f"UMAP_embedding_3d_eigsh{int(pca_dim)}_nn{int(n_neighbors)}_md{float(min_dist):g}"




# ───────────────────── register_zf alignment visualization helpers ─────────────────────

_REGZF_ALIGNMENT_REQUIRED = (
    "slice_smoothed_ratio_fields.npz",
    "slice_assigned_aggregated_feature_maps.npz",
    "aggregated_nodes_slice_mapped_coords.npz",
)

# Current optimOps coarsen/register_zf artifacts.  The final solve remains the
# ordinary scalar GSE solve; these files document the coarse registration, the
# fine-node prior moments, and the post-GSE prior refinement that publishes the
# aligned GSEoutput.txt used for final clustering.
_REGZF_FINE_MOMENTS_NPZ = "coarsen_align_fine_node_mu_cov.npz"
_REGZF_FINE_MOMENTS_META = "coarsen_align_fine_node_mu_cov.meta.json"
_REGZF_PRIOR_REFINE_META = "coarsen_align_prior_refine.meta.json"
_REGZF_PRIOR_REFINE_REGISTERED_NPZ = "coarsen_align_reference_registered.npz"


def _register_zf_match_dir_for_run(run_dir: Path | str, zf_flag: str) -> Path:
    zf_flag = str(zf_flag).strip().lower()
    return Path(run_dir) / "coarsened_largest_component" / "component0" / f"match_result_{zf_flag}"


def _npz_nonempty(path: Path | str, required_keys: tuple[str, ...] = ()) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    if not required_keys:
        return True
    try:
        with np.load(str(path), allow_pickle=True) as z:
            return set(required_keys).issubset(set(z.files))
    except Exception:
        return False


def _json_file(path: Path | str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _register_zf_backend_outputs_ready(run_dir: Path | str) -> bool:
    """Return True when current optimOps coarsen/register_zf artifacts are usable."""
    run_dir = Path(run_dir)

    required_files = (
        "coarse_anchor_coords_final_gse.npy",
        "coarse_anchor_coords_registered.npy",
        "coarse_anchor_coords_registered.meta.json",
        "coarse_anchor_coords_prior_mean.npy",
        "coarse_anchor_coords_witness.npy",
        "transformed_matrix_fine_to_coarse.npz",
        _REGZF_FINE_MOMENTS_NPZ,
        _REGZF_FINE_MOMENTS_META,
        _REGZF_PRIOR_REFINE_META,
        _REGZF_PRIOR_REFINE_REGISTERED_NPZ,
    )
    if not all((run_dir / name).exists() for name in required_files):
        return False

    prior_meta = _json_file(run_dir / _REGZF_FINE_MOMENTS_META)
    if not isinstance(prior_meta, dict):
        return False
    if str(prior_meta.get("mode", "")) != "coarsen_register_zf_prior_generation":
        return False
    if int(prior_meta.get("Npts", 0)) <= 0 or int(prior_meta.get("inference_dim", 0)) <= 0:
        return False

    refine_meta = _json_file(run_dir / _REGZF_PRIOR_REFINE_META)
    if not isinstance(refine_meta, dict):
        return False
    if int(refine_meta.get("layout_version", -1)) < 1:
        return False

    try:
        with np.load(str(run_dir / _REGZF_FINE_MOMENTS_NPZ), allow_pickle=False) as z:
            if "mu" not in z.files or ("cov_observed" not in z.files and "cov" not in z.files):
                return False
            mu = z["mu"]
            return mu.ndim == 2 and int(mu.shape[0]) == int(prior_meta.get("Npts", mu.shape[0])) and int(mu.shape[1]) > 0
    except Exception:
        return False


def _load_register_zf_fine_prior_for_run(run_dir: Path | str) -> tuple[np.ndarray | None, int | None, dict[str, Any]]:
    """Load the fine-node register_zf prior moments written by current optimOps."""
    run_dir = Path(run_dir)
    prior_path = run_dir / _REGZF_FINE_MOMENTS_NPZ
    meta_path = run_dir / _REGZF_FINE_MOMENTS_META
    if not prior_path.exists():
        return None, None, {"available": False, "reason": f"{_REGZF_FINE_MOMENTS_NPZ}_not_found"}

    try:
        with np.load(str(prior_path), allow_pickle=False) as z:
            if "mu" not in z.files:
                return None, None, {"available": False, "reason": "fine_prior_missing_mu"}
            prior = np.asarray(z["mu"], dtype=float)
            if "obs_dim" in z.files:
                obs_dim = int(np.asarray(z["obs_dim"]).reshape(-1)[0])
            else:
                cov_key = "cov_observed" if "cov_observed" in z.files else ("cov" if "cov" in z.files else None)
                obs_dim = int(z[cov_key].shape[1]) if cov_key is not None else min(2, int(prior.shape[1]))
    except Exception as exc:
        return None, None, {"available": False, "reason": "fine_prior_load_failed", "error": str(exc)}
    if prior.ndim != 2:
        return None, None, {"available": False, "reason": "fine_prior_mu_not_2d", "shape": list(prior.shape)}

    prior_meta = _json_file(meta_path) or {}
    refine_meta = _json_file(run_dir / _REGZF_PRIOR_REFINE_META) or {}
    anchor_meta = _json_file(run_dir / "coarse_anchor_coords_registered.meta.json") or {}
    obs_dim = max(1, min(int(obs_dim), int(prior.shape[1])))

    report = {
        "available": True,
        "source": _REGZF_FINE_MOMENTS_NPZ,
        "prior_shape": [int(x) for x in prior.shape],
        "obs_dim": int(obs_dim),
        "fine_moments_meta": prior_meta,
        "prior_refine_meta": refine_meta,
        "anchor_meta": anchor_meta,
        "moments_persisted": True,
    }
    return prior, int(obs_dim), report


def _rank_rescale_over_mask(values: np.ndarray, mask: np.ndarray | None = None, *, neutral: float = 0.5) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    if mask is None:
        valid = np.isfinite(values)
    else:
        valid = np.asarray(mask, dtype=bool).ravel() & np.isfinite(values)
    out = np.full(values.shape, float(neutral), dtype=float)
    if int(valid.sum()) < 2:
        return out
    try:
        from scipy import stats as _stats
        ranks = _stats.rankdata(values[valid], method="average").astype(float)
    except Exception:
        order = np.argsort(values[valid], kind="mergesort")
        ranks = np.empty(int(valid.sum()), dtype=float)
        ranks[order] = np.arange(1, int(valid.sum()) + 1, dtype=float)
    rmin = float(np.min(ranks))
    rmax = float(np.max(ranks))
    if rmax > rmin:
        out[valid] = (ranks - rmin) / (rmax - rmin)
    return out


def _robust_abs_limit(x: np.ndarray, q: float = 0.995) -> float:
    vals = np.asarray(x, dtype=float).ravel()
    finite = np.isfinite(vals)
    if not np.any(finite):
        return 1.0
    abs_vals = np.abs(vals[finite])
    vmax = float(np.quantile(abs_vals, float(q))) if abs_vals.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(abs_vals)) if abs_vals.size else 1.0
    return float(vmax if vmax > 0.0 else 1.0)


def _mpl_set_register_zf_plot_fonts(*, base_size: float = 11.0) -> None:
    """Lightweight sans-serif plot defaults without attempting font downloads."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": float(base_size),
        "axes.titlesize": float(base_size) + 1.0,
        "axes.labelsize": float(base_size),
        "xtick.labelsize": float(base_size) - 1.0,
        "ytick.labelsize": float(base_size) - 1.0,
        "legend.fontsize": float(base_size) - 1.0,
        "savefig.dpi": 260,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return float("nan")
    xv = x[valid]
    yv = y[valid]
    if float(np.std(xv)) <= 1e-12 or float(np.std(yv)) <= 1e-12:
        return float("nan")
    try:
        from scipy import stats as _stats
        rho = _stats.spearmanr(xv, yv).correlation
    except Exception:
        xr = _rank_rescale_over_mask(xv, None)
        yr = _rank_rescale_over_mask(yv, None)
        rho = np.corrcoef(xr, yr)[0, 1]
    return float(rho) if np.isfinite(rho) else float("nan")


def _alignment_channel_bundle(match_dir: Path, *, pair_idx: int, rank_neutral: float = 0.5) -> dict[str, Any]:
    maps = np.load(match_dir / "slice_assigned_aggregated_feature_maps.npz", allow_pickle=True)
    slice_data = np.load(match_dir / "slice_smoothed_ratio_fields.npz", allow_pickle=True)
    mapped = np.load(match_dir / "aggregated_nodes_slice_mapped_coords.npz", allow_pickle=True)

    coords = np.asarray(maps["coords"], dtype=float)
    slice_vals_all = np.asarray(
        slice_data["ratio_feature_01"] if "ratio_feature_01" in slice_data.files else slice_data["ratio"],
        dtype=float,
    )
    if slice_vals_all.ndim == 1:
        slice_vals_all = slice_vals_all[:, None]
    support_all = np.asarray(slice_data["support"], dtype=float) if "support" in slice_data.files else np.ones_like(slice_vals_all)
    if support_all.ndim == 1:
        support_all = support_all[:, None]

    base_all = np.asarray(maps["feature_mean_base"], dtype=float)
    final_all = np.asarray(maps["feature_mean_final"], dtype=float)
    if base_all.ndim == 1:
        base_all = base_all[:, None]
    if final_all.ndim == 1:
        final_all = final_all[:, None]

    pair_ids_maps = np.ravel(maps["pair_ids"]) if "pair_ids" in maps.files else np.arange(base_all.shape[1])
    pair_ids_slice = np.ravel(slice_data["pair_ids"]) if "pair_ids" in slice_data.files else np.arange(slice_vals_all.shape[1])
    pair_idx = int(max(0, min(pair_idx, base_all.shape[1] - 1)))
    channel_name = str(pair_ids_maps[pair_idx]) if pair_ids_maps.size else f"pair{pair_idx:02d}"
    try:
        slice_pair_idx = [str(x) for x in pair_ids_slice].index(channel_name)
    except ValueError:
        slice_pair_idx = pair_idx
    slice_pair_idx = int(max(0, min(slice_pair_idx, slice_vals_all.shape[1] - 1)))

    raw_slice = np.asarray(slice_vals_all[:, slice_pair_idx], dtype=float)
    raw_base = np.asarray(base_all[:, pair_idx], dtype=float)
    raw_final = np.asarray(final_all[:, pair_idx], dtype=float)
    support = np.asarray(support_all[:, slice_pair_idx], dtype=float)

    finite_slice = np.isfinite(raw_slice)
    finite_common = np.isfinite(raw_slice) & np.isfinite(raw_base) & np.isfinite(raw_final)
    eval_mask = finite_common & (support > 1e-10)
    if int(eval_mask.sum()) < 2:
        eval_mask = finite_common
    if int(eval_mask.sum()) < 2:
        eval_mask = finite_slice
    plot_mask = finite_slice

    slice_vals = _rank_rescale_over_mask(raw_slice, eval_mask, neutral=float(rank_neutral))
    base_vals = _rank_rescale_over_mask(raw_base, eval_mask, neutral=float(rank_neutral))
    final_vals = _rank_rescale_over_mask(raw_final, eval_mask, neutral=float(rank_neutral))
    non_eval = plot_mask & (~eval_mask)
    slice_vals[non_eval] = float(rank_neutral)
    base_vals[non_eval] = float(rank_neutral)
    final_vals[non_eval] = float(rank_neutral)

    base_resid = base_vals - slice_vals
    final_resid = final_vals - slice_vals
    delta_vals = final_vals - base_vals
    base_resid[non_eval] = 0.0
    final_resid[non_eval] = 0.0
    delta_vals[non_eval] = 0.0

    count_base = np.asarray(maps["count_base"], dtype=float) if "count_base" in maps.files else np.zeros(coords.shape[0])
    count_final = np.asarray(maps["count_final"], dtype=float) if "count_final" in maps.files else np.zeros(coords.shape[0])
    count_delta = np.asarray(count_final, dtype=float).ravel() - np.asarray(count_base, dtype=float).ravel()

    return {
        "coords": coords,
        "plot_mask": plot_mask,
        "eval_mask": eval_mask,
        "channel_name": channel_name,
        "slice_vals": slice_vals,
        "base_vals": base_vals,
        "final_vals": final_vals,
        "delta_vals": delta_vals,
        "base_resid": base_resid,
        "final_resid": final_resid,
        "count_base": np.asarray(count_base, dtype=float).ravel(),
        "count_final": np.asarray(count_final, dtype=float).ravel(),
        "count_delta": count_delta,
        "mapped": mapped,
    }


def _plot_register_zf_pair_alignment(match_dir: Path, *, pair_idx: int, point_size: float, robust_q: float, rank_neutral: float, force: bool = False) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _mpl_set_register_zf_plot_fonts(base_size=11.0)
    out_path = match_dir / f"register_zf_alignment_pair{int(pair_idx):02d}.png"
    if out_path.exists() and not force:
        return out_path

    b = _alignment_channel_bundle(match_dir, pair_idx=int(pair_idx), rank_neutral=float(rank_neutral))
    coords = np.asarray(b["coords"], dtype=float)
    plot_mask = np.asarray(b["plot_mask"], dtype=bool)
    eval_mask = np.asarray(b["eval_mask"], dtype=bool)
    channel_name = str(b["channel_name"])

    residual_abs = _robust_abs_limit(np.concatenate([b["base_resid"][eval_mask], b["final_resid"][eval_mask]]), q=float(robust_q))
    delta_abs = _robust_abs_limit(np.asarray(b["delta_vals"])[eval_mask], q=float(robust_q))
    count_delta = np.asarray(b["count_delta"], dtype=float)
    count_abs = _robust_abs_limit(count_delta, q=float(robust_q))

    fig, axes = plt.subplots(2, 4, figsize=(23, 11), constrained_layout=True)
    panels = [
        (b["slice_vals"], "viridis", 0.0, 1.0, f"Slice target\n{channel_name}\npost-rank [0,1]", "rank"),
        (b["base_vals"], "viridis", 0.0, 1.0, "Transported aggregate mean\nbase assignment", "rank"),
        (b["final_vals"], "viridis", 0.0, 1.0, "Transported aggregate mean\ngraph-refined assignment", "rank"),
        (b["delta_vals"], "coolwarm", -delta_abs, delta_abs, "Graph-refinement delta\nfinal − base", "rank difference"),
        (b["base_resid"], "coolwarm", -residual_abs, residual_abs, "Base − slice target\npost-rank residual", "rank residual"),
        (b["final_resid"], "coolwarm", -residual_abs, residual_abs, "Graph-refined − slice target\npost-rank residual", "rank residual"),
        (count_delta, "coolwarm", -count_abs, count_abs, "Per-slice assignment count change\ncount_final − count_base", "count change"),
    ]
    for ax, (vals, cmap, vmin, vmax, title, cbar_label) in zip(axes.ravel()[:7], panels):
        vals = np.asarray(vals, dtype=float)
        sc = ax.scatter(coords[plot_mask, 0], coords[plot_mask, 1], c=vals[plot_mask], s=float(point_size), cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("slice x")
        ax.set_ylabel("slice y")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.85, label=cbar_label)

    ax = axes.ravel()[7]
    mapped = b["mapped"]
    moved_mask = np.asarray(mapped["moved_mask"]).astype(bool) if "moved_mask" in mapped.files else np.zeros(0, dtype=bool)
    coords_final = np.asarray(mapped["coords_final"], dtype=float) if "coords_final" in mapped.files else np.zeros((0, 2))
    move_dist = np.asarray(mapped["move_distance_normalized"], dtype=float) if "move_distance_normalized" in mapped.files else np.zeros(coords_final.shape[0])
    ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=max(1.0, float(point_size) * 0.5), linewidths=0, alpha=0.25, label="slice nodes")
    if moved_mask.size and coords_final.shape[0] == moved_mask.shape[0] and np.any(moved_mask):
        sc = ax.scatter(coords_final[moved_mask, 0], coords_final[moved_mask, 1], c=move_dist[moved_mask], s=max(1.0, float(point_size) * 0.7), cmap="magma", linewidths=0, label="moved agg nodes")
        fig.colorbar(sc, ax=ax, shrink=0.85, label="normalized move distance")
    ax.set_title("Where graph refinement changed\naggregated-node assignments")
    ax.set_xlabel("slice x")
    ax.set_ylabel("slice y")
    ax.set_aspect("equal")
    ax.legend(loc="best")

    fig.suptitle(f"register_zf alignment diagnostic: {match_dir.parent.name}/{match_dir.name}", y=1.02)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _unit_or_rank_rescale(values: np.ndarray, mask: np.ndarray | None = None, *, neutral: float = 0.5) -> np.ndarray:
    """Use [0, 1] values directly when already scaled; otherwise rank-rescale over mask."""
    values = np.asarray(values, dtype=float).ravel()
    out = np.full(values.shape, float(neutral), dtype=float)

    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).ravel()
    if int(valid.sum()) < 2:
        return out

    vmin = float(np.nanmin(values[valid]))
    vmax = float(np.nanmax(values[valid]))
    if vmin >= -1e-6 and vmax <= 1.0 + 1e-6:
        out[valid] = np.clip(values[valid], 0.0, 1.0)
    else:
        out = _rank_rescale_over_mask(values, valid, neutral=float(neutral))
    return out


def _plot_register_zf_slice_superposition(
    match_dir: Path,
    *,
    pair_idx: int,
    point_size: float,
    rank_neutral: float = 0.5,
    force: bool = False,
) -> Path | None:
    """Overlay the slice target field and assigned aggregate-node field on the same slice axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _mpl_set_register_zf_plot_fonts(base_size=11.0)

    match_dir = Path(match_dir)
    out_path = match_dir / f"register_zf_slice_superposition_pair{int(pair_idx):02d}.png"
    if out_path.exists() and not force:
        return out_path

    slice_path = match_dir / "slice_smoothed_ratio_fields.npz"
    agg_path = match_dir / "aggregated_gse_ranked_ratio_vectors.npz"
    mapped_path = match_dir / "aggregated_nodes_slice_mapped_coords.npz"
    maps_path = match_dir / "slice_assigned_aggregated_feature_maps.npz"
    if not all(p.exists() for p in (slice_path, agg_path, mapped_path, maps_path)):
        return None

    with np.load(slice_path, allow_pickle=True) as s, \
         np.load(agg_path, allow_pickle=True) as a, \
         np.load(mapped_path, allow_pickle=True) as m, \
         np.load(maps_path, allow_pickle=True) as maps:

        slice_xy = np.asarray(s["coords"], dtype=float)
        agg_xy = np.asarray(m["coords_final"], dtype=float)

        slice_mat = np.asarray(
            s["ratio_feature_01"] if "ratio_feature_01" in s.files else s["ratio"],
            dtype=float,
        )
        agg_mat = np.asarray(
            a["ratio_feature_01"] if "ratio_feature_01" in a.files else a["ratio_raw"],
            dtype=float,
        )
        if slice_mat.ndim == 1:
            slice_mat = slice_mat[:, None]
        if agg_mat.ndim == 1:
            agg_mat = agg_mat[:, None]

        support = np.asarray(s["support"], dtype=float) if "support" in s.files else np.ones_like(slice_mat)
        if support.ndim == 1:
            support = support[:, None]

        slice_pair_ids = [str(x) for x in (np.ravel(s["pair_ids"]) if "pair_ids" in s.files else np.arange(slice_mat.shape[1]))]
        agg_pair_ids = [str(x) for x in (np.ravel(a["pair_ids"]) if "pair_ids" in a.files else np.arange(agg_mat.shape[1]))]

        pair_idx = int(max(0, min(int(pair_idx), slice_mat.shape[1] - 1)))
        channel_name = slice_pair_ids[pair_idx] if slice_pair_ids else f"pair{pair_idx:02d}"
        try:
            agg_pair_idx = agg_pair_ids.index(channel_name)
        except ValueError:
            agg_pair_idx = min(pair_idx, agg_mat.shape[1] - 1)

        support_idx = min(pair_idx, support.shape[1] - 1)
        slice_vals = _unit_or_rank_rescale(
            slice_mat[:, pair_idx],
            support[:, support_idx] > 1e-10,
            neutral=float(rank_neutral),
        )
        agg_vals = _unit_or_rank_rescale(
            agg_mat[:, agg_pair_idx],
            None,
            neutral=float(rank_neutral),
        )
        slice_capacity = np.asarray(maps["slice_capacity"], dtype=float) if "slice_capacity" in maps.files else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    ax = axes[0]
    ax.scatter(
        slice_xy[:, 0],
        slice_xy[:, 1],
        c=slice_vals,
        s=max(1.0, float(point_size) * 0.5),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        alpha=0.35,
        linewidths=0,
        label="slice target field",
    )
    sc = ax.scatter(
        agg_xy[:, 0],
        agg_xy[:, 1],
        c=agg_vals,
        s=max(2.0, float(point_size)),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        alpha=0.90,
        linewidths=0.15,
        edgecolors="black",
        label="assigned aggregate nodes",
    )
    ax.set_title(f"Superposition: slice target + assigned aggregate field\n{channel_name}")
    ax.set_xlabel("slice x")
    ax.set_ylabel("slice y")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, shrink=0.85, label="rank-rescaled ratio")

    ax = axes[1]
    if slice_capacity is not None and slice_capacity.size == slice_xy.shape[0]:
        finite_capacity = slice_capacity[np.isfinite(slice_capacity)]
        vmax = float(np.nanquantile(finite_capacity, 0.99)) if finite_capacity.size else 1.0
        vmax = max(vmax, 1.0)
        sc2 = ax.scatter(
            slice_xy[:, 0],
            slice_xy[:, 1],
            c=slice_capacity,
            s=max(1.0, float(point_size) * 0.5),
            cmap="Greys",
            vmin=0.0,
            vmax=vmax,
            alpha=0.60,
            linewidths=0,
        )
        fig.colorbar(sc2, ax=ax, shrink=0.85, label="slice capacity")
    else:
        ax.scatter(
            slice_xy[:, 0],
            slice_xy[:, 1],
            c="lightgray",
            s=max(1.0, float(point_size) * 0.5),
            alpha=0.35,
            linewidths=0,
        )

    ax.scatter(
        agg_xy[:, 0],
        agg_xy[:, 1],
        c="tab:red",
        s=max(1.0, float(point_size) * 0.6),
        alpha=0.45,
        linewidths=0,
        label="assigned aggregate footprint",
    )
    ax.set_title("Spatial support: slice capacity/background + assignment footprint")
    ax.set_xlabel("slice x")
    ax.set_ylabel("slice y")
    ax.set_aspect("equal")
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out_path



def _similarity_report_2d(src: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    """Best-fit 2D similarity report for two point sets with known row pairing."""
    X = np.asarray(src, dtype=float)
    Y = np.asarray(ref, dtype=float)
    report: dict[str, Any] = {
        "available": False,
        "n_src": int(X.shape[0]) if X.ndim >= 1 else 0,
        "n_ref": int(Y.shape[0]) if Y.ndim >= 1 else 0,
    }
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] < 2 or Y.shape[1] < 2:
        report["reason"] = "inputs_are_not_2d_coordinate_matrices"
        return report
    if X.shape[0] != Y.shape[0] or X.shape[0] < 2:
        report["reason"] = "row_counts_differ_or_too_small_for_paired_procrustes"
        return report
    X2 = np.asarray(X[:, :2], dtype=float)
    Y2 = np.asarray(Y[:, :2], dtype=float)
    finite = np.isfinite(X2).all(axis=1) & np.isfinite(Y2).all(axis=1)
    if int(finite.sum()) < 2:
        report["reason"] = "too_few_finite_paired_rows"
        return report
    X2 = X2[finite]
    Y2 = Y2[finite]
    aligned, tf = _similarity_procrustes(X2, Y2)
    resid = aligned - Y2
    rms = float(np.sqrt(np.mean(np.sum(resid * resid, axis=1))))
    span = float(np.linalg.norm(np.ptp(Y2, axis=0)))
    if not np.isfinite(span) or span <= 1.0e-12:
        span = 1.0
    R = np.asarray(tf.get("R", np.eye(2)), dtype=float)
    if R.shape != (2, 2):
        R = np.eye(2, dtype=float)
    angle_deg = float(np.degrees(np.arctan2(R[0, 1], R[0, 0])))
    det = float(np.linalg.det(R))
    return {
        **report,
        "available": True,
        "n_used": int(X2.shape[0]),
        "scale": float(np.asarray(tf.get("s", 1.0)).ravel()[0]),
        "determinant": det,
        "rotation_angle_degrees": angle_deg,
        "translation": [float(x) for x in np.asarray(tf.get("t", np.zeros(2))).ravel()[:2]],
        "rms_residual": rms,
        "normalized_rms_residual": float(rms / span),
        "max_abs_identity_error": float(np.max(np.abs(X2 - Y2))) if X2.shape == Y2.shape else None,
    }


def _load_anchor_exact_slice_witness(
    run_dir: Path,
    slice_xy: np.ndarray,
    *,
    obs_dim: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load exact slice-row coordinates for registered anchors when available."""
    idx_path = Path(run_dir) / "coarse_anchor_slice_indices.npy"
    report: dict[str, Any] = {"available": False, "path": str(idx_path)}
    if not idx_path.exists():
        report["reason"] = "coarse_anchor_slice_indices.npy_not_found"
        return None, report
    try:
        idx = np.asarray(np.load(idx_path), dtype=np.int64).ravel()
        slice_xy = np.asarray(slice_xy, dtype=float)
        obs_dim = int(max(1, min(obs_dim, slice_xy.shape[1])))
        valid = idx.size > 0 and np.all(idx >= 0) and np.all(idx < slice_xy.shape[0])
        report.update({"available": bool(valid), "n_indices": int(idx.size)})
        if not valid:
            report["reason"] = "slice_indices_out_of_bounds_or_empty"
            return None, report
        return np.asarray(slice_xy[idx, :obs_dim], dtype=float), report
    except Exception as exc:
        report["reason"] = "load_failed"
        report["error"] = str(exc)
        return None, report


def _plot_register_zf_xy_frame_diagnostic(
    run_dir: Path | str,
    match_dir: Path | str,
    *,
    gse_name: str = "GSEoutput.txt",
    max_points: int = 12000,
    force: bool = False,
) -> Path | None:
    """Top-down shared-frame check for slice, registered anchors, lifted prior, and final GSE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _mpl_set_register_zf_plot_fonts(base_size=11.0)
    run_dir = Path(run_dir)
    match_dir = Path(match_dir)
    out_path = match_dir / "register_zf_alignment_xy_frame.png"
    report_path = match_dir / "register_zf_xy_frame_report.json"
    if out_path.exists() and report_path.exists() and not force:
        return out_path

    gse_path = run_dir / str(gse_name)
    slice_path = match_dir / "slice_smoothed_ratio_fields.npz"
    if not gse_path.exists() or not slice_path.exists():
        return None

    idx, coords = _load_gse_output_txt(gse_path)
    X = np.asarray(coords, dtype=float)
    prior_index = np.asarray(idx, dtype=np.int64)
    prior, obs_dim_loaded, lift_report = _load_register_zf_fine_prior_for_run(run_dir)
    if prior is None or obs_dim_loaded is None:
        return None
    if prior.ndim != 2 or prior.shape[0] <= int(prior_index.max()):
        return None
    prior_ord = prior[prior_index]
    obs_dim = max(1, min(2, int(obs_dim_loaded), X.shape[1], prior_ord.shape[1]))

    with np.load(slice_path, allow_pickle=True) as s:
        slice_xy = np.asarray(s["coords"], dtype=float)
        slice_mat = np.asarray(s["ratio_feature_01"] if "ratio_feature_01" in s.files else s["ratio"], dtype=float)
        if slice_mat.ndim == 1:
            slice_mat = slice_mat[:, None]
        slice_color = np.nanmean(np.clip(slice_mat, 0.0, 1.0), axis=1)

    try:
        X_aligned, final_tf = _similarity_procrustes(X[:, :obs_dim], prior_ord[:, :obs_dim])
        final_note = f"final xy aligned to lifted prior, s={float(final_tf['s']):.3g}"
    except Exception:
        X_aligned = X[:, :obs_dim]
        final_note = "final xy shown without Procrustes"

    anchors = None
    anchor_exact_xy = None
    anchor_path = run_dir / "coarse_anchor_coords_registered.npy"
    if anchor_path.exists():
        try:
            anchors = np.asarray(np.load(anchor_path), dtype=float)
            if anchors.ndim == 2:
                anchors = anchors[:, :obs_dim]
                anchor_exact_xy, anchor_witness_report = _load_anchor_exact_slice_witness(run_dir, slice_xy, obs_dim=obs_dim)
            else:
                anchor_witness_report = {"available": False, "reason": "anchor_array_not_2d"}
        except Exception as exc:
            anchor_witness_report = {"available": False, "reason": "anchor_load_failed", "error": str(exc)}
    else:
        anchor_witness_report = {"available": False, "reason": "coarse_anchor_coords_registered.npy_not_found"}

    report: dict[str, Any] = {
        "layout_version": 1,
        "obs_dim": int(obs_dim),
        "slice_node_count": int(slice_xy.shape[0]),
        "fine_node_count_in_plot": int(X.shape[0]),
        "prior_node_count": int(prior_ord.shape[0]),
        "final_to_lifted_prior_xy": _similarity_report_2d(X[:, :obs_dim], prior_ord[:, :obs_dim]),
        "lifted_prior_to_slice_ordered_xy": _similarity_report_2d(prior_ord[:, :obs_dim], slice_xy[:, :obs_dim]),
        "fine_to_coarse_lift": lift_report,
        "anchor_exact_slice_witness": anchor_witness_report,
    }
    if anchors is not None:
        report["registered_anchor_to_exact_slice_xy"] = (
            _similarity_report_2d(anchors[:, :obs_dim], anchor_exact_xy[:, :obs_dim])
            if anchor_exact_xy is not None
            else {"available": False, "reason": "exact_slice_witness_unavailable"}
        )
    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    except Exception:
        pass

    rng = np.random.default_rng(0)
    pick = np.arange(X_aligned.shape[0])
    if pick.size > int(max_points):
        pick = rng.choice(pick, size=int(max_points), replace=False)
    pick_slice = np.arange(slice_xy.shape[0])
    if pick_slice.size > int(max_points):
        pick_slice = rng.choice(pick_slice, size=int(max_points), replace=False)

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 8.0))
    ax.scatter(slice_xy[pick_slice, 0], slice_xy[pick_slice, 1], c=slice_color[pick_slice], cmap="viridis", s=2, alpha=0.22, linewidths=0, label="raw slice xy")
    ax.scatter(prior_ord[pick, 0], prior_ord[pick, 1], c="black", marker="x", s=6, alpha=0.18, label="lifted centroid prior xy")
    ax.scatter(X_aligned[pick, 0], X_aligned[pick, 1], c="tab:red", s=3, alpha=0.35, linewidths=0, label="final fine GSE xy")
    if anchors is not None:
        ax.scatter(anchors[:, 0], anchors[:, 1], c="black", marker="^", s=28, alpha=0.90, label="registered coarse anchors xy")
    if anchor_exact_xy is not None:
        ax.scatter(anchor_exact_xy[:, 0], anchor_exact_xy[:, 1], facecolors="none", edgecolors="tab:blue", marker="o", s=38, alpha=0.85, label="exact selected slice rows")
    ax.set_title("register_zf top-down shared-frame check\n" + final_note)
    ax.set_xlabel("raw slice / registered x")
    ax.set_ylabel("raw slice / registered y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", frameon=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out_path

def _plot_register_zf_3d_alignment_scatter(
    run_dir: Path | str,
    match_dir: Path | str,
    *,
    gse_name: str = "GSEoutput.txt",
    max_points: int = 12000,
    force: bool = False,
) -> Path | None:
    """3D check: slice plane, lifted centroid prior, registered coarse anchors, final fine GSE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _mpl_set_register_zf_plot_fonts(base_size=11.0)

    run_dir = Path(run_dir)
    match_dir = Path(match_dir)
    out_path = match_dir / "register_zf_alignment_3d_scatter.png"
    if out_path.exists() and not force:
        return out_path

    gse_path = run_dir / str(gse_name)
    slice_path = match_dir / "slice_smoothed_ratio_fields.npz"

    if not gse_path.exists() or not slice_path.exists():
        return None

    idx, coords = _load_gse_output_txt(gse_path)
    X = _pad_or_truncate_to_3d(coords)

    prior_index = np.asarray(idx, dtype=np.int64)
    prior, obs_dim_loaded, _lift_report = _load_register_zf_fine_prior_for_run(run_dir)
    if prior is None or obs_dim_loaded is None:
        return None
    if prior.ndim != 2 or prior.shape[0] <= int(prior_index.max()):
        return None

    prior_ord = _pad_or_truncate_to_3d(prior[prior_index])
    obs_dim = max(1, min(2, int(obs_dim_loaded), X.shape[1], prior_ord.shape[1]))

    # Align only the observed slice block.  A full 3D Procrustes transform would
    # rotate latent-z into the observed plane and can hide failures in the graph-only axis.
    try:
        xy_aligned, tf = _similarity_procrustes(X[:, :obs_dim], prior_ord[:, :obs_dim])
        aligned_note = f"xy Procrustes to lifted slice prior (s={float(tf['s']):.3g})"
    except Exception:
        xy_aligned = X[:, :obs_dim]
        aligned_note = "final xy shown without Procrustes alignment"

    X_plot = np.zeros((X.shape[0], 3), dtype=float)
    X_plot[:, :obs_dim] = xy_aligned
    if X.shape[1] >= 3:
        z = np.asarray(X[:, 2], dtype=float)
        prior_z = np.asarray(prior_ord[:, 2], dtype=float)
        z_sd = float(np.nanstd(z))
        prior_z_sd = float(np.nanstd(prior_z))
        X_plot[:, 2] = (
            (z - float(np.nanmean(z)))
            / (z_sd if z_sd > 1e-12 else 1.0)
            * (prior_z_sd if prior_z_sd > 1e-12 else 1.0)
            + float(np.nanmean(prior_z))
        )
    else:
        X_plot[:, 2] = prior_ord[:, 2] if prior_ord.shape[1] >= 3 else 0.0

    with np.load(slice_path, allow_pickle=True) as s:
        slice_xy = np.asarray(s["coords"], dtype=float)
        slice_mat = np.asarray(
            s["ratio_feature_01"] if "ratio_feature_01" in s.files else s["ratio"],
            dtype=float,
        )
        if slice_mat.ndim == 1:
            slice_mat = slice_mat[:, None]
        slice_color = np.nanmean(np.clip(slice_mat, 0.0, 1.0), axis=1)

    rng = np.random.default_rng(0)
    pick = np.arange(X_plot.shape[0])
    if pick.size > int(max_points):
        pick = rng.choice(pick, size=int(max_points), replace=False)

    z_floor = float(np.nanquantile(X_plot[:, 2], 0.03)) if X_plot.size else 0.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        slice_xy[:, 0],
        slice_xy[:, 1],
        np.full(slice_xy.shape[0], z_floor),
        c=slice_color,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=2,
        alpha=0.25,
        linewidths=0,
        label="slice field on xy plane",
    )

    ax.scatter(
        prior_ord[pick, 0],
        prior_ord[pick, 1],
        prior_ord[pick, 2],
        c="black",
        marker="x",
        s=8,
        alpha=0.25,
        label="lifted centroid prior",
    )

    ax.scatter(
        X_plot[pick, 0],
        X_plot[pick, 1],
        X_plot[pick, 2],
        c=X_plot[pick, 2],
        cmap="coolwarm",
        s=3,
        alpha=0.45,
        linewidths=0,
        label="final fine GSE",
    )

    anchor_path = run_dir / "coarse_anchor_coords_registered.npy"
    if anchor_path.exists():
        try:
            anchors = _pad_or_truncate_to_3d(np.load(anchor_path))
            ax.scatter(
                anchors[:, 0],
                anchors[:, 1],
                anchors[:, 2],
                c="black",
                marker="^",
                s=24,
                alpha=0.85,
                label="registered coarse anchors",
            )
        except Exception:
            pass

    ax.set_title("Coarsen-and-align 3D check\n" + aligned_note)
    ax.set_xlabel("registered / aligned x")
    ax.set_ylabel("registered / aligned y")
    ax.set_zlabel("graph latent z / scaled final z")
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass
    ax.legend(loc="upper left")

    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_register_zf_ensemble_uncertainty(match_dir: Path, *, point_size: float, force: bool = False) -> Path | None:
    ens_path = match_dir / "aggregated_nodes_slice_mapped_coords_ensemble.npz"
    if not ens_path.exists():
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _mpl_set_register_zf_plot_fonts(base_size=11.0)
    out_path = match_dir / "register_zf_ensemble_uncertainty.png"
    if out_path.exists() and not force:
        return out_path

    with np.load(ens_path, allow_pickle=False) as z:
        coords_stack = np.asarray(z["coords_final"], dtype=float)
        a_to_slice_stack = np.asarray(z["a_to_slice"], dtype=int) if "a_to_slice" in z.files else None
        obj = np.asarray(z["objective_base_float"], dtype=float) if "objective_base_float" in z.files else np.arange(coords_stack.shape[0], dtype=float)
    if coords_stack.ndim != 3 or coords_stack.shape[0] < 1:
        return None

    mean_coords = np.mean(coords_stack, axis=0)
    coord_sd = np.sqrt(np.sum(np.var(coords_stack, axis=0), axis=1))
    unique_counts = None
    if a_to_slice_stack is not None and a_to_slice_stack.ndim == 2:
        sorted_assign = np.sort(a_to_slice_stack, axis=0)
        unique_counts = 1 + np.sum(sorted_assign[1:, :] != sorted_assign[:-1, :], axis=0)

    slice_coords = None
    maps_path = match_dir / "slice_assigned_aggregated_feature_maps.npz"
    if maps_path.exists():
        try:
            with np.load(maps_path, allow_pickle=True) as maps:
                slice_coords = np.asarray(maps["coords"], dtype=float)
        except Exception:
            slice_coords = None

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    if slice_coords is not None:
        for ax in axes.ravel()[:2]:
            ax.scatter(slice_coords[:, 0], slice_coords[:, 1], c="lightgray", s=max(0.5, float(point_size) * 0.35), linewidths=0, alpha=0.25)

    sd_abs = _robust_abs_limit(coord_sd, q=0.99)
    sc0 = axes[0, 0].scatter(mean_coords[:, 0], mean_coords[:, 1], c=coord_sd, s=max(1.0, float(point_size) * 0.8), cmap="magma", vmin=0.0, vmax=sd_abs, linewidths=0)
    axes[0, 0].set_title("Ensemble coordinate spread\nmean assigned slice coord colored by SD")
    axes[0, 0].set_xlabel("slice x")
    axes[0, 0].set_ylabel("slice y")
    axes[0, 0].set_aspect("equal")
    fig.colorbar(sc0, ax=axes[0, 0], shrink=0.85, label="coordinate SD")

    if unique_counts is not None:
        vmax = max(1.0, float(np.quantile(unique_counts, 0.99)))
        sc1 = axes[0, 1].scatter(mean_coords[:, 0], mean_coords[:, 1], c=unique_counts, s=max(1.0, float(point_size) * 0.8), cmap="viridis", vmin=1.0, vmax=vmax, linewidths=0)
        fig.colorbar(sc1, ax=axes[0, 1], shrink=0.85, label="unique slice assignments")
    else:
        axes[0, 1].scatter(mean_coords[:, 0], mean_coords[:, 1], c=coord_sd, s=max(1.0, float(point_size) * 0.8), cmap="magma", linewidths=0)
    axes[0, 1].set_title("Assignment degeneracy\nunique slice ids per aggregated node")
    axes[0, 1].set_xlabel("slice x")
    axes[0, 1].set_ylabel("slice y")
    axes[0, 1].set_aspect("equal")

    axes[1, 0].hist(coord_sd[np.isfinite(coord_sd)], bins=40)
    axes[1, 0].set_title("Distribution of ensemble coordinate SD")
    axes[1, 0].set_xlabel("coordinate SD")
    axes[1, 0].set_ylabel("aggregated nodes")

    axes[1, 1].plot(np.arange(obj.size), obj, marker="o", linewidth=1.2)
    axes[1, 1].set_title("Accepted ensemble objective values")
    axes[1, 1].set_xlabel("accepted ensemble member")
    axes[1, 1].set_ylabel("base float objective")
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle(f"register_zf ensemble uncertainty (accepted={coords_stack.shape[0]})", y=1.02)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _collect_register_zf_alignment_summary(match_dir: Path) -> dict[str, Any]:
    match_dir = Path(match_dir)
    out: dict[str, Any] = {
        "register_zf_match_dir": str(match_dir),
        "register_zf_alignment_available": False,
    }
    if not all((match_dir / name).exists() for name in _REGZF_ALIGNMENT_REQUIRED):
        return out
    try:
        mapped = np.load(match_dir / "aggregated_nodes_slice_mapped_coords.npz", allow_pickle=True)
        moved = np.asarray(mapped["moved_mask"]).astype(bool) if "moved_mask" in mapped.files else np.zeros(0, dtype=bool)
        move_norm = np.asarray(mapped["move_distance_normalized"], dtype=float) if "move_distance_normalized" in mapped.files else np.zeros(moved.shape[0])
        out.update({
            "register_zf_alignment_available": True,
            "regzf_fraction_reassigned_by_graph_refinement": float(np.mean(moved)) if moved.size else float("nan"),
            "regzf_median_move_distance_normalized": float(np.nanmedian(move_norm[moved])) if moved.size and np.any(moved) else 0.0,
            "regzf_p95_move_distance_normalized": float(np.nanquantile(move_norm[moved], 0.95)) if moved.size and np.any(moved) else 0.0,
            "regzf_max_move_distance_normalized": float(np.nanmax(move_norm[moved])) if moved.size and np.any(moved) else 0.0,
        })
    except Exception:
        pass

    try:
        maps = np.load(match_dir / "slice_assigned_aggregated_feature_maps.npz", allow_pickle=True)
        slice_data = np.load(match_dir / "slice_smoothed_ratio_fields.npz", allow_pickle=True)
        slice_vals_all = np.asarray(slice_data["ratio_feature_01"] if "ratio_feature_01" in slice_data.files else slice_data["ratio"], dtype=float)
        if slice_vals_all.ndim == 1:
            slice_vals_all = slice_vals_all[:, None]
        support_all = np.asarray(slice_data["support"], dtype=float) if "support" in slice_data.files else np.ones_like(slice_vals_all)
        if support_all.ndim == 1:
            support_all = support_all[:, None]
        base_all = np.asarray(maps["feature_mean_base"], dtype=float)
        final_all = np.asarray(maps["feature_mean_final"], dtype=float)
        if base_all.ndim == 1:
            base_all = base_all[:, None]
        if final_all.ndim == 1:
            final_all = final_all[:, None]
        n_chan = min(slice_vals_all.shape[1], base_all.shape[1], final_all.shape[1])
        base_mae = []
        final_mae = []
        base_rho = []
        final_rho = []
        for j in range(n_chan):
            raw_slice = slice_vals_all[:, j]
            raw_base = base_all[:, j]
            raw_final = final_all[:, j]
            support = support_all[:, j] if j < support_all.shape[1] else np.ones(raw_slice.shape[0])
            mask = np.isfinite(raw_slice) & np.isfinite(raw_base) & np.isfinite(raw_final) & (support > 1e-10)
            if int(mask.sum()) < 2:
                mask = np.isfinite(raw_slice) & np.isfinite(raw_base) & np.isfinite(raw_final)
            if int(mask.sum()) < 2:
                continue
            s_rank = _rank_rescale_over_mask(raw_slice, mask)
            b_rank = _rank_rescale_over_mask(raw_base, mask)
            f_rank = _rank_rescale_over_mask(raw_final, mask)
            base_mae.append(float(np.mean(np.abs(b_rank[mask] - s_rank[mask]))))
            final_mae.append(float(np.mean(np.abs(f_rank[mask] - s_rank[mask]))))
            base_rho.append(_safe_spearman(b_rank[mask], s_rank[mask]))
            final_rho.append(_safe_spearman(f_rank[mask], s_rank[mask]))
        if n_chan > 0:
            out.update({
                "regzf_n_feature_pairs": int(n_chan),
                "regzf_base_postrank_mae_mean": float(np.nanmean(base_mae)) if base_mae else float("nan"),
                "regzf_final_postrank_mae_mean": float(np.nanmean(final_mae)) if final_mae else float("nan"),
                "regzf_final_minus_base_postrank_mae_mean": float(np.nanmean(final_mae) - np.nanmean(base_mae)) if base_mae and final_mae else float("nan"),
                "regzf_base_postrank_spearman_mean": float(np.nanmean(base_rho)) if base_rho else float("nan"),
                "regzf_final_postrank_spearman_mean": float(np.nanmean(final_rho)) if final_rho else float("nan"),
                "regzf_final_minus_base_postrank_spearman_mean": float(np.nanmean(final_rho) - np.nanmean(base_rho)) if base_rho and final_rho else float("nan"),
            })
    except Exception:
        pass

    xy_report_path = match_dir / "register_zf_xy_frame_report.json"
    if xy_report_path.exists():
        try:
            with open(xy_report_path, "r", encoding="utf-8") as fh:
                xy_report = json.load(fh)
            out["regzf_xy_frame_report_json"] = str(xy_report_path)
            anchor_rep = xy_report.get("registered_anchor_to_exact_slice_xy", {}) if isinstance(xy_report, dict) else {}
            if isinstance(anchor_rep, dict) and anchor_rep.get("available"):
                out["regzf_anchor_exact_slice_identity_max_abs_error"] = float(anchor_rep.get("max_abs_identity_error", float("nan")))
                out["regzf_anchor_exact_slice_rotation_angle_degrees"] = float(anchor_rep.get("rotation_angle_degrees", float("nan")))
                out["regzf_anchor_exact_slice_normalized_rms"] = float(anchor_rep.get("normalized_rms_residual", float("nan")))
            prior_rep = xy_report.get("lifted_prior_to_slice_ordered_xy", {}) if isinstance(xy_report, dict) else {}
            if isinstance(prior_rep, dict) and prior_rep.get("available"):
                out["regzf_lifted_prior_to_slice_ordered_rotation_angle_degrees"] = float(prior_rep.get("rotation_angle_degrees", float("nan")))
                out["regzf_lifted_prior_to_slice_ordered_normalized_rms"] = float(prior_rep.get("normalized_rms_residual", float("nan")))
        except Exception:
            pass

    ens_path = match_dir / "aggregated_nodes_slice_mapped_coords_ensemble.npz"
    if ens_path.exists():
        try:
            with np.load(ens_path, allow_pickle=False) as z:
                coords_stack = np.asarray(z["coords_final"], dtype=float)
                out["regzf_ensemble_accepted_size"] = int(coords_stack.shape[0]) if coords_stack.ndim == 3 else 0
                if coords_stack.ndim == 3 and coords_stack.shape[0] > 0:
                    coord_sd = np.sqrt(np.sum(np.var(coords_stack, axis=0), axis=1))
                    out.update({
                        "regzf_ensemble_coord_sd_mean": float(np.nanmean(coord_sd)),
                        "regzf_ensemble_coord_sd_median": float(np.nanmedian(coord_sd)),
                        "regzf_ensemble_coord_sd_p95": float(np.nanquantile(coord_sd, 0.95)),
                        "regzf_ensemble_coord_sd_max": float(np.nanmax(coord_sd)),
                    })
                if "a_to_slice" in z.files:
                    a = np.asarray(z["a_to_slice"], dtype=int)
                    if a.ndim == 2 and a.shape[0] > 0:
                        unique_counts = 1 + np.sum(np.sort(a, axis=0)[1:, :] != np.sort(a, axis=0)[:-1, :], axis=0)
                        out.update({
                            "regzf_ensemble_assignment_variable_fraction": float(np.mean(unique_counts > 1)),
                            "regzf_ensemble_unique_slices_per_agg_mean": float(np.mean(unique_counts)),
                            "regzf_ensemble_unique_slices_per_agg_p95": float(np.quantile(unique_counts, 0.95)),
                        })
        except Exception:
            pass
    return out


def render_register_zf_alignment_diagnostics_for_run(run_dir: Path | str, args: argparse.Namespace, *, force: bool | None = None) -> dict[str, Any] | None:
    """Write per-run register_zf alignment figures next to match_result_<ZF_FLAG>."""
    if not _optimops_register_zf_requested(args):
        return None
    if not bool(getattr(args, "viz_register_zf_alignment", True)):
        return None
    zf_flag = _optimops_register_zf_flag(args)
    if not zf_flag:
        return None
    match_dir = _register_zf_match_dir_for_run(run_dir, zf_flag)
    if not match_dir.exists():
        return None
    if not all((match_dir / name).exists() for name in _REGZF_ALIGNMENT_REQUIRED):
        return None
    force_eff = bool(getattr(args, "optimops_force", False) if force is None else force)
    point_size = float(getattr(args, "viz_register_zf_alignment_point_size", 7.0))
    max_pairs = int(max(0, getattr(args, "viz_register_zf_alignment_max_pairs", 2)))
    robust_q = float(getattr(args, "viz_register_zf_alignment_robust_quantile", 0.995))
    rank_neutral = 0.5

    written: list[str] = []
    try:
        with np.load(match_dir / "slice_assigned_aggregated_feature_maps.npz", allow_pickle=True) as maps:
            n_pairs = int(np.asarray(maps["feature_mean_final"]).shape[1]) if np.asarray(maps["feature_mean_final"]).ndim > 1 else 1
        for pair_idx in range(min(max_pairs, n_pairs)):
            outp = _plot_register_zf_pair_alignment(
                match_dir,
                pair_idx=pair_idx,
                point_size=point_size,
                robust_q=robust_q,
                rank_neutral=rank_neutral,
                force=force_eff,
            )
            if outp is not None:
                written.append(str(outp))

            sup = _plot_register_zf_slice_superposition(
                match_dir,
                pair_idx=pair_idx,
                point_size=point_size,
                rank_neutral=rank_neutral,
                force=force_eff,
            )
            if sup is not None:
                written.append(str(sup))

        xy_frame = _plot_register_zf_xy_frame_diagnostic(
            Path(run_dir),
            match_dir,
            gse_name=str(getattr(args, "optimops_output_name", "GSEoutput.txt")),
            force=force_eff,
        )
        if xy_frame is not None:
            written.append(str(xy_frame))

        scatter3d = _plot_register_zf_3d_alignment_scatter(
            Path(run_dir),
            match_dir,
            gse_name=str(getattr(args, "optimops_output_name", "GSEoutput.txt")),
            force=force_eff,
        )
        if scatter3d is not None:
            written.append(str(scatter3d))

        ens_out = _plot_register_zf_ensemble_uncertainty(match_dir, point_size=point_size, force=force_eff)
        if ens_out is not None:
            written.append(str(ens_out))
    except Exception as exc:
        print(f"[WARN] register_zf alignment visualization failed for {match_dir}: {exc}")

    summary = _collect_register_zf_alignment_summary(match_dir)
    summary["register_zf_alignment_plot_files"] = written
    try:
        with open(match_dir / "register_zf_alignment_visualization_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
    except Exception:
        pass
    return summary


def _load_graph_for_dir(final_dir: Path | str,
                        *,
                        graph_name: str = "link_assoc_reindexed.npz") -> tuple[csr_matrix | None, Path | None]:
    """Best-effort load of the graph file associated with a dataset directory."""
    final_dir = Path(final_dir)
    graph_path = final_dir / str(graph_name)
    if not graph_path.exists():
        gp = _search_upwards_for_file(final_dir, str(graph_name), max_up=4)
        if gp is not None:
            graph_path = gp
    if not graph_path.exists():
        return None, None

    try:
        from scipy.sparse import load_npz
        return load_npz(str(graph_path)).tocsr(), graph_path
    except Exception:
        return None, graph_path



def _compute_graph_umap_embedding_from_adjacency(A: csr_matrix,
                                                *,
                                                pca_dim: int,
                                                n_neighbors: int,
                                                min_dist: float,
                                                metric: str,
                                                seed: int) -> np.ndarray | None:
    """Compute the graph-only spectral-UMAP embedding used for the 3D UMAP views."""
    # Optional dependency.
    try:
        import umap  # type: ignore
    except Exception as e:
        warnings.warn(f"umap-learn not available; skipping graph UMAP embedding: {e}")
        return None

    n_nodes = int(A.shape[0])
    if n_nodes < 4:
        return None

    A_sym = _symmetrize_graph(A).tocsr()

    # --- Spectral stage: normalized-adjacency eigendecomposition (via eigsh)
    # `pca_dim` is the number of algebraic eigenvectors retained before the UMAP step.
    try:
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
        from sklearn.preprocessing import StandardScaler

        n_feat = int(max(2, min(int(pca_dim), n_nodes - 2)))

        deg = np.asarray(A_sym.sum(axis=1)).reshape(-1)
        deg = np.maximum(deg, 1e-12)
        inv_sqrt = 1.0 / np.sqrt(deg)
        D = sp.diags(inv_sqrt)

        M = (D @ A_sym @ D).tocsr()

        # Compute k+1 eigenvectors and drop the trivial top eigenvector (~constant).
        k = int(min(n_feat + 1, n_nodes - 1))
        rng = np.random.default_rng(int(seed))
        v0 = rng.standard_normal(n_nodes)

        vals, vecs = eigsh(M, k=k, which="LA", v0=v0, tol=1e-3)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        if vecs.shape[1] > 1:
            vecs = vecs[:, 1:]
            vals = vals[1:]

        vecs = vecs[:, :min(n_feat, vecs.shape[1])]
        if vecs.shape[1] < 2:
            return None

        # Standardize features before UMAP.
        Xp = StandardScaler().fit_transform(np.asarray(vecs, dtype=float))
    except Exception as e:
        warnings.warn(f"spectral eigendecomposition failed; skipping graph UMAP embedding: {e}")
        return None

    # --- UMAP to 3D
    try:
        um = umap.UMAP(
            n_components=3,
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            metric=str(metric),
            init="spectral",
            random_state=int(seed),
        )
        return np.asarray(um.fit_transform(Xp), dtype=float)
    except Exception as e:
        warnings.warn(f"UMAP fit failed; skipping graph UMAP embedding: {e}")
        return None



def _compute_or_load_graph_umap_embedding(final_dir: Path | str,
                                          *,
                                          graph_name: str = "link_assoc_reindexed.npz",
                                          pca_dim: int = 15,
                                          n_neighbors: int = 30,
                                          min_dist: float = 0.1,
                                          metric: str = "cosine",
                                          seed: int = 0,
                                          force: bool = False) -> tuple[np.ndarray | None, csr_matrix | None]:
    """Load a cached graph-only UMAP embedding for a dataset directory, or compute it."""
    final_dir = Path(final_dir)
    A, _graph_path = _load_graph_for_dir(final_dir, graph_name=str(graph_name))
    if A is None:
        return None, None

    stem = _graph_umap_cache_stem(
        pca_dim=int(pca_dim),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
    )
    cache_path = final_dir / f"{stem}.npy"

    if cache_path.exists() and not force:
        try:
            X = np.asarray(np.load(str(cache_path))).astype(float, copy=False)
            if X.ndim == 2 and X.shape[0] == int(A.shape[0]):
                return X, A
        except Exception:
            pass

    X = _compute_graph_umap_embedding_from_adjacency(
        A,
        pca_dim=int(pca_dim),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        seed=int(seed),
    )
    if X is None:
        return None, A

    try:
        np.save(str(cache_path), np.asarray(X, dtype=np.float32))
    except Exception:
        pass
    return X, A


def _fit_hdbscan_labels_like_clusterplot(
    X: np.ndarray,
    adjacency: csr_matrix | np.ndarray,
    *,
    min_cluster_size: int = _UMAP_HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = _UMAP_HDBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    """Run the same HDBSCAN + graph-curation route used by clusterplot.execute_clusters_hdbscan."""
    try:
        from clusterplot import execute_clusters_hdbscan
    except Exception as e:
        raise ImportError(
            "Could not import clusterplot.execute_clusters_hdbscan"
        ) from e

    adj = adjacency if isinstance(adjacency, csr_matrix) else csr_matrix(adjacency)
    labels = execute_clusters_hdbscan(
        Xpts=np.asarray(X, dtype=float),
        adjacency=adj,
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
    )
    return np.asarray(labels, dtype=np.int32, copy=False)


def _compute_graph_umap_hdbscan_nmi_for_dir(final_dir: Path | str,
                                            *,
                                            graph_name: str = "link_assoc_reindexed.npz",
                                            pca_dim: int = _UMAP_BENCHMARK_PCA_DIM,
                                            n_neighbors: int = _UMAP_BENCHMARK_N_NEIGHBORS,
                                            min_dist: float = _UMAP_BENCHMARK_MIN_DIST,
                                            metric: str = _UMAP_BENCHMARK_METRIC,
                                            seed: int = 0,
                                            hdbscan_min_cluster_size: int = _UMAP_HDBSCAN_MIN_CLUSTER_SIZE,
                                            hdbscan_min_samples: int = _UMAP_HDBSCAN_MIN_SAMPLES,
                                            force: bool = False) -> float:
    """Cluster the graph-only UMAP embedding with clusterplot's HDBSCAN route and return NMI."""
    from sklearn.metrics import normalized_mutual_info_score

    final_dir = Path(final_dir)
    stem = _graph_umap_cache_stem(
        pca_dim=int(pca_dim),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
    )
    hdbscan_tag = f"hdbscan_clusterplot_v{int(_UMAP_HDBSCAN_CACHE_VERSION)}_mcs{int(hdbscan_min_cluster_size)}_ms{int(hdbscan_min_samples)}"
    nmi_path = final_dir / f"{stem}_{hdbscan_tag}_nmi.json"
    labels_path = final_dir / f"{stem}_{hdbscan_tag}_labels.npy"

    X, A = _compute_or_load_graph_umap_embedding(
        final_dir,
        graph_name=str(graph_name),
        pca_dim=int(pca_dim),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        seed=int(seed),
        force=bool(force),
    )
    if X is None or A is None:
        return float("nan")

    n_nodes = int(A.shape[0])
    _ref_pos, keep = _load_positions_scaled_for_dir(final_dir, n_expected=n_nodes)
    true_cell = _load_true_cells_for_dir(final_dir, n_expected=n_nodes, keep_nodes_global=keep)
    if true_cell is None or true_cell.shape[0] < n_nodes:
        warnings.warn(f"Ground-truth cell ids unavailable for graph UMAP/HDBSCAN NMI in {final_dir}")
        return float("nan")

    labels = None
    if labels_path.exists() and not force:
        try:
            labels = np.asarray(np.load(str(labels_path))).astype(np.int32, copy=False)
            if labels.ndim != 1 or labels.shape[0] != n_nodes:
                labels = None
        except Exception:
            labels = None

    if labels is None:
        try:
            labels = _fit_hdbscan_labels_like_clusterplot(
                X,
                A,
                min_cluster_size=int(hdbscan_min_cluster_size),
                min_samples=int(hdbscan_min_samples),
            )
        except ImportError as e:
            warnings.warn(f"Skipping graph UMAP/HDBSCAN NMI because HDBSCAN is unavailable: {e}")
            return float("nan")
        try:
            np.save(str(labels_path), labels.astype(np.int32, copy=False))
        except Exception:
            pass

    nmi = float(
        normalized_mutual_info_score(
            np.asarray(true_cell, dtype=np.int32),
            np.asarray(labels, dtype=np.int32),
            average_method="arithmetic",
        )
    )

    try:
        payload = {
            "method": _UMAP_HDBSCAN_METHOD_KEY,
            "nmi": nmi,
            "pca_dim": int(pca_dim),
            "n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
            "metric": str(metric),
            "hdbscan_min_cluster_size": int(hdbscan_min_cluster_size),
            "hdbscan_min_samples": int(hdbscan_min_samples),
            "n_nodes": int(n_nodes),
            "n_clusters_total": int(np.unique(labels).size),
            "n_clusters_nonnoise": int(np.sum(np.unique(labels) >= 0)),
        }
        with open(nmi_path, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    except Exception:
        pass

    return nmi



def _process_neighborhood_like_upstream(gse_coords: np.ndarray, pos_coords: np.ndarray) -> np.ndarray:
    """Match upstream_analysis.process_neighborhood() for local Procrustes alignment."""
    gse_center = np.mean(gse_coords, axis=0)
    pos_center = np.mean(pos_coords, axis=0)
    gse_centered = gse_coords - gse_center
    pos_centered = pos_coords - pos_center

    gse_var = float(np.sum(np.var(gse_centered, axis=0)))
    pos_var = float(np.sum(np.var(pos_centered, axis=0)))
    scale_factor = math.sqrt(pos_var / (gse_var + 1e-20)) if pos_var > 0.0 else 1.0

    gse_scaled = gse_centered * scale_factor
    R, _ = orthogonal_procrustes(gse_scaled, pos_centered)
    return np.dot(gse_scaled, R) + pos_center


def _calculate_rmsd_like_upstream(points1: np.ndarray, points2: np.ndarray) -> float:
    """Match upstream_analysis.calculate_rmsd()."""
    diff = np.asarray(points1, dtype=float) - np.asarray(points2, dtype=float)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _default_decoherence_sizes(n_nodes: int) -> list[int]:
    """Geometric neighborhood sizes for decoherence analysis."""
    if int(n_nodes) < 2:
        return []
    base = [4, 8, 16, 32, 64, 100, 200, 400, 800, 1600, 3200, 6400]
    out: list[int] = []
    for k in base:
        kk = int(min(int(n_nodes), int(k)))
        if kk >= 2 and (not out or kk != out[-1]):
            out.append(kk)
    if not out:
        out = [int(min(int(n_nodes), 2))]
    if out[-1] < int(n_nodes):
        tail = int(n_nodes)
        if tail >= 2 and tail != out[-1]:
            out.append(tail)
    return out


def _estimate_threshold_crossing_radius(mean_spread: np.ndarray,
                                        mean_rmsd: np.ndarray,
                                        threshold: float) -> float:
    """Estimate the first radius where the average decoherence error exceeds a threshold."""
    spread = np.asarray(mean_spread, dtype=float)
    rmsd = np.asarray(mean_rmsd, dtype=float)
    good = np.isfinite(spread) & np.isfinite(rmsd)
    if not np.any(good):
        return float("nan")

    spread = spread[good]
    rmsd = rmsd[good]
    if spread.size == 0:
        return float("nan")

    cross = np.flatnonzero(rmsd > float(threshold))
    if cross.size == 0:
        return float("nan")

    i1 = int(cross[0])
    if i1 == 0:
        return float(spread[0])

    i0 = i1 - 1
    x0 = float(spread[i0])
    x1 = float(spread[i1])
    y0 = float(rmsd[i0])
    y1 = float(rmsd[i1])
    if (not np.isfinite(y0)) or (not np.isfinite(y1)) or abs(y1 - y0) <= 1e-12:
        return float(x1)

    t = (float(threshold) - y0) / (y1 - y0)
    t = float(np.clip(t, 0.0, 1.0))
    return float(x0 + t * (x1 - x0))


def _compute_gse_decoherence_radius_for_dir(final_dir: Path | str,
                                            *,
                                            gse_name: str = "GSEoutput.txt",
                                            graph_name: str = "link_assoc_reindexed.npz",
                                            seed: int = 0,
                                            n_samples: int = 200,
                                            decoherence_sizes: list[int] | None = None,
                                            force: bool = False) -> tuple[float, dict[str, Any]]:
    """Estimate the GSE decoherence radius using upstream_analysis.analyze_neighborhoods logic."""
    final_dir = Path(final_dir)
    cache_path = final_dir / "GSE_embedding_decoherence_radius.json"
    if cache_path.exists() and not force:
        try:
            with open(cache_path, "r") as fh:
                payload = json.load(fh)
            radius = float(payload.get(_BENCHMARK_DECOHERENCE_RADIUS_KEY, float("nan")))
            return radius, payload
        except Exception:
            pass

    gse_path = final_dir / str(gse_name)
    if not gse_path.exists():
        payload = {
            _BENCHMARK_DECOHERENCE_RADIUS_KEY: float("nan"),
            "reason": f"missing {gse_name}",
        }
        return float("nan"), payload

    idx, coords = _load_gse_output_txt(gse_path)
    n_expected = int(idx.max() + 1) if idx.size > 0 else 0
    if n_expected < 2:
        payload = {
            _BENCHMARK_DECOHERENCE_RADIUS_KEY: float("nan"),
            "reason": "too few embedding nodes",
        }
        return float("nan"), payload

    ref_pos, _keep = _load_positions_scaled_for_dir(final_dir, n_expected=n_expected)
    if ref_pos is None or ref_pos.shape[0] < n_expected:
        payload = {
            _BENCHMARK_DECOHERENCE_RADIUS_KEY: float("nan"),
            "reason": "scaled node positions unavailable",
        }
        return float("nan"), payload

    pos_coords = _pad_or_truncate_to_3d(ref_pos[idx, :])
    gse_coords = _pad_or_truncate_to_3d(coords)
    if pos_coords.shape[0] != gse_coords.shape[0] or pos_coords.shape[0] < 2:
        payload = {
            _BENCHMARK_DECOHERENCE_RADIUS_KEY: float("nan"),
            "reason": "embedding/position alignment failed",
        }
        return float("nan"), payload

    A_counts, _ = _load_graph_for_dir(final_dir, graph_name=str(graph_name))
    if A_counts is None:
        avg_counts_per_node = float("nan")
        threshold = float("nan")
    else:
        counts_per_node = np.asarray(_symmetrize_graph(A_counts).sum(axis=1)).reshape(-1).astype(float, copy=False)
        avg_counts_per_node = float(np.mean(counts_per_node)) if counts_per_node.size > 0 else float("nan")
        threshold = 2.0 / math.sqrt(max(avg_counts_per_node, 1e-12)) if np.isfinite(avg_counts_per_node) else float("nan")

    n_nodes = int(pos_coords.shape[0])
    sizes = decoherence_sizes if decoherence_sizes is not None else _default_decoherence_sizes(n_nodes)
    sizes = [int(k) for k in sizes if 2 <= int(k) <= n_nodes]
    if not sizes:
        payload = {
            _BENCHMARK_DECOHERENCE_RADIUS_KEY: float("nan"),
            "reason": "no valid neighborhood sizes",
            "avg_total_counts_per_node": avg_counts_per_node,
            "threshold": threshold,
        }
        return float("nan"), payload

    rng = np.random.default_rng(int(seed))
    sample_count = int(max(1, min(int(n_samples), n_nodes)))
    mean_spreads: list[float] = []
    mean_rmsds: list[float] = []
    nn_model = NearestNeighbors(metric="euclidean")
    nn_model.fit(pos_coords)

    for k in sizes:
        k_eff = int(min(int(k), n_nodes))
        sample_idx = rng.choice(n_nodes, size=sample_count, replace=False)
        spreads: list[float] = []
        rmsds: list[float] = []
        for idx0 in sample_idx:
            _, nbr_idx = nn_model.kneighbors(pos_coords[int(idx0)].reshape(1, -1), n_neighbors=k_eff)
            nbr_idx = np.asarray(nbr_idx[0], dtype=np.int64)
            gse_neighborhood = gse_coords[nbr_idx]
            pos_neighborhood = pos_coords[nbr_idx]
            try:
                gse_transformed = _process_neighborhood_like_upstream(gse_neighborhood, pos_neighborhood)
            except Exception:
                continue
            rmsds.append(_calculate_rmsd_like_upstream(gse_transformed, pos_neighborhood))
            spreads.append(float(np.mean(np.linalg.norm(pos_neighborhood - pos_neighborhood[0], axis=1))))

        mean_spreads.append(float(np.mean(spreads)) if spreads else float("nan"))
        mean_rmsds.append(float(np.mean(rmsds)) if rmsds else float("nan"))

    radius = (
        _estimate_threshold_crossing_radius(np.asarray(mean_spreads), np.asarray(mean_rmsds), float(threshold))
        if np.isfinite(threshold) else float("nan")
    )

    payload = {
        _BENCHMARK_DECOHERENCE_RADIUS_KEY: float(radius),
        "avg_total_counts_per_node": float(avg_counts_per_node),
        "threshold": float(threshold),
        "decoherence_sizes": [int(k) for k in sizes],
        "mean_spread": [float(x) for x in mean_spreads],
        "mean_rmsd": [float(x) for x in mean_rmsds],
        "n_samples": int(sample_count),
        "graph_name": str(graph_name),
        "gse_name": str(gse_name),
    }
    try:
        with open(cache_path, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    except Exception:
        pass
    return float(radius), payload


def _compute_graph_umap_decoherence_radius_for_dir(final_dir: Path | str,
                                                   *,
                                                   graph_name: str = "link_assoc_reindexed.npz",
                                                   pca_dim: int = _UMAP_BENCHMARK_PCA_DIM,
                                                   n_neighbors: int = _UMAP_BENCHMARK_N_NEIGHBORS,
                                                   min_dist: float = _UMAP_BENCHMARK_MIN_DIST,
                                                   metric: str = _UMAP_BENCHMARK_METRIC,
                                                   umap_seed: int = 0,
                                                   seed: int = 0,
                                                   n_samples: int = 200,
                                                   decoherence_sizes: list[int] | None = None,
                                                   force: bool = False) -> tuple[float, dict[str, Any]]:
    """Estimate the graph-UMAP decoherence radius with the same local RMSD criterion used for GSE."""
    final_dir = Path(final_dir)
    stem = _graph_umap_cache_stem(
        pca_dim=int(pca_dim),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
    )
    cache_path = final_dir / f"{stem}_decoherence_radius.json"
    if cache_path.exists() and not force:
        try:
            with open(cache_path, "r") as fh:
                payload = json.load(fh)
            radius = float(payload.get(_BENCHMARK_DECOHERENCE_RADIUS_KEY, float("nan")))
            return radius, payload
        except Exception:
            pass

    X, A_counts = _compute_or_load_graph_umap_embedding(
        final_dir,
        graph_name=str(graph_name),
        pca_dim=int(pca_dim),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        seed=int(umap_seed),
        force=bool(force),
    )
    if X is None:
        payload = {
            _BENCHMARK_DECOHERENCE_RADIUS_KEY: float("nan"),
            "reason": "graph UMAP embedding unavailable",
            "graph_name": str(graph_name),
        }
        return float("nan"), payload

    n_expected = int(X.shape[0])
    if n_expected < 2:
        payload = {
            _BENCHMARK_DECOHERENCE_RADIUS_KEY: float("nan"),
            "reason": "too few embedding nodes",
            "graph_name": str(graph_name),
        }
        return float("nan"), payload

    ref_pos, _keep = _load_positions_scaled_for_dir(final_dir, n_expected=n_expected)
    if ref_pos is None or ref_pos.shape[0] < n_expected:
        payload = {
            _BENCHMARK_DECOHERENCE_RADIUS_KEY: float("nan"),
            "reason": "scaled node positions unavailable",
            "graph_name": str(graph_name),
        }
        return float("nan"), payload

    idx = np.arange(n_expected, dtype=np.int64)
    pos_coords = _pad_or_truncate_to_3d(ref_pos[idx, :])
    umap_coords = _pad_or_truncate_to_3d(np.asarray(X, dtype=float)[idx, :])
    if pos_coords.shape[0] != umap_coords.shape[0] or pos_coords.shape[0] < 2:
        payload = {
            _BENCHMARK_DECOHERENCE_RADIUS_KEY: float("nan"),
            "reason": "embedding/position alignment failed",
            "graph_name": str(graph_name),
        }
        return float("nan"), payload

    if A_counts is None:
        A_counts, _ = _load_graph_for_dir(final_dir, graph_name=str(graph_name))
    if A_counts is None:
        avg_counts_per_node = float("nan")
        threshold = float("nan")
    else:
        counts_per_node = np.asarray(_symmetrize_graph(A_counts).sum(axis=1)).reshape(-1).astype(float, copy=False)
        avg_counts_per_node = float(np.mean(counts_per_node)) if counts_per_node.size > 0 else float("nan")
        threshold = 2.0 / math.sqrt(max(avg_counts_per_node, 1e-12)) if np.isfinite(avg_counts_per_node) else float("nan")

    n_nodes = int(pos_coords.shape[0])
    sizes = decoherence_sizes if decoherence_sizes is not None else _default_decoherence_sizes(n_nodes)
    sizes = [int(k) for k in sizes if 2 <= int(k) <= n_nodes]
    if not sizes:
        payload = {
            _BENCHMARK_DECOHERENCE_RADIUS_KEY: float("nan"),
            "reason": "no valid neighborhood sizes",
            "avg_total_counts_per_node": avg_counts_per_node,
            "threshold": threshold,
            "graph_name": str(graph_name),
        }
        return float("nan"), payload

    rng = np.random.default_rng(int(seed))
    sample_count = int(max(1, min(int(n_samples), n_nodes)))
    mean_spreads: list[float] = []
    mean_rmsds: list[float] = []
    nn_model = NearestNeighbors(metric="euclidean")
    nn_model.fit(pos_coords)

    for k in sizes:
        k_eff = int(min(int(k), n_nodes))
        sample_idx = rng.choice(n_nodes, size=sample_count, replace=False)
        spreads: list[float] = []
        rmsds: list[float] = []
        for idx0 in sample_idx:
            _, nbr_idx = nn_model.kneighbors(pos_coords[int(idx0)].reshape(1, -1), n_neighbors=k_eff)
            nbr_idx = np.asarray(nbr_idx[0], dtype=np.int64)
            umap_neighborhood = umap_coords[nbr_idx]
            pos_neighborhood = pos_coords[nbr_idx]
            try:
                umap_transformed = _process_neighborhood_like_upstream(umap_neighborhood, pos_neighborhood)
            except Exception:
                continue
            rmsds.append(_calculate_rmsd_like_upstream(umap_transformed, pos_neighborhood))
            spreads.append(float(np.mean(np.linalg.norm(pos_neighborhood - pos_neighborhood[0], axis=1))))

        mean_spreads.append(float(np.mean(spreads)) if spreads else float("nan"))
        mean_rmsds.append(float(np.mean(rmsds)) if rmsds else float("nan"))

    radius = (
        _estimate_threshold_crossing_radius(np.asarray(mean_spreads), np.asarray(mean_rmsds), float(threshold))
        if np.isfinite(threshold) else float("nan")
    )

    payload = {
        _BENCHMARK_DECOHERENCE_RADIUS_KEY: float(radius),
        "avg_total_counts_per_node": float(avg_counts_per_node),
        "threshold": float(threshold),
        "decoherence_sizes": [int(k) for k in sizes],
        "mean_spread": [float(x) for x in mean_spreads],
        "mean_rmsd": [float(x) for x in mean_rmsds],
        "n_samples": int(sample_count),
        "graph_name": str(graph_name),
        "pca_dim": int(pca_dim),
        "umap_n_neighbors": int(n_neighbors),
        "umap_min_dist": float(min_dist),
        "umap_metric": str(metric),
        "umap_seed": int(umap_seed),
    }
    try:
        with open(cache_path, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    except Exception:
        pass
    return float(radius), payload



def render_graph_umap_ground_truth_scatter_3d(
    final_dir: Path | str,
    *,
    graph_name: str = "link_assoc_reindexed.npz",
    out_name: str = "UMAP_embedding_3d_ground_truth.png",
    pca_dim: int = 15,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
    seed: int = 0,
    max_edges: int = 35000,
    edge_seed: int | None = 0,
    force: bool = False,
    font_size: float = 12.5,
    cmap_name: str = "tab20",
    show_top_down: bool = False,
) -> Path | None:
    """Compute a 3D UMAP embedding from the graph alone, then align + color by ground truth.

    IMPORTANT (bipartite mitigation):
      The simulated UEI graph is bipartite by construction. A naïve SVD ("PCA") on the adjacency uses
      *singular values* and is therefore insensitive to the sign of eigenvalues; this tends to pull in the
      strong negative modes associated with bipartite oscillation.

      Here we instead build the *symmetric normalized adjacency*
          M = D^{-1/2} A D^{-1/2}
      and compute its largest *algebraic* eigenpairs (sparse eigendecomposition). This keeps the
      informative positive modes and avoids the -1 bipartite mode.

      These spectral features are then fed into UMAP.
    """
    final_dir = Path(final_dir)
    outpath = final_dir / str(out_name)
    if outpath.exists() and not force:
        return outpath

    X, A = _compute_or_load_graph_umap_embedding(
        final_dir,
        graph_name=str(graph_name),
        pca_dim=int(pca_dim),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        seed=int(seed),
        force=bool(force),
    )
    if X is None or A is None:
        return None

    n_nodes = int(A.shape[0])

    # Reference positions + ground truth labels (for alignment + coloring).
    ref_pos, keep = _load_positions_scaled_for_dir(final_dir, n_expected=n_nodes)
    true_cell = _load_true_cells_for_dir(final_dir, n_expected=n_nodes, keep_nodes_global=keep)
    if true_cell is None or true_cell.shape[0] < n_nodes:
        return None
    cell = np.asarray(true_cell, dtype=int)

    aligned_note = "Unaligned (no reference positions)"
    if ref_pos is not None and ref_pos.shape[0] >= n_nodes:
        try:
            X_aligned, tf = _similarity_procrustes(_pad_or_truncate_to_3d(X), _pad_or_truncate_to_3d(ref_pos))
            X = X_aligned
            aligned_note = f"Procrustes aligned (s={float(tf['s']):.3g})"
        except Exception:
            aligned_note = "Unaligned (Procrustes failed)"

    # Edge sampling (for context only).
    edges_i = edges_j = None
    try:
        edges_i, edges_j = _sample_edges_for_plot(A, max_edges=int(max_edges), seed=edge_seed)
    except Exception:
        edges_i = edges_j = None

    title = (
        f"UMAP (3D) from graph (norm-adj eigvecs→UMAP) colored by ground-truth cell id  |  "
        f"eigsh_components={int(pca_dim)}, n_neighbors={int(n_neighbors)}, min_dist={float(min_dist):.2g}, metric={metric}  |  {aligned_note}"
    )
    ref_bbox = _pad_or_truncate_to_3d(ref_pos) if ref_pos is not None else None
    _plot_cells_scatter_3d_common(
        X,
        cell,
        title=title,
        outpath=outpath,
        xlabel="x",
        ylabel="y",
        zlabel="z",
        ref_bbox=ref_bbox,
        edges_i=edges_i,
        edges_j=edges_j,
        font_size=float(font_size),
        cmap_name=str(cmap_name),
        show_top_down=bool(show_top_down),
    )
    return outpath


def maybe_render_optimops_outputs(final_dir: Path | str,
                                 *,
                                 gse_name: str = "GSEoutput.txt",
                                 force: bool = False,
                                 render_roc: bool = False,
                                 show_top_down: bool = False) -> None:
    """Best-effort: render embedding scatter + (if possible) a pairwise ROC curve."""
    final_dir = Path(final_dir)

    # 1) 3D embedding scatter (one image per label column when available)
    lab_path = final_dir / "cluster_labels.npy"
    n_cols = 1
    try:
        _labs = np.load(str(lab_path))
        _labs = np.asarray(_labs)
        if _labs.ndim == 1:
            n_cols = 1
        elif _labs.ndim == 2:
            n_cols = int(_labs.shape[1])
        else:
            n_cols = int(_labs.reshape(_labs.shape[0], -1).shape[1])
    except Exception:
        n_cols = 1

    meta_methods = None
    meta_path = final_dir / "cluster_labels_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as fh:
                meta = json.load(fh)
            methods0 = _optimops_cluster_method_names_from_meta(meta, int(n_cols))
            if methods0:
                meta_methods = [str(x) for x in methods0]
        except Exception:
            meta_methods = None

    if meta_methods is not None:
        col_specs = [(i, meta_methods[i]) for i in range(min(int(n_cols), len(meta_methods)))]
        if len(col_specs) == 0:
            col_specs = [(0, "hdbscan")]
    else:
        inferred_methods = list(_optimops_default_cluster_method_names(int(n_cols)))
        if len(inferred_methods) == 0:
            inferred_methods = ["hdbscan"]
        col_specs = [(i, inferred_methods[i]) for i in range(len(inferred_methods))]

    out_imgs: list[Path] = []
    out_nms: list[str] = []
    for c, nm in col_specs:
        out = render_optimops_embedding_scatter_3d(
            final_dir,
            gse_name=str(gse_name),
            label_col=int(c),
            label_col_name=str(nm),
            out_name=f"GSE_embedding_3d_{nm}.png",
            force=bool(force),
            show_top_down=bool(show_top_down),
        )
        if out is not None:
            out_imgs.append(out)
            out_nms.append(str(nm))


    # 1b) Ground-truth: reference positions + ground-truth Voronoi cell memberships.
    render_ground_truth_scatter_3d(
        final_dir,
        gse_name=str(gse_name),
        out_name="GSE_ground_truth_nodes_3d.png",
        force=bool(force),
        show_top_down=bool(show_top_down),
    )

    # 1c) Embedding: Procrustes-aligned GSE embedding colored by *ground-truth* cell memberships.
    # This is clustering-method agnostic (colors do not depend on hdbscan/infomap labels).
    render_embedding_ground_truth_scatter_3d(
        final_dir,
        gse_name=str(gse_name),
        out_name="GSE_embedding_ground_truth_nodes_3d.png",
        force=bool(force),
        show_top_down=bool(show_top_down),
    )

    # 1d) Focused graph-only 3D UMAP parameterizations (norm-adj eigvecs → UMAP),
    # aligned + colored by ground truth. We keep the UMAP analysis fixed to the
    # 15-component eigsh basis and 30 nearest neighbors, varying only min_dist.
    render_graph_umap_ground_truth_scatter_3d(
        final_dir,
        out_name="UMAP_embedding_3d_eigsh15_nn30_md0.1_ground_truth.png",
        pca_dim=15,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        seed=0,
        force=bool(force),
        show_top_down=bool(show_top_down),
    )
    render_graph_umap_ground_truth_scatter_3d(
        final_dir,
        out_name="UMAP_embedding_3d_eigsh15_nn30_md0.99_ground_truth.png",
        pca_dim=15,
        n_neighbors=30,
        min_dist=0.99,
        metric="cosine",
        seed=0,
        force=bool(force),
        show_top_down=bool(show_top_down),
    )

    # 1e) Graph-only UMAP (min_dist=0.99) clustered with the same HDBSCAN+
    # graph-curation route exposed by clusterplot.execute_clusters_hdbscan().
    # The helper caches both labels and a small NMI summary next to the other
    # per-run outputs.
    try:
        _compute_graph_umap_hdbscan_nmi_for_dir(
            final_dir,
            pca_dim=int(_UMAP_BENCHMARK_PCA_DIM),
            n_neighbors=int(_UMAP_BENCHMARK_N_NEIGHBORS),
            min_dist=float(_UMAP_BENCHMARK_MIN_DIST),
            metric=str(_UMAP_BENCHMARK_METRIC),
            seed=0,
            hdbscan_min_cluster_size=int(_UMAP_HDBSCAN_MIN_CLUSTER_SIZE),
            hdbscan_min_samples=int(_UMAP_HDBSCAN_MIN_SAMPLES),
            force=bool(force),
        )
    except Exception:
        pass

    # 2) Pairwise ROC curve (optional; requires ground truth cell ids)
    if (not out_imgs) or (not bool(render_roc)):
        return
    try:
        idx, coords = _load_gse_output_txt(final_dir / str(gse_name))
        X = _pad_or_truncate_to_3d(coords)
        ref_pos, keep = _load_positions_scaled_for_dir(final_dir, n_expected=int(idx.max() + 1))
        if ref_pos is not None and ref_pos.shape[0] >= int(idx.max() + 1):
            X = _similarity_procrustes(X, _pad_or_truncate_to_3d(ref_pos[idx, :]))[0]
        true_cell = _load_true_cells_for_dir(final_dir, n_expected=int(idx.max() + 1), keep_nodes_global=keep)
        if true_cell is not None:
            roc_path = final_dir / "GSE_embedding_pairwise_ROC.png"
            if (not roc_path.exists()) or force:
                _render_pairwise_roc_from_embedding(
                    X, true_cell[idx], roc_path,
                    title="Pairwise ROC: same Voronoi cell vs embedding distance",
                )
    except Exception:
        pass

# ───────────────────── Segmentation benchmarking utilities ─────────────────────

def _parse_csv_floats(s: str) -> list[float]:
    """
    Parse comma-separated floats from a CLI string.

    Examples:
        "0.5,0.2,0.1" -> [0.5, 0.2, 0.1]
    """
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    out: list[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def _parse_csv_ints(s: str) -> list[int]:
    """Parse comma-separated ints from a CLI string."""
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _ordered_unique_floats(values: list[float]) -> list[float]:
    """Return floats in first-seen order, removing duplicate numeric values."""
    out: list[float] = []
    seen: set[str] = set()
    for x in values:
        xf = float(x)
        key = f"{xf:.12g}"
        if key in seen:
            continue
        seen.add(key)
        out.append(xf)
    return out


def _ordered_unique_ints(values: list[int]) -> list[int]:
    """Return ints in first-seen order, removing duplicates."""
    out: list[int] = []
    seen: set[int] = set()
    for x in values:
        xi = int(x)
        if xi in seen:
            continue
        seen.add(xi)
        out.append(xi)
    return out


def _benchmark_value_key(value: Any) -> str:
    """Stable dict key for heatmap axis values."""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.12g}"
    return str(value)


def _load_segmentation_module(spec: str | None):
    """
    Load a Python module by name or from a .py file path.

    The module must define callables:
        method1(graph), method2(graph), method3(graph)

    Returns:
        python module object
    """
    spec = "" if spec is None else str(spec).strip()
    if not spec:
        spec = "segmentation_methods"

    if spec.endswith(".py") and Path(spec).exists():
        mod_path = Path(spec).resolve()
        mod_dir = str(mod_path.parent)
        if mod_dir not in sys.path:
            sys.path.insert(0, mod_dir)
        module_name = mod_path.stem
        m_spec = importlib.util.spec_from_file_location(module_name, str(mod_path))
        if m_spec is None or m_spec.loader is None:
            raise ImportError(f"Failed to load module from file: {mod_path}")
        mod = importlib.util.module_from_spec(m_spec)
        sys.modules[module_name] = mod
        m_spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod
    else:
        return importlib.import_module(spec)


def load_segmentation_methods(module_spec: str | None) -> dict[str, Callable[[Any], np.ndarray]]:
    """
    Import method1/method2/method3 from a user-provided module.

    The returned dict keys are "method1", "method2", "method3".
    """
    mod = _load_segmentation_module(module_spec)

    methods: dict[str, Callable[[Any], np.ndarray]] = {}
    missing = []
    for name in ("method1", "method2", "method3"):
        fn = getattr(mod, name, None)
        if fn is None or (not callable(fn)):
            missing.append(name)
        else:
            methods[name] = fn

    if missing:
        raise AttributeError(
            f"Segmentation module {getattr(mod, '__name__', mod)!r} is missing callable(s): {missing}. "
            f"Define method1(graph), method2(graph), method3(graph)."
        )
    return methods


def _looks_like_optimops_spec(spec: str | None) -> bool:
    """Heuristic: does `spec` look like it refers to optimOps.py?"""
    if spec is None:
        return False
    s = str(spec).strip()
    if not s:
        return False
    low = s.lower()
    if low in {"optimops", "optimops.py", "optimops_new", "optimops_new.py"}:
        return True
    base = os.path.basename(low)
    # Accept timestamped/local copies such as optimOps(85).py as optimOps backends.
    if base.startswith("optimops") and base.endswith(".py"):
        return True
    return False


def _load_optimops_module(spec: str | None):
    """Load optimOps either by module name or from a .py path."""
    if spec is None or (not str(spec).strip()):
        spec = "optimOps"
    return _load_segmentation_module(str(spec))


def make_optimops_segmentation_methods(args: argparse.Namespace) -> dict[str, Callable[[Any], np.ndarray]]:
    """Create benchmark segmentation wrappers backed by optimOps.run_GSE().

    This plugs directly into the existing benchmark machinery by returning a dict of callables
    that accept a single `graph` argument (CSR or dict), but the wrappers locate the dataset
    directory via `graph.run_dir` (CSR mode) or `graph['run_dir']` (dict mode).

    The wrappers run optimOps once per dataset (cached) and then return the
    final-label layout written by the attached optimOps:
      - col 0: HDBSCAN labels (final embedding)
      - col 1: Infomap labels (final transformed matrix)

    Notes
    -----
    * The dataset directory MUST contain:
        - link_assoc_reindexed.npz
        - index_key.npy
      The patched builder writes these automatically.
    * With the attached optimOps, coarsened/register_zf runs create coarse
      component caches under coarsened_largest_component/, but final
      cluster_labels.npy and GSEoutput.txt live in the parent run directory.
    """
    _prime_optimops_thread_env_from_args(args)
    mod = _load_optimops_module(getattr(args, "optimops_module", None) or getattr(args, "segmentation_module", None))
    _warn_about_unsupported_optimops_flags(args, backend_mod=mod)

    backend_layout_version = int(getattr(mod, "_FINAL_CLUSTER_LABELS_LAYOUT_VERSION", _OPTIMOPS_FINAL_CLUSTER_LAYOUT_VERSION))
    backend_final_min_cluster_size = int(getattr(mod, "_FINAL_CLUSTER_MIN_CLUSTER_SIZE", _OPTIMOPS_FINAL_MIN_CLUSTER_SIZE))
    backend_final_hdbscan_min_samples = int(getattr(mod, "_FINAL_HDBSCAN_MIN_SAMPLES", _OPTIMOPS_FINAL_HDBSCAN_MIN_SAMPLES))
    backend_cluster_methods = tuple(_OPTIMOPS_FINAL_CLUSTER_METHODS)

    run_GSE = getattr(mod, "run_GSE", None)
    if run_GSE is None or (not callable(run_GSE)):
        raise AttributeError(
            f"optimOps integration requested, but module {getattr(mod, '__name__', mod)!r} "
            f"does not define callable run_GSE(output_name, params, coarsen=True)."
        )

    # Per-run cache: key -> (labels_mat, meta)
    cache: dict[tuple, np.ndarray] = {}

    def _extract_run_dir(graph_obj: Any) -> tuple[str, int]:
        """Return (run_dir, n_nodes) from the method input object."""
        if isinstance(graph_obj, dict):
            rd = graph_obj.get("run_dir") or graph_obj.get("dataset_dir") or graph_obj.get("path")
            A0 = graph_obj.get("adjacency")
            if A0 is None:
                raise ValueError("graph dict is missing key 'adjacency'")
            n_nodes = int(A0.shape[0])
        else:
            rd = getattr(graph_obj, "run_dir", None) or getattr(graph_obj, "_run_dir", None)
            n_nodes = int(getattr(graph_obj, "shape")[0])
        if rd is None:
            raise ValueError(
                "optimOps wrappers require a dataset directory path. "
                "Use the patched builder (it attaches run_dir), or run with --method-input dict."
            )
        return _ensure_trailing_sep(str(rd)), n_nodes

    def _normalize_cluster_labels(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 0:
            raise ValueError("cluster_labels.npy contained a scalar; expected (N,) or (N,K)")
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr

    def _ensure_cluster_labels(graph_obj: Any) -> np.ndarray:
        run_dir, n_nodes = _extract_run_dir(graph_obj)
        regzf_flag = _optimops_register_zf_flag(args)
        resolved_register_zf_slice_path = (
            str(_optimops_zf_slice_path_for_run(run_dir, args)) if regzf_flag else ""
        )
        # Cache key includes the tunable optimOps knobs so different settings don't collide.
        desired_final_min_cluster_size = int(backend_final_min_cluster_size)
        desired_final_hdbscan_min_samples = int(backend_final_hdbscan_min_samples)
        expected_label_cols = int(len(backend_cluster_methods))

        # In-process cache key contains only knobs consumed by the current
        # attached optimOps.py run_GSE() public surface.  Runtime-only ensemble
        # process/thread counts are intentionally excluded because they do not
        # change numerical outputs.
        key = (
            run_dir,
            str(getattr(args, "optimops_output_name", "GSEoutput.txt")),
            int(getattr(args, "optimops_inference_dim", 2)),
            int(getattr(args, "optimops_inference_eignum", 30)),
            int(getattr(args, "optimops_final_eignum", 100)),
            int(getattr(args, "optimops_scales", 1)),
            bool(regzf_flag),
            (
                None
                if getattr(args, "optimops_coarsen_annotation_binary_threshold", None) is None
                else int(getattr(args, "optimops_coarsen_annotation_binary_threshold"))
            ),
            regzf_flag,
            resolved_register_zf_slice_path,
            getattr(args, "optimops_register_zf_match_lam_dir", None),
            getattr(args, "optimops_register_zf_match_refine_iter", None),
            getattr(args, "optimops_register_zf_ensemble_size", None),
        )
        if key in cache:
            return cache[key]

        if bool(getattr(args, "optimops_coarsen_infomap", False)) and not regzf_flag:
            raise ValueError(
                "Current optimOps supports -coarsen_infomap only as part of the coarsen/register_zf route; "
                "pass --optimops-register-zf, or remove --optimops-coarsen-infomap."
            )
        coarsen = bool(regzf_flag)
        force = bool(getattr(args, "optimops_force", False))

        out_dir_guess = _ensure_trailing_sep(str(_optimops_final_output_dir_for_run(run_dir, args)))

        cl_path_guess = os.path.join(out_dir_guess, "cluster_labels.npy")
        cl_meta_guess = os.path.join(out_dir_guess, "cluster_labels_meta.json")
        label_dir = out_dir_guess
        reuse_cached_labels = False
        if (not force) and os.path.exists(cl_path_guess):
            try:
                labs_guess = _normalize_cluster_labels(np.load(cl_path_guess))
                meta_guess = None
                if os.path.exists(cl_meta_guess):
                    with open(cl_meta_guess, "r") as fh:
                        meta_guess = json.load(fh)

                methods_guess = _optimops_cluster_method_names_from_meta(meta_guess, int(labs_guess.shape[1]))

                layout_ok = True
                if isinstance(meta_guess, dict) and ("layout_version" in meta_guess):
                    layout_ok = int(meta_guess.get("layout_version", -1)) == int(backend_layout_version)

                hdbscan_mcs_ok = True
                if isinstance(meta_guess, dict) and ("hdbscan_min_cluster_size" in meta_guess):
                    hdbscan_mcs_ok = int(meta_guess.get("hdbscan_min_cluster_size", -1)) == desired_final_min_cluster_size

                hdbscan_ms_ok = True
                if isinstance(meta_guess, dict) and ("hdbscan_min_samples" in meta_guess):
                    hdbscan_ms_ok = int(meta_guess.get("hdbscan_min_samples", -1)) == desired_final_hdbscan_min_samples

                methods_ok = (
                    tuple(methods_guess[:expected_label_cols])
                    == tuple(backend_cluster_methods[:expected_label_cols])
                )

                reuse_cached_labels = (
                    labs_guess.ndim == 2
                    and labs_guess.shape[1] >= expected_label_cols
                    and layout_ok
                    and hdbscan_mcs_ok
                    and hdbscan_ms_ok
                    and methods_ok
                )
                if reuse_cached_labels and regzf_flag:
                    match_dir = _register_zf_match_dir_for_run(run_dir, regzf_flag)
                    regzf_ready = match_dir.exists() and all(
                        (match_dir / name).exists() for name in _REGZF_ALIGNMENT_REQUIRED
                    )
                    regzf_ready = regzf_ready and _register_zf_backend_outputs_ready(run_dir)
                    if not regzf_ready:
                        reuse_cached_labels = False
                if reuse_cached_labels:
                    labs = labs_guess
            except Exception:
                reuse_cached_labels = False
        if not reuse_cached_labels:
            output_name = str(getattr(args, "optimops_output_name", "GSEoutput.txt"))

            # Prefer rerunning directly in the dataset directory that already contains
            # the embedding output so missing cluster_labels.npy can be regenerated
            # without re-exporting the graph or recomputing dimensionality reduction.
            target_dir = out_dir_guess
            target_has_embedding = os.path.exists(os.path.join(target_dir, output_name))
            if regzf_flag:
                # register_zf is wired into optimOps' parent coarsen path.  Start
                # from the parent run directory; coarsened component directories are intermediate caches.
                target_dir = run_dir
            elif not target_has_embedding:
                target_dir = run_dir

            # If existing labels failed the logical/method/artifact checks above,
            # remove them before run_GSE(); optimOps intentionally skips final
            # clustering whenever cluster_labels.npy already exists.
            for stale_name in ("cluster_labels.npy", "cluster_labels_meta.json"):
                stale_path = os.path.join(str(target_dir), stale_name)
                if os.path.exists(stale_path):
                    try:
                        os.remove(stale_path)
                    except Exception:
                        pass

            # Build params dict expected by optimOps.run_GSE.  Preserve the old
            # dummy -calc_final behavior for non-register-zf runs; use an explicit
            # absolute label root only for register_zf so optimOps consumes the
            # synthetic final.h5ad in the parent run directory.
            params = {
                "-path": target_dir,
                "-inference_dim": int(getattr(args, "optimops_inference_dim", 2)),
                "-inference_eignum": int(getattr(args, "optimops_inference_eignum", 30)),
                "-final_eignum": int(getattr(args, "optimops_final_eignum", 100)),
                "-scales": int(getattr(args, "optimops_scales", 1)),
                # Any non-None triggers the final clustering stage that writes cluster_labels.npy
                "-calc_final": (os.path.abspath(str(target_dir)) if regzf_flag else "1"),
            }
            if regzf_flag:
                params["_h5ad_label_root"] = os.path.abspath(str(target_dir))
            if regzf_flag:
                # The graph builder usually wrote these at dataset creation time;
                # this best-effort refresh also covers cached graph-only benchmark runs.
                if (
                    not _h5ad_looks_readable(Path(run_dir) / _REGZF_FIXTURE_PARENT_H5AD)
                    or not os.path.exists(resolved_register_zf_slice_path)

                ):
                    # Rebuilding from the cached graph requires the full simulator positions and ground truth.
                    pos_full = np.asarray(np.load(os.path.join(run_dir, "node_positions_scaled.npy")), dtype=float)
                    gt = pd.read_csv(os.path.join(run_dir, "ground_truth_cells.csv"))
                    write_synthetic_register_zf_fixtures(
                        Path(run_dir),
                        sim_pos=pos_full,
                        partition_labels=gt["partition_label"].to_numpy(dtype=np.int32),
                        cell_ids=(gt["cell_id"].to_numpy(dtype=np.int32) if "cell_id" in gt.columns else None),
                        zf_flag=regzf_flag,
                        slice_h5ad_path=resolved_register_zf_slice_path,
                        write_slice_h5ad=not bool(str(getattr(args, "optimops_slice_path", "") or "").strip()),
                        slice_n=int(getattr(args, "optimops_zf_slice_n", 0)),
                        num_pole_pairs=int(getattr(args, "optimops_zf_num_pole_pairs", 3)),
                        genes_per_pole=int(getattr(args, "optimops_zf_genes_per_pole", 3)),
                        seed=getattr(args, "seed", None),
                    )

            if coarsen:
                params["-coarsen_infomap"] = True
                if getattr(args, "optimops_coarsen_annotation_binary_threshold", None) is not None:
                    params["-coarsen_annotation_binary_threshold"] = int(
                        args.optimops_coarsen_annotation_binary_threshold
                    )
                if regzf_flag:
                    # Public current optimOps.py register_zf surface:
                    #   -register_zf, -slice_path, -register_zf_match_lam_dir,
                    #   -register_zf_match_refine_iter, -register_zf_ensemble_size.
                    # Process/thread runtime controls are not params; optimOps.py
                    # reads them from REGISTER_ZF_ENSEMBLE_* environment variables.
                    params["-register_zf"] = regzf_flag
                    if not resolved_register_zf_slice_path:
                        raise ValueError(
                            "--optimops-register-zf requires --optimops-slice-path or generated register_zf fixtures."
                        )
                    if not os.path.exists(resolved_register_zf_slice_path):
                        raise FileNotFoundError(
                            "register_zf slice h5ad not found for this run: " + resolved_register_zf_slice_path
                        )
                    params["-slice_path"] = resolved_register_zf_slice_path

                    if getattr(args, "optimops_register_zf_match_lam_dir", None) is not None:
                        params["-register_zf_match_lam_dir"] = float(args.optimops_register_zf_match_lam_dir)
                    if getattr(args, "optimops_register_zf_match_refine_iter", None) is not None:
                        params["-register_zf_match_refine_iter"] = int(args.optimops_register_zf_match_refine_iter)
                    if getattr(args, "optimops_register_zf_ensemble_size", None) is not None:
                        params["-register_zf_ensemble_size"] = int(args.optimops_register_zf_ensemble_size)
            else:
                params["-coarsen_infomap"] = None

            regzf_env: dict[str, str | int | None] = {}
            if regzf_flag:
                if getattr(args, "optimops_register_zf_ensemble_n_jobs", None) is not None:
                    regzf_env["REGISTER_ZF_ENSEMBLE_N_JOBS"] = int(args.optimops_register_zf_ensemble_n_jobs)
                if getattr(args, "optimops_register_zf_ensemble_threads_per_worker", None) is not None:
                    regzf_env["REGISTER_ZF_ENSEMBLE_THREADS_PER_WORKER"] = int(args.optimops_register_zf_ensemble_threads_per_worker)
                if getattr(args, "optimops_register_zf_ensemble_mp_start_method", None) is not None:
                    regzf_env["REGISTER_ZF_ENSEMBLE_MP_START_METHOD"] = str(args.optimops_register_zf_ensemble_mp_start_method)

            # Run the pipeline (reusing existing transformed_matrix/GSEoutput when present).
            with _temporary_env_overrides(regzf_env):
                run_GSE(output_name, params, coarsen=coarsen)

            if os.path.normpath(str(target_dir)) == os.path.normpath(str(run_dir)):
                final_dir = _ensure_trailing_sep(str(_optimops_final_output_dir_for_run(run_dir, args)))
            else:
                final_dir = _ensure_trailing_sep(str(target_dir))
            label_dir = final_dir

            cl_path = os.path.join(final_dir, "cluster_labels.npy")
            if not os.path.exists(cl_path):
                raise FileNotFoundError(f"optimOps did not produce cluster_labels.npy at: {cl_path}")
            labs = _normalize_cluster_labels(np.load(cl_path))

            meta_path = os.path.join(final_dir, "cluster_labels_meta.json")
            if not os.path.exists(meta_path):
                meta_out = {
                    "layout_version": int(backend_layout_version),
                    "hdbscan_min_cluster_size": desired_final_min_cluster_size,
                    "hdbscan_min_samples": desired_final_hdbscan_min_samples,
                    "cluster_labels_methods": list(backend_cluster_methods[:labs.shape[1]]),
                    "cluster_methods": list(backend_cluster_methods[:labs.shape[1]]),
                }
                try:
                    with open(meta_path, "w") as fh:
                        json.dump(meta_out, fh, indent=2, sort_keys=True)
                except Exception:
                    pass

        # If coarsening produced a fine-scale dataset, cluster_labels.npy may be on that
        # fine index.  When keep_nodes_global.npy is present, map those labels back to the
        # full node set so benchmark NMI can still be computed against ground truth.
        if labs.shape[0] != n_nodes:
            keep_path = os.path.join(label_dir, "keep_nodes_global.npy")
            if os.path.exists(keep_path):
                keep = np.asarray(np.load(keep_path)).astype(np.int64, copy=False).ravel()

                # Prefer an explicit composed mapping if optimOps exported one.
                keep_full_path = os.path.join(label_dir, "keep_nodes_global_full.npy")
                if os.path.exists(keep_full_path):
                    try:
                        keep = np.asarray(np.load(keep_full_path)).astype(np.int64, copy=False).ravel()
                    except Exception:
                        pass
                else:
                    # Compose nested keep_nodes_global.npy mappings across parent directories when present.
                    try:
                        if os.path.normpath(str(label_dir)) != os.path.normpath(str(run_dir)):
                            keep2 = _compose_keep_nodes_global_chain(Path(label_dir), keep, max_up=4)
                            if keep2 is not None:
                                keep = keep2
                    except Exception:
                        pass
                # If cluster_labels are saved as (K, N_fine), transpose before mapping.
                if keep.shape[0] != labs.shape[0] and keep.shape[0] == labs.shape[1]:
                    labs = labs.T

                if keep.shape[0] == labs.shape[0]:
                    # Fill nodes that are outside the processed support with a
                    # *distinct* sentinel so we can ignore them during NMI
                    # computation (and not confuse them with legitimate noise
                    # labels like -1 from HDBSCAN).
                    out = np.full((n_nodes, labs.shape[1]), _UNPROCESSED_LABEL, dtype=np.int32)
                    out[keep, :] = labs.astype(np.int32, copy=False)
                    labs = out
        # Best-effort orientation fix (sometimes cluster_labels are saved transposed).
        if labs.shape[0] != n_nodes and labs.shape[1] == n_nodes:
            labs = labs.T

        if labs.shape[0] != n_nodes:
            raise ValueError(
                f"optimOps cluster_labels has shape {labs.shape}, but expected N={n_nodes} rows. "
                "If coarsening is enabled, ensure keep_nodes_global.npy is present so labels can be mapped."
            )

        if labs.shape[1] < expected_label_cols:
            raise ValueError(
                f"optimOps cluster_labels has shape {labs.shape}, but expected at least {expected_label_cols} columns "
                f"for methods {list(backend_cluster_methods)}."
            )

        labs = np.asarray(labs, dtype=np.int32)

        # Best-effort visualization: 3D embedding scatter (colored by cluster labels) + faint graph edges.
        # This uses the on-disk optimOps outputs in the *final* directory (cluster_labels.npy, GSEoutput.txt,
        # link_assoc_reindexed.npz) and Procrustes-aligns the embedding to the simulator geometry when
        # node_positions_scaled.npy is available.
        try:
            maybe_render_optimops_outputs(
                Path(label_dir),
                gse_name=str(getattr(args, "optimops_output_name", "GSEoutput.txt")),
                force=force,
                render_roc=bool(getattr(args, "optimops_render_roc", False)),
                show_top_down=bool(getattr(args, "hole", False)),
            )
        except Exception:
            pass

        if regzf_flag:
            try:
                render_register_zf_alignment_diagnostics_for_run(Path(run_dir), args, force=force)
            except Exception as e:
                print(f"[WARN] register_zf alignment diagnostic rendering failed: {e}")

        cache[key] = labs
        return labs

    def _column_method(col_idx: int):
        def _fn(graph_obj: Any) -> np.ndarray:
            labs = _ensure_cluster_labels(graph_obj)
            if col_idx >= labs.shape[1]:
                return np.full(labs.shape[0], _UNPROCESSED_LABEL, dtype=np.int32)
            return labs[:, col_idx].copy()
        return _fn

    return {
        f"optimops_{name}": _column_method(i)
        for i, name in enumerate(backend_cluster_methods)
    }


def _membrane_strength_for_barrier_ratio(*, barrier_ratio: float, D_in: float, D_out: float) -> float:
    """
    Map a desired membrane-vs-intracellular diffusion ratio to membrane_strength (beta_mem).

    In the diffusion-field construction (see model.tex supplement), the boundary (membrane) diffusivity
    is approximately:
        D_membrane ≈ D_out * exp(-beta_mem)
    while intracellular baseline is D_in.

    If we want:
        barrier_ratio = D_membrane / D_in
    then:
        beta_mem = log(D_out / (barrier_ratio * D_in)).

    Note:
      - If D_out <= 0 it is treated as D_in (consistent with builder defaults).
      - Returned beta_mem is clipped to >= 0.
    """
    barrier_ratio = float(barrier_ratio)
    if (not np.isfinite(barrier_ratio)) or barrier_ratio <= 0:
        raise ValueError("barrier_ratio must be finite and > 0")
    D_in = float(D_in)
    if (not np.isfinite(D_in)) or D_in <= 0:
        raise ValueError("D_in must be finite and > 0")
    D_out = float(D_out)
    if (not np.isfinite(D_out)) or D_out <= 0:
        D_out = D_in
    beta = float(np.log(D_out / (barrier_ratio * D_in)))
    return float(max(0.0, beta))


def _symmetrize_graph(G: csr_matrix) -> csr_matrix:
    """Make an undirected adjacency (CSR) from a possibly directed CSR matrix."""
    if not isinstance(G, csr_matrix):
        G = G.tocsr()
    A = (G + G.T).tocsr()
    A.sum_duplicates()
    return A


def _giant_component_fraction(A: csr_matrix) -> float:
    """Return largest connected component size / n for an undirected graph adjacency."""
    from scipy.sparse.csgraph import connected_components
    if A.shape[0] == 0:
        return 0.0
    n_comp, lab = connected_components(A, directed=False)
    if n_comp <= 1:
        return 1.0
    sizes = np.bincount(lab, minlength=int(n_comp))
    return float(np.max(sizes) / float(A.shape[0]))


def _plot_diffusion_slices_3d(field: CellularDiffusionField,
                             sim_pos: np.ndarray,
                             outpath: Path,
                             *,
                             nx: int = 260,
                             ny: int = 260,
                             z_slices: int = 3,
                             title: str = "log D_eff(x) slices with Voronoi boundaries (scaled coords)") -> None:
    """
    Reproduce the 3D cross-section heatmap visualization (log D_eff) used in the builder,
    but as a reusable helper.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo = np.min(sim_pos, axis=0)
    hi = np.max(sim_pos, axis=0)
    xs = np.linspace(lo[0], hi[0], int(nx))
    ys = np.linspace(lo[1], hi[1], int(ny))
    z_vals = np.linspace(lo[2], hi[2], int(z_slices))

    fig, axes = plt.subplots(1, int(z_slices), figsize=(4.5 * int(z_slices), 4.0))
    if int(z_slices) == 1:
        axes = [axes]

    im = None
    for ax, z in zip(axes, z_vals):
        XX, YY = np.meshgrid(xs, ys, indexing='xy')
        grid = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, z)], axis=1)
        Dg = field(grid).reshape(int(ny), int(nx))
        cid = field.assign_cells(grid).reshape(int(ny), int(nx))
        boundary = np.zeros((int(ny), int(nx)), dtype=bool)
        boundary[:, 1:] |= (cid[:, 1:] != cid[:, :-1])
        boundary[1:, :] |= (cid[1:, :] != cid[:-1, :])

        im = ax.imshow(np.log(Dg), origin='lower',
                       extent=(lo[0], hi[0], lo[1], hi[1]),
                       aspect='auto')
        ax.contour(boundary.astype(float), levels=[0.5], origin='lower',
                   extent=(lo[0], hi[0], lo[1], hi[1]),
                   linewidths=0.5)
        ax.set_title(f"z={z:.2f}")
        ax.set_xlabel("x (scaled)")
        ax.set_ylabel("y (scaled)")

    fig.suptitle(str(title))
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.8, label="log D_eff")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outpath), dpi=200)
    plt.close(fig)


def _plot_voronoi_boundary_isosurface_marching_cubes(
        field: CellularDiffusionField,
        sim_pos: np.ndarray,
        outpath: Path,
        *,
        grid_res: int = 85,
        level: float | None = None,
        show_points: bool = True,
        point_size: float = 1.0,
        screenshot_scale: int = 1,
        title: str = "Voronoi boundary isosurface (δ(x)=const)") -> None:
    """
    Render an approximate Voronoi boundary isosurface via marching cubes on δ(x),
    where δ(x) is the proxy distance-to-boundary from the cellular diffusion field.

    This does NOT require pyvista; it uses skimage.measure.marching_cubes and matplotlib.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _mpl_set_arial_fonts(base_size=12.0)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    grid_res = int(max(25, grid_res))

    lo = np.min(sim_pos, axis=0)
    hi = np.max(sim_pos, axis=0)
    span = np.maximum(hi - lo, 1e-9)

    nx = ny = nz = grid_res
    xs = np.linspace(lo[0], hi[0], nx, dtype=float)
    ys = np.linspace(lo[1], hi[1], ny, dtype=float)
    zs = np.linspace(lo[2], hi[2], nz, dtype=float)

    dx = float(span[0] / float(nx - 1))
    dy = float(span[1] / float(ny - 1))
    dz = float(span[2] / float(nz - 1))

    # Choose a default isosurface level ~ within the membrane band if available.
    if level is None:
        if field.membrane_width > 0:
            level = 0.5 * float(field.membrane_width)
        else:
            # Use a small fraction of median center spacing if membrane width is disabled.
            _tree = cKDTree(field.centers)
            _dnn, _ = _tree.query(field.centers, k=2)
            sp = float(np.median(_dnn[:, 1])) if _dnn.shape[1] > 1 else float(np.median(_dnn))
            level = 0.02 * max(sp, 1e-6)
    level = float(max(1e-9, level))

    # Build grid points and evaluate δ(x) in chunks (KDTree queries can be heavy at full size).
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    delta = np.empty((pts.shape[0],), dtype=float)
    chunk = 200000
    try:
        workers = _resolve_n_threads(int(os.getenv("SLURM_CPUS_PER_TASK", "0") or 0))
    except Exception:
        workers = None
    for s in range(0, pts.shape[0], chunk):
        e = min(s + chunk, pts.shape[0])
        delta[s:e], _ = field._boundary_distance(pts[s:e], workers=workers)
    delta_grid = delta.reshape((nx, ny, nz))

    # Marching cubes. spacing makes verts come out in physical units with origin at 0; add lo as offset.
    verts, faces, _, _ = marching_cubes(delta_grid, level=level, spacing=(dx, dy, dz))
    verts = verts + lo[None, :]

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.35)
    # Avoid heavy edge rendering.
    mesh.set_edgecolor((0, 0, 0, 0))
    ax.add_collection3d(mesh)

    if show_points:
        # Downsample points for speed if needed.
        pts_show = sim_pos
        if pts_show.shape[0] > 6000:
            rng = np.random.default_rng(0)
            pts_show = pts_show[rng.choice(pts_show.shape[0], size=6000, replace=False)]
        ax.scatter(pts_show[:, 0], pts_show[:, 1], pts_show[:, 2], s=float(point_size), alpha=0.35)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_title(f"{title}\n(level δ={level:.3g}, grid={grid_res}^3)")

    ax.set_xlabel("x (scaled)")
    ax.set_ylabel("y (scaled)")
    ax.set_zlabel("z (scaled)")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outpath), dpi=int(200 * max(1, int(screenshot_scale))))
    plt.close(fig)



# ───────────────────────── AO Voronoi isosurface helpers ─────────────────────────
def _render_isosurface_3d_ao_voronoi_3d_patch(
    centers: np.ndarray,
    sim_pos: np.ndarray,
    labels: np.ndarray,
    outpath: Path,
    *,
    D0: float = 1.0,
    D_min: float = 0.05,
    D_max: float = 10.0,
    qp_modes: int = 6,
    qp_amp: float = 0.7,
    cell_sigma: float = 0.5,
    cell_q_corr: float = 0.6,
    membrane_width: float | None = None,
    membrane_width_frac: float = 0.15,
    membrane_strength: float = 2.0,
    grid_res: int = 100,
    seed: int | None = None,
    panel_size: tuple[int, int] = (1400, 1100),
    show_points: bool = False,
    point_size: float = 4.0,
    screenshot_scale: int = 1,
) -> bool:
    """Try AO isosurface rendering using an external patch renderer.

    Returns
    -------
    bool
        True if the AO renderer ran successfully, False otherwise.

    Notes
    -----
    This remains a soft dependency: if PyVista (or the patch module) is not
    available, we return False and let the caller fall back.
    """
    try:
        from voronoi_3d_patch import render_isosurface_3d
    except Exception:
        return False

    try:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)

        # Matplotlib's bbox_inches="tight" can sometimes clip 3-panel AO composites
        # from off-screen renders on certain backends. Disable 'tight' bbox for this call.
        _MplFigure = None
        _orig_savefig = None
        try:
            from matplotlib.figure import Figure as _MplFigure  # type: ignore
            _orig_savefig = _MplFigure.savefig

            def _savefig_no_tight(self, *a, **k):
                if k.get("bbox_inches", None) == "tight":
                    k.pop("bbox_inches", None)
                    k.pop("pad_inches", None)
                return _orig_savefig(self, *a, **k)

            _MplFigure.savefig = _savefig_no_tight  # type: ignore
        except Exception:
            _MplFigure = None
            _orig_savefig = None

        try:
            render_isosurface_3d(
            centers=np.asarray(centers, dtype=float),
            sim_pos=np.asarray(sim_pos, dtype=float),
            labels=np.asarray(labels, dtype=int),
            D0=float(D0),
            D_min=float(D_min),
            D_max=float(D_max),
            qp_modes=int(qp_modes),
            qp_amp=float(qp_amp),
            cell_sigma=float(cell_sigma),
            cell_q_corr=float(cell_q_corr),
            membrane_width=(None if membrane_width is None else float(membrane_width)),
            membrane_width_frac=float(membrane_width_frac),
            membrane_strength=float(membrane_strength),
            grid_res=int(grid_res),
            seed=(None if seed is None else int(seed)),
            outpath=str(outpath),
            panel_size=(int(panel_size[0]), int(panel_size[1])),
            show_points=bool(show_points),
            point_size=float(point_size),
            screenshot_scale=int(max(1, screenshot_scale)),
        )
        finally:
            if _MplFigure is not None and _orig_savefig is not None:
                try:
                    _MplFigure.savefig = _orig_savefig  # type: ignore
                except Exception:
                    pass
        return True
    except Exception:
        return False


def _ensure_voronoi_isosurface(
    *,
    centers: np.ndarray,
    sim_pos: np.ndarray,
    labels: np.ndarray,
    field: Any | None,
    outpath: Path,
    D0: float,
    D_min: float,
    D_max: float,
    qp_modes: int,
    qp_amp: float,
    cell_sigma: float,
    cell_q_corr: float,
    membrane_width: float | None,
    membrane_width_frac: float,
    membrane_strength: float,
    grid_res: int,
    seed: int | None,
    panel_size: tuple[int, int] = (1400, 1100),
    show_points: bool = False,
    point_size: float = 4.0,
    screenshot_scale: int = 1,
) -> None:
    """Ensure a Voronoi boundary isosurface image exists at `outpath`.

    Preferred backend (SSAO):
      * `voronoi_3d_patch.render_isosurface_3d` (PyVista + ambient occlusion)

    Fallback backend:
      * `_plot_voronoi_boundary_isosurface_marching_cubes` (matplotlib only)

    The fallback is used only if the AO renderer is unavailable or errors.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    # This renderer can be one of the slowest pieces of a large benchmark sweep.
    # Treat an existing nonempty image as a valid cache artifact so resumed runs
    # do not spend minutes to hours re-rendering the same surface.
    try:
        if outpath.exists() and outpath.stat().st_size > 0:
            return
    except Exception:
        pass

    ok = _render_isosurface_3d_ao_voronoi_3d_patch(
        centers=centers,
        sim_pos=sim_pos,
        labels=labels,
        outpath=outpath,
        D0=D0,
        D_min=D_min,
        D_max=D_max,
        qp_modes=qp_modes,
        qp_amp=qp_amp,
        cell_sigma=cell_sigma,
        cell_q_corr=cell_q_corr,
        membrane_width=membrane_width,
        membrane_width_frac=membrane_width_frac,
        membrane_strength=membrane_strength,
        grid_res=grid_res,
        seed=seed,
        panel_size=panel_size,
        show_points=show_points,
        point_size=point_size,
        screenshot_scale=1,
    )
    if ok:
        return

    if field is None:
        print("[WARN] AO isosurface renderer unavailable and no fallback field provided; skipping isosurface.")
        return

    print("[WARN] AO isosurface renderer unavailable; falling back to marching-cubes isosurface (no SSAO).")
    try:
        _plot_voronoi_boundary_isosurface_marching_cubes(
            field,
            sim_pos,
            outpath,
            grid_res=int(max(25, min(200, grid_res))),
            show_points=bool(show_points),
            point_size=float(point_size),
            screenshot_scale=int(max(1, screenshot_scale)),
        )
    except Exception as e:
        print(f"[WARN] Fallback marching-cubes isosurface rendering failed: {e}")


def _make_graph_object_for_methods(A: csr_matrix,
                                  *,
                                  n0: int,
                                  n1: int,
                                  method_input: str,
                                  run_dir: Path | str | None = None) -> Any:
    """
    Construct the object passed to user segmentation methods.

    method_input:
      - "csr": pass A (CSR adjacency)
      - "dict": pass {"adjacency": A, "n0": n0, "n1": n1, "run_dir": <str>}

    Notes
    -----
    Some segmentation backends (e.g. optimOps) need access to the on-disk dataset directory
    to read/write intermediate files.  For convenience we attach `run_dir` either as:
      - dict key "run_dir" (when --method-input dict), or
      - attribute `A.run_dir` (when --method-input csr).
    """
    mi = str(method_input).strip().lower()
    if mi == "dict":
        d = {"adjacency": A, "n0": int(n0), "n1": int(n1)}
        if run_dir is not None:
            d["run_dir"] = str(run_dir)
        return d

    # Attach run_dir to the CSR object (safe: csr_matrix has a __dict__).
    if run_dir is not None:
        try:
            setattr(A, "run_dir", str(run_dir))
        except Exception:
            pass
    return A


def _evaluate_methods_on_run(methods: dict[str, Callable[[Any], np.ndarray]],
                             *,
                             A: csr_matrix,
                             true_cell: np.ndarray,
                             n0: int,
                             n1: int,
                             method_input: str,
                             run_dir: Path | str | None = None) -> dict[str, float]:
    """Compute NMI for each method on one dataset."""
    from sklearn.metrics import normalized_mutual_info_score

    graph_obj = _make_graph_object_for_methods(A, n0=n0, n1=n1, method_input=method_input, run_dir=run_dir)

    scores: dict[str, float] = {}
    for name, fn in methods.items():
        pred = fn(graph_obj)
        pred = np.asarray(pred)
        if pred.ndim != 1 or pred.shape[0] != true_cell.shape[0]:
            raise ValueError(
                f"{name} returned labels of shape {pred.shape}; expected (n_nodes,) = ({true_cell.shape[0]},)"
            )
        # Some backends (notably optimOps with --optimops-coarsen-infomap) only
        # solve/clusters a strict node subset and then pad back to the full node
        # set with a sentinel label. Treat those nodes as *unprocessed* rather
        # than as a real cluster when computing NMI.
        if np.any(pred == _UNPROCESSED_LABEL):
            mask = (pred != _UNPROCESSED_LABEL)
            n_valid = int(np.sum(mask))
            if n_valid < 2:
                scores[name] = float('nan')
            else:
                scores[name] = float(
                    normalized_mutual_info_score(
                        true_cell[mask],
                        pred[mask],
                        average_method="arithmetic",
                    )
                )
        else:
            scores[name] = float(
                normalized_mutual_info_score(
                    true_cell,
                    pred,
                    average_method="arithmetic",
                )
            )

    return scores


def _benchmark_uses_optimops_backend(args: argparse.Namespace) -> bool:
    """Return True when the benchmark methods are backed by optimOps."""
    return bool(getattr(args, "use_optimops", False)) or _looks_like_optimops_spec(
        getattr(args, "optimops_module", None) or getattr(args, "segmentation_module", None)
    )


def _optimops_final_output_dir_for_run(run_dir: Path | str,
                                       args: argparse.Namespace) -> Path:
    """Return the directory expected to contain current optimOps final outputs.

    The attached optimOps writes GSEoutput.txt and cluster_labels.npy in the
    parent dataset directory.  Coarsened/component directories are intermediate
    caches only.
    """
    return Path(run_dir)


def _benchmark_embedding_dir_for_run(run_dir: Path | str,
                                     args: argparse.Namespace) -> Path:
    """Return the dataset directory used for graph-only UMAP/HDBSCAN benchmarking."""
    run_dir = Path(run_dir)
    if _benchmark_uses_optimops_backend(args):
        final_dir = _optimops_final_output_dir_for_run(run_dir, args)
        if final_dir.exists():
            return final_dir
    return run_dir


def _run_has_reusable_optimops_results(run_dir: Path | str,
                                       args: argparse.Namespace) -> bool:
    """Whether a benchmark run's cached simulation artifacts can be reused.

    Missing downstream optimOps analysis outputs (for example cluster_labels.npy)
    should not force graph / ground-truth regeneration; the optimOps wrapper can
    regenerate those outputs from the cached dataset directory.
    """
    if not _benchmark_uses_optimops_backend(args):
        return False

    run_dir = Path(run_dir)
    required = [
        run_dir / "graph.npz",
        run_dir / "ground_truth_cells.csv",
        run_dir / "link_assoc_reindexed.npz",
        run_dir / "index_key.npy",
        run_dir / "node_positions_scaled.npy",
    ]
    # Do not require final.h5ad/register_zf_slice.h5ad for simulation-cache reuse.
    # Those are downstream optimOps/register_zf fixtures, and _ensure_cluster_labels()
    # can regenerate them from node_positions_scaled.npy + ground_truth_cells.csv.
    # Requiring them here would force an expensive graph resimulation after a
    # fixture-write failure, even though the graph/ground-truth cache is valid.
    return all(p.exists() for p in required)


def _load_existing_benchmark_run(run_dir: Path | str) -> tuple[csr_matrix, np.ndarray, int, int, float]:
    """Load a previously-built benchmark run from disk."""
    from scipy.sparse import load_npz

    run_dir = Path(run_dir)
    graph_path = run_dir / "graph.npz"
    gt_path = run_dir / "ground_truth_cells.csv"
    if (not graph_path.exists()) or (not gt_path.exists()):
        raise FileNotFoundError(
            f"Expected cached benchmark artifacts at {run_dir} (missing graph.npz and/or ground_truth_cells.csv)."
        )

    G = load_npz(graph_path).tocsr()
    A = _symmetrize_graph(G)

    gt = pd.read_csv(gt_path)
    if "cell_id" not in gt.columns:
        raise ValueError(f"{gt_path} is missing required column 'cell_id'.")
    if "partition_label" not in gt.columns:
        raise ValueError(f"{gt_path} is missing required column 'partition_label'.")

    true_cell = gt["cell_id"].to_numpy(dtype=int)
    if true_cell.shape[0] != A.shape[0]:
        raise ValueError(
            f"Cached run at {run_dir} is inconsistent: graph has {A.shape[0]} nodes but "
            f"ground truth has {true_cell.shape[0]} rows."
        )

    part = gt["partition_label"].to_numpy(dtype=int)
    n0 = int(np.sum(part == 0))
    n1 = int(np.sum(part == 1))
    giant_frac = _giant_component_fraction(A)
    return A, true_cell, n0, n1, giant_frac


def _load_or_build_benchmark_run(*,
                                 run_dir: Path | str,
                                 args: argparse.Namespace,
                                 build_fn: Callable[[], None]) -> tuple[csr_matrix, np.ndarray, int, int, float]:
    """Load cached simulation artifacts when possible, otherwise build and reload from disk."""
    run_dir = Path(run_dir)
    if _run_has_reusable_optimops_results(run_dir, args):
        print(f"[INFO] Reusing cached benchmark simulation: {run_dir}")
        return _load_existing_benchmark_run(run_dir)

    build_fn()
    return _load_existing_benchmark_run(run_dir)


def run_segmentation_benchmark(args: argparse.Namespace) -> None:
    """
    Benchmark external segmentation methods against the simulated Voronoi 'cell' ground truth.

    Sweeps
    ------
      1) joint sweep of node sampling density × cell spatial density
         (node_scale × n_cells)
      2) optional joint sweep of cell spatial density × membrane/intracellular
         diffusivity ratio (n_cells × barrier_ratio)
      3) optional joint sweep of cell spatial density × false fraction
         (n_cells × false_fraction)

    Outputs (written under args.outdir):
      - benchmark_results.csv
      - benchmark_plot_joint_density_NMI.png
      - benchmark_plot_joint_density_decoherence_radius.png
      - benchmark_plot_cell_density_barrier_ratio_NMI.png          [if selected]
      - benchmark_plot_cell_density_barrier_ratio_decoherence_radius.png  [if selected]
      - benchmark_plot_cell_density_false_fraction_NMI.png         [if selected]
      - benchmark_plot_cell_density_false_fraction_decoherence_radius.png [if selected]
    """
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.n0 is None or args.n1 is None:
        raise SystemExit("--benchmark requires base --n0 and --n1 (used as the baseline node sampling density).")

    second_sweep = str(getattr(args, "benchmark_second_sweep", "barrier_ratio")).lower()
    if second_sweep not in {"barrier_ratio", "false_fraction", "both"}:
        raise SystemExit("--benchmark-second-sweep must be one of: barrier_ratio, false_fraction, both.")
    run_barrier_sweep = second_sweep in {"barrier_ratio", "both"}
    run_false_sweep = second_sweep in {"false_fraction", "both"}

    if _benchmark_uses_optimops_backend(args):
        methods = make_optimops_segmentation_methods(args)
    else:
        methods = load_segmentation_methods(args.segmentation_module)

    _pretty_method_name = {
        "optimops_hdbscan": "optimOps: HDBSCAN (final embedding)",
        "optimops_infomap": "optimOps: Infomap (final transport graph)",
        "gse_embedding_decoherence": "GSE: decoherence radius",
        _UMAP_DECOHERENCE_METHOD_KEY: _UMAP_DECOHERENCE_METHOD_LABEL,
        _UMAP_HDBSCAN_METHOD_KEY: _UMAP_HDBSCAN_METHOD_LABEL,
        "register_zf_alignment": "register_zf: alignment diagnostics",
    }

    def _m_label(name: str) -> str:
        return _pretty_method_name.get(str(name), str(name))

    required_nmi_methods = list(methods.keys()) + [_UMAP_HDBSCAN_METHOD_KEY]

    barrier_ratios = _ordered_unique_floats(_parse_csv_floats(args.sweep_barrier_ratios))
    if not barrier_ratios:
        barrier_ratios = [0.5, 0.2, 0.1, 0.05, 0.02]
    if any((not np.isfinite(br)) or br <= 0.0 for br in barrier_ratios):
        raise SystemExit("--sweep-barrier-ratios values must be finite and > 0.")

    false_fracs = _ordered_unique_floats(_parse_csv_floats(getattr(args, "sweep_false_fracs", "")))
    if not false_fracs:
        false_fracs = [0.0, 0.01, 0.02, 0.05, 0.10]
    if any((ff < 0.0) or (ff > 1.0) for ff in false_fracs):
        raise SystemExit("--sweep-false-fracs values must lie in [0, 1].")

    node_scales = _ordered_unique_floats(_parse_csv_floats(args.sweep_node_scales))
    if not node_scales:
        node_scales = [0.5, 1.0, 2.0]

    n_cells_raw = _parse_csv_ints(args.sweep_ncells)
    n_cells_list = _ordered_unique_ints(n_cells_raw) if n_cells_raw else [100, 200, 400, 800]

    reps = int(max(1, args.sweep_reps))
    base_seed = 0 if args.seed is None else int(args.seed)
    base_n0 = int(args.n0)
    base_n1 = int(args.n1)
    base_false_frac = float(args.false_edge_frac)
    false_mode = ("nodes" if bool(args.false_target_nodes) else "edges")

    rows: list[dict[str, Any]] = []

    bbox_lo = (tuple(args.bbox_lo) if args.bbox_lo is not None else None)
    bbox_hi = (tuple(args.bbox_hi) if args.bbox_hi is not None else None)

    def _corruption_dir(frac: float | None = None) -> str:
        if frac is None:
            return "false_nodes" if bool(args.false_target_nodes) else "false_edges"
        prefix = "false_nodes" if bool(args.false_target_nodes) else "false_edges"
        return f"{prefix}_{float(frac):.6g}"

    def _base_row(*,
                  sweep: str,
                  x: float,
                  rep: int,
                  method: str,
                  metric_value: float,
                  giant_frac: float,
                  n0: int,
                  n1: int,
                  n_cells: int,
                  node_scale: float,
                  barrier_ratio: float | None,
                  membrane_strength: float,
                  false_fraction: float) -> dict[str, Any]:
        return {
            "benchmark_layout_version": int(_BENCHMARK_LAYOUT_VERSION),
            "sweep": str(sweep),
            "benchmark_stat": _BENCHMARK_METHOD_STAT_KEY,
            "x": float(x),
            "replicate": int(rep),
            "method": str(method),
            _BENCHMARK_METRIC_KEY: float(metric_value),
            "giant_component_frac": float(giant_frac),
            "n0": int(n0),
            "n1": int(n1),
            "n_cells": int(n_cells),
            "node_scale": float(node_scale),
            "barrier_ratio": (float(barrier_ratio) if barrier_ratio is not None else float("nan")),
            "membrane_strength": float(membrane_strength),
            "hole": bool(args.hole),
            "false_fraction": float(false_fraction),
            "false_edge_frac": float(false_fraction),  # compatibility alias
            "false_mode": str(false_mode),
            "false_target_nodes": bool(args.false_target_nodes),
            "optimops_register_zf": str(_optimops_register_zf_flag(args)),
            "optimops_slice_path": str(getattr(args, "optimops_slice_path", "") or ""),
            "optimops_zf_synthetic_fixtures": bool(getattr(args, "optimops_zf_synthetic_fixtures", False)),
            _BENCHMARK_DECOHERENCE_RADIUS_KEY: float("nan"),
            "avg_total_counts_per_node": float("nan"),
            "decoherence_threshold": float("nan"),
        }

    def _append_score_rows(*,
                           sweep: str,
                           x: float,
                           rep: int,
                           scores: dict[str, float],
                           umap_hdbscan_nmi: float,
                           giant_frac: float,
                           n0: int,
                           n1: int,
                           n_cells: int,
                           node_scale: float,
                           barrier_ratio: float | None,
                           membrane_strength: float,
                           false_fraction: float) -> None:
        common = dict(
            sweep=sweep,
            x=x,
            rep=rep,
            giant_frac=giant_frac,
            n0=n0,
            n1=n1,
            n_cells=n_cells,
            node_scale=node_scale,
            barrier_ratio=barrier_ratio,
            membrane_strength=membrane_strength,
            false_fraction=false_fraction,
        )
        for mname, nmi in scores.items():
            rows.append(_base_row(method=str(mname), metric_value=float(nmi), **common))
        umap_row = _base_row(
            method=_UMAP_HDBSCAN_METHOD_KEY,
            metric_value=float(umap_hdbscan_nmi),
            **common,
        )
        umap_row["benchmark_stat"] = _BENCHMARK_UMAP_HDBSCAN_STAT_KEY
        umap_row.update({
            "umap_min_dist": float(_UMAP_BENCHMARK_MIN_DIST),
            "umap_n_neighbors": int(_UMAP_BENCHMARK_N_NEIGHBORS),
            "umap_metric": str(_UMAP_BENCHMARK_METRIC),
            "hdbscan_min_cluster_size": int(_UMAP_HDBSCAN_MIN_CLUSTER_SIZE),
            "hdbscan_min_samples": int(_UMAP_HDBSCAN_MIN_SAMPLES),
        })
        rows.append(umap_row)

    def _append_decoherence_row(*,
                                sweep: str,
                                x: float,
                                rep: int,
                                method: str,
                                decoherence_radius: float,
                                avg_total_counts_per_node: float,
                                decoherence_threshold: float,
                                giant_frac: float,
                                n0: int,
                                n1: int,
                                n_cells: int,
                                node_scale: float,
                                barrier_ratio: float | None,
                                membrane_strength: float,
                                false_fraction: float) -> None:
        row = _base_row(
            sweep=sweep,
            x=x,
            rep=rep,
            method=str(method),
            metric_value=float("nan"),
            giant_frac=giant_frac,
            n0=n0,
            n1=n1,
            n_cells=n_cells,
            node_scale=node_scale,
            barrier_ratio=barrier_ratio,
            membrane_strength=membrane_strength,
            false_fraction=false_fraction,
        )
        row["benchmark_stat"] = _BENCHMARK_DECOHERENCE_RADIUS_STAT_KEY
        row[_BENCHMARK_METRIC_KEY] = float("nan")
        row[_BENCHMARK_DECOHERENCE_RADIUS_KEY] = float(decoherence_radius)
        row["avg_total_counts_per_node"] = float(avg_total_counts_per_node)
        row["decoherence_threshold"] = float(decoherence_threshold)
        rows.append(row)

    def _append_run_metric_rows(*,
                                run_dir: Path,
                                sweep: str,
                                x: float,
                                rep: int,
                                scores: dict[str, float],
                                umap_hdbscan_nmi: float,
                                giant_frac: float,
                                n0: int,
                                n1: int,
                                n_cells: int,
                                node_scale: float,
                                barrier_ratio: float | None,
                                membrane_strength: float,
                                false_fraction: float,
                                deco_seed: int) -> None:
        _append_score_rows(
            sweep=sweep,
            x=x,
            rep=rep,
            scores=scores,
            umap_hdbscan_nmi=float(umap_hdbscan_nmi),
            giant_frac=float(giant_frac),
            n0=int(n0),
            n1=int(n1),
            n_cells=int(n_cells),
            node_scale=float(node_scale),
            barrier_ratio=barrier_ratio,
            membrane_strength=float(membrane_strength),
            false_fraction=float(false_fraction),
        )

        final_dir = _benchmark_embedding_dir_for_run(run_dir, args)
        deco_radius, deco_payload = _compute_gse_decoherence_radius_for_dir(
            final_dir,
            gse_name=str(getattr(args, "optimops_output_name", "GSEoutput.txt")),
            graph_name="link_assoc_reindexed.npz",
            seed=int(deco_seed),
            force=bool(getattr(args, "optimops_force", False)),
        )
        _append_decoherence_row(
            sweep=sweep,
            x=x,
            rep=rep,
            method="gse_embedding_decoherence",
            decoherence_radius=float(deco_radius),
            avg_total_counts_per_node=float(deco_payload.get("avg_total_counts_per_node", float("nan"))),
            decoherence_threshold=float(deco_payload.get("threshold", float("nan"))),
            giant_frac=float(giant_frac),
            n0=int(n0),
            n1=int(n1),
            n_cells=int(n_cells),
            node_scale=float(node_scale),
            barrier_ratio=barrier_ratio,
            membrane_strength=float(membrane_strength),
            false_fraction=float(false_fraction),
        )

        umap_deco_radius, umap_deco_payload = _compute_graph_umap_decoherence_radius_for_dir(
            final_dir,
            graph_name="link_assoc_reindexed.npz",
            pca_dim=int(_UMAP_BENCHMARK_PCA_DIM),
            n_neighbors=int(_UMAP_BENCHMARK_N_NEIGHBORS),
            min_dist=float(_UMAP_BENCHMARK_MIN_DIST),
            metric=str(_UMAP_BENCHMARK_METRIC),
            umap_seed=0,
            seed=int(deco_seed),
            force=bool(getattr(args, "optimops_force", False)),
        )
        _append_decoherence_row(
            sweep=sweep,
            x=x,
            rep=rep,
            method=_UMAP_DECOHERENCE_METHOD_KEY,
            decoherence_radius=float(umap_deco_radius),
            avg_total_counts_per_node=float(umap_deco_payload.get("avg_total_counts_per_node", float("nan"))),
            decoherence_threshold=float(umap_deco_payload.get("threshold", float("nan"))),
            giant_frac=float(giant_frac),
            n0=int(n0),
            n1=int(n1),
            n_cells=int(n_cells),
            node_scale=float(node_scale),
            barrier_ratio=barrier_ratio,
            membrane_strength=float(membrane_strength),
            false_fraction=float(false_fraction),
        )

        if _optimops_register_zf_requested(args):
            try:
                align_summary = render_register_zf_alignment_diagnostics_for_run(Path(run_dir), args)
                if align_summary is None:
                    match_dir = _register_zf_match_dir_for_run(run_dir, _optimops_register_zf_flag(args))
                    align_summary = _collect_register_zf_alignment_summary(match_dir)
                if bool(align_summary.get("register_zf_alignment_available", False)):
                    metric_val = float(align_summary.get("regzf_final_postrank_spearman_mean", float("nan")))
                    row = _base_row(
                        sweep=sweep,
                        x=x,
                        rep=rep,
                        method="register_zf_alignment",
                        metric_value=metric_val,
                        giant_frac=float(giant_frac),
                        n0=int(n0),
                        n1=int(n1),
                        n_cells=int(n_cells),
                        node_scale=float(node_scale),
                        barrier_ratio=barrier_ratio,
                        membrane_strength=float(membrane_strength),
                        false_fraction=float(false_fraction),
                    )
                    row["benchmark_stat"] = _REGZF_ALIGNMENT_STAT_KEY
                    row.update(align_summary)
                    rows.append(row)
            except Exception as e:
                print(f"[WARN] Could not collect register_zf alignment summary for {run_dir}: {e}")

    def _build_and_score(*,
                         run_dir: Path,
                         n0: int,
                         n1: int,
                         n_cells: int,
                         membrane_strength: float,
                         false_edge_frac: float,
                         false_edge_targets_nodes: bool,
                         seed: int,
                         do_viz: bool = False):
        run_dir.mkdir(parents=True, exist_ok=True)

        def _build_run() -> None:
            centers_u, points_u, labels_u = generate_3d_voronoi_positions(
                n0=int(n0),
                n1=int(n1),
                n_cells=int(n_cells),
                cell_jitter=float(args.cell_jitter),
                bbox_lo=bbox_lo,
                bbox_hi=bbox_hi,
                boundary_margin_frac=float(args.boundary_margin_frac),
                hole=bool(args.hole),
                threads=int(args.threads),
                seed=int(seed),
            )

            pos_csv = run_dir / "positions_generated.csv"
            np.savetxt(pos_csv, positions_to_csv_array(points_u, labels_u), delimiter=',', fmt='%.6g')

            if do_viz:
                pos = points_u
                varsum = float(np.sum(np.var(pos, axis=0)))
                scale = (float(args.rescale) / math.sqrt(varsum)) if varsum > 0.0 else 1.0
                sim_pos = pos * scale
                centers = centers_u * scale

                _tmp_tree = cKDTree(centers)
                _dnn, _ = _tmp_tree.query(centers, k=2)
                center_spacing = float(np.median(_dnn[:, 1])) if _dnn.shape[1] > 1 else float(np.median(_dnn))
                if (not np.isfinite(center_spacing)) or center_spacing <= 0:
                    center_spacing = 1.0

                membrane_width = float(args.membrane_width)
                if membrane_width <= 0.0:
                    membrane_width = float(args.membrane_width_frac) * center_spacing
                ecs_width = float(args.ecs_width)
                if ecs_width <= 0.0:
                    ecs_width = float(args.ecs_width_frac) * center_spacing

                field = CellularDiffusionField(
                    centers=centers,
                    D_in=float(args.D_in),
                    D_out=(float(args.D_out) if float(args.D_out) > 0 else float(args.D_in)),
                    ecs_width=float(max(0.0, ecs_width)),
                    D_min=float(args.D_min),
                    D_max=float(args.D_max),
                    qp_modes=int(args.qp_modes),
                    qp_wavelength=(None if float(args.qp_wavelength) <= 0 else float(args.qp_wavelength)),
                    qp_amp=float(args.qp_amp),
                    qp_mode=str(args.qp_mode),
                    cell_sigma=float(args.cell_sigma),
                    cell_q_corr=float(args.cell_q_corr),
                    membrane_width=float(max(0.0, membrane_width)),
                    membrane_strength=float(membrane_strength),
                    seed=int(seed) + 23456,
                )
                _plot_diffusion_slices_3d(field, sim_pos, run_dir / "viz_diffusion_slices.png")

                try:
                    if int(field.d) == 3:
                        _ensure_voronoi_isosurface(
                            centers=centers,
                            sim_pos=sim_pos,
                            labels=labels_u,
                            field=field,
                            outpath=run_dir / "viz_boundary_isosurface.png",
                            D0=float(args.D_in),
                            D_min=float(args.D_min),
                            D_max=float(args.D_max),
                            qp_modes=int(args.qp_modes),
                            qp_amp=float(args.qp_amp),
                            cell_sigma=float(args.cell_sigma),
                            cell_q_corr=float(args.cell_q_corr),
                            membrane_width=float(membrane_width),
                            membrane_width_frac=float(args.membrane_width_frac),
                            membrane_strength=float(membrane_strength),
                            grid_res=int(args.viz_isosurface_grid_res),
                            seed=int(seed),
                            panel_size=tuple(int(x) for x in args.isosurface_panel_size),
                            show_points=bool(args.viz_isosurface_show_points),
                            point_size=float(args.viz_isosurface_point_size),
                        )
                except Exception as e:
                    print(f"[WARN] benchmark viz isosurface failed: {e}")

            build_graph_from_positions_cellular(
                pos_csv=str(pos_csv),
                outdir=run_dir,
                rescale=float(args.rescale),
                mperPt=float(args.mperPt),
                pi_short=float(args.pi_short),
                sigma_s=float(args.sigma_s),
                short_trunc=float(args.short_trunc),
                k_capture=float(args.k_capture),
                long_trunc=float(args.long_trunc),
                long_eps=float(args.long_eps),
                path_samples=int(args.path_samples),
                path_mode=str(args.path_mode),
                max_nbrs_per_source=int(args.max_nbrs_per_source),
                n_cells=int(n_cells),
                cell_jitter=float(args.cell_jitter),
                D_in=float(args.D_in),
                D_out=float(args.D_out),
                ecs_width=float(args.ecs_width),
                ecs_width_frac=float(args.ecs_width_frac),
                P_mem=float(args.P_mem),
                D_min=float(args.D_min),
                D_max=float(args.D_max),
                qp_modes=int(args.qp_modes),
                qp_wavelength=(None if float(args.qp_wavelength) <= 0 else float(args.qp_wavelength)),
                qp_amp=float(args.qp_amp),
                cell_sigma=float(args.cell_sigma),
                cell_q_corr=float(args.cell_q_corr),
                membrane_width=float(args.membrane_width),
                membrane_width_frac=float(args.membrane_width_frac),
                membrane_strength=float(membrane_strength),
                qp_mode=str(args.qp_mode),
                amp_dispersion=float(args.amp_dispersion),
                false_edge_frac=float(false_edge_frac),
                false_edge_targets_nodes=bool(false_edge_targets_nodes),
                write_synthetic_zf_fixtures=bool(getattr(args, "optimops_zf_synthetic_fixtures", False) or _optimops_register_zf_requested(args)),
                synthetic_zf_flag=str(_optimops_register_zf_flag(args) or "18hpf"),
                synthetic_zf_slice_path=(
                    None if not str(getattr(args, "optimops_slice_path", "") or "").strip()
                    else str(getattr(args, "optimops_slice_path"))
                ),
                synthetic_zf_write_slice=bool(getattr(args, "optimops_zf_synthetic_fixtures", False) or not str(getattr(args, "optimops_slice_path", "") or "").strip()),
                synthetic_zf_slice_n=int(getattr(args, "optimops_zf_slice_n", 0)),
                synthetic_zf_num_pole_pairs=int(getattr(args, "optimops_zf_num_pole_pairs", 3)),
                synthetic_zf_genes_per_pole=int(getattr(args, "optimops_zf_genes_per_pole", 3)),
                threads=int(args.threads),
                graph_chunk_size=int(args.graph_chunk_size),
                seed=int(seed),
                voronoi_centers=centers_u,
                # In benchmark sweeps, --viz-isosurfaces already writes the selected
                # low/high diagnostic surfaces above.  Do not also invoke the
                # single-run default renderer for every sweep point, because the
                # matplotlib marching-cubes fallback is slow and looks like a stall.
                render_isosurface=False,
                isosurface_grid_res=int(args.isosurface_grid_res),
                isosurface_show_points=bool(args.isosurface_show_points),
                isosurface_point_size=float(args.isosurface_point_size),
                isosurface_panel_size=tuple(int(x) for x in args.isosurface_panel_size),
            )

        A, true_cell, n0_out, n1_out, giant_frac = _load_or_build_benchmark_run(
            run_dir=run_dir,
            args=args,
            build_fn=_build_run,
        )

        scores = _evaluate_methods_on_run(
            methods,
            A=A,
            true_cell=true_cell,
            n0=int(n0_out),
            n1=int(n1_out),
            method_input=str(args.method_input),
            run_dir=run_dir,
        )
        umap_hdbscan_nmi = _compute_graph_umap_hdbscan_nmi_for_dir(
            _benchmark_embedding_dir_for_run(run_dir, args),
            pca_dim=int(_UMAP_BENCHMARK_PCA_DIM),
            n_neighbors=int(_UMAP_BENCHMARK_N_NEIGHBORS),
            min_dist=float(_UMAP_BENCHMARK_MIN_DIST),
            metric=str(_UMAP_BENCHMARK_METRIC),
            seed=0,
            hdbscan_min_cluster_size=int(_UMAP_HDBSCAN_MIN_CLUSTER_SIZE),
            hdbscan_min_samples=int(_UMAP_HDBSCAN_MIN_SAMPLES),
            force=bool(getattr(args, "optimops_force", False)),
        )

        return giant_frac, scores, float(umap_hdbscan_nmi)

    def _load_or_generate_fixed_geometry(*,
                                         pos_csv_path: Path,
                                         centers_path: Path,
                                         n0: int,
                                         n1: int,
                                         n_cells: int,
                                         seed: int) -> np.ndarray:
        if pos_csv_path.exists() and centers_path.exists():
            return np.asarray(np.load(str(centers_path)), dtype=float)

        centers_u, points_u, labels_u = generate_3d_voronoi_positions(
            n0=int(n0),
            n1=int(n1),
            n_cells=int(n_cells),
            cell_jitter=float(args.cell_jitter),
            bbox_lo=bbox_lo,
            bbox_hi=bbox_hi,
            boundary_margin_frac=float(args.boundary_margin_frac),
            hole=bool(args.hole),
            threads=int(args.threads),
            seed=int(seed),
        )
        np.savetxt(pos_csv_path, positions_to_csv_array(points_u, labels_u), delimiter=',', fmt='%.6g')
        np.save(centers_path, np.asarray(centers_u, dtype=float))
        return centers_u

    def _build_from_fixed_geometry_and_score(*,
                                             run_dir: Path,
                                             pos_csv: Path | str,
                                             n_cells: int,
                                             membrane_strength: float,
                                             false_edge_frac: float,
                                             false_edge_targets_nodes: bool,
                                             seed: int,
                                             voronoi_centers: np.ndarray | None = None,
                                             centers_path: Path | str | None = None):
        run_dir.mkdir(parents=True, exist_ok=True)

        def _build_run() -> None:
            centers_local = voronoi_centers
            if centers_local is None:
                if centers_path is None or (not Path(centers_path).exists()):
                    raise FileNotFoundError(
                        f"Missing cached Voronoi centers for fixed-geometry benchmark run at {run_dir}."
                    )
                centers_local = np.asarray(np.load(str(centers_path)), dtype=float)

            build_graph_from_positions_cellular(
                pos_csv=str(pos_csv),
                outdir=run_dir,
                rescale=float(args.rescale),
                mperPt=float(args.mperPt),
                pi_short=float(args.pi_short),
                sigma_s=float(args.sigma_s),
                short_trunc=float(args.short_trunc),
                k_capture=float(args.k_capture),
                long_trunc=float(args.long_trunc),
                long_eps=float(args.long_eps),
                path_samples=int(args.path_samples),
                path_mode=str(args.path_mode),
                max_nbrs_per_source=int(args.max_nbrs_per_source),
                n_cells=int(n_cells),
                cell_jitter=float(args.cell_jitter),
                D_in=float(args.D_in),
                D_out=float(args.D_out),
                ecs_width=float(args.ecs_width),
                ecs_width_frac=float(args.ecs_width_frac),
                P_mem=float(args.P_mem),
                D_min=float(args.D_min),
                D_max=float(args.D_max),
                qp_modes=int(args.qp_modes),
                qp_wavelength=(None if float(args.qp_wavelength) <= 0 else float(args.qp_wavelength)),
                qp_amp=float(args.qp_amp),
                cell_sigma=float(args.cell_sigma),
                cell_q_corr=float(args.cell_q_corr),
                membrane_width=float(args.membrane_width),
                membrane_width_frac=float(args.membrane_width_frac),
                membrane_strength=float(membrane_strength),
                qp_mode=str(args.qp_mode),
                amp_dispersion=float(args.amp_dispersion),
                false_edge_frac=float(false_edge_frac),
                false_edge_targets_nodes=bool(false_edge_targets_nodes),
                write_synthetic_zf_fixtures=bool(getattr(args, "optimops_zf_synthetic_fixtures", False) or _optimops_register_zf_requested(args)),
                synthetic_zf_flag=str(_optimops_register_zf_flag(args) or "18hpf"),
                synthetic_zf_slice_path=(
                    None if not str(getattr(args, "optimops_slice_path", "") or "").strip()
                    else str(getattr(args, "optimops_slice_path"))
                ),
                synthetic_zf_write_slice=bool(getattr(args, "optimops_zf_synthetic_fixtures", False) or not str(getattr(args, "optimops_slice_path", "") or "").strip()),
                synthetic_zf_slice_n=int(getattr(args, "optimops_zf_slice_n", 0)),
                synthetic_zf_num_pole_pairs=int(getattr(args, "optimops_zf_num_pole_pairs", 3)),
                synthetic_zf_genes_per_pole=int(getattr(args, "optimops_zf_genes_per_pole", 3)),
                threads=int(args.threads),
                graph_chunk_size=int(args.graph_chunk_size),
                seed=int(seed),
                voronoi_centers=centers_local,
                # In benchmark sweeps, --viz-isosurfaces already writes the selected
                # low/high diagnostic surfaces above.  Do not also invoke the
                # single-run default renderer for every sweep point, because the
                # matplotlib marching-cubes fallback is slow and looks like a stall.
                render_isosurface=False,
                isosurface_grid_res=int(args.isosurface_grid_res),
                isosurface_show_points=bool(args.isosurface_show_points),
                isosurface_point_size=float(args.isosurface_point_size),
                isosurface_panel_size=tuple(int(x) for x in args.isosurface_panel_size),
            )

        A, true_cell, n0_out, n1_out, giant_frac = _load_or_build_benchmark_run(
            run_dir=run_dir,
            args=args,
            build_fn=_build_run,
        )
        scores = _evaluate_methods_on_run(
            methods,
            A=A,
            true_cell=true_cell,
            n0=int(n0_out),
            n1=int(n1_out),
            method_input=str(args.method_input),
            run_dir=run_dir,
        )
        umap_hdbscan_nmi = _compute_graph_umap_hdbscan_nmi_for_dir(
            _benchmark_embedding_dir_for_run(run_dir, args),
            pca_dim=int(_UMAP_BENCHMARK_PCA_DIM),
            n_neighbors=int(_UMAP_BENCHMARK_N_NEIGHBORS),
            min_dist=float(_UMAP_BENCHMARK_MIN_DIST),
            metric=str(_UMAP_BENCHMARK_METRIC),
            seed=0,
            hdbscan_min_cluster_size=int(_UMAP_HDBSCAN_MIN_CLUSTER_SIZE),
            hdbscan_min_samples=int(_UMAP_HDBSCAN_MIN_SAMPLES),
            force=bool(getattr(args, "optimops_force", False)),
        )
        return giant_frac, scores, float(umap_hdbscan_nmi), int(n0_out), int(n1_out)

    def _unique_numeric(df: pd.DataFrame, col: str) -> np.ndarray:
        if col not in df.columns:
            return np.asarray([], dtype=float)
        return np.sort(pd.to_numeric(df[col], errors="coerce").dropna().astype(float).unique())

    def _numeric_values_match(existing: np.ndarray, expected: list[float]) -> bool:
        exp = np.sort(np.asarray(expected, dtype=float))
        ex = np.sort(np.asarray(existing, dtype=float))
        return ex.shape == exp.shape and np.allclose(ex, exp, rtol=1e-9, atol=1e-12)

    bench_root = outdir / "benchmark"
    bench_root.mkdir(parents=True, exist_ok=True)

    existing_csv: Optional[Path] = None
    for cand in (bench_root / "benchmark_results.csv", outdir / "benchmark_results.csv"):
        if cand.exists():
            existing_csv = cand
            break

    required_sweeps = {"joint_density"}
    if run_barrier_sweep:
        required_sweeps.add("cell_density_barrier_ratio")
    if run_false_sweep:
        required_sweeps.add("cell_density_false_fraction")
    required_metric_groups = {
        _BENCHMARK_METHOD_STAT_KEY,
        _BENCHMARK_UMAP_HDBSCAN_STAT_KEY,
        _BENCHMARK_DECOHERENCE_RADIUS_STAT_KEY,
    }
    required_benchmark_layout_version = int(_BENCHMARK_LAYOUT_VERSION)
    reuse_existing = False

    if existing_csv is not None:
        try:
            _df_existing = pd.read_csv(existing_csv)
            issues: list[str] = []

            if _BENCHMARK_METRIC_KEY not in _df_existing.columns:
                if "ARI" in _df_existing.columns:
                    issues.append(
                        f"they use ARI scores instead of {_BENCHMARK_METRIC_KEY}"
                    )
                else:
                    issues.append(f"they are missing required column {_BENCHMARK_METRIC_KEY!r}")

            available_sweeps = set(_df_existing["sweep"].astype(str)) if "sweep" in _df_existing.columns else set()
            missing_sweeps = sorted(required_sweeps - available_sweeps)
            if missing_sweeps:
                issues.append(f"they are missing required sweeps {missing_sweeps}")

            available_metric_groups = (
                set(_df_existing["benchmark_stat"].astype(str))
                if "benchmark_stat" in _df_existing.columns
                else ({_BENCHMARK_METHOD_STAT_KEY} if _BENCHMARK_METRIC_KEY in _df_existing.columns else set())
            )
            missing_metric_groups = sorted(required_metric_groups - available_metric_groups)
            if missing_metric_groups:
                issues.append(f"they are missing required benchmark stat groups {missing_metric_groups}")

            available_nmi_methods = (
                set(
                    _df_existing.loc[
                        _df_existing["benchmark_stat"].astype(str).isin([
                            _BENCHMARK_METHOD_STAT_KEY,
                            _BENCHMARK_UMAP_HDBSCAN_STAT_KEY,
                        ]),
                        "method",
                    ].astype(str)
                )
                if {"benchmark_stat", "method"}.issubset(_df_existing.columns)
                else set()
            )
            missing_nmi_methods = [m for m in required_nmi_methods if m not in available_nmi_methods]
            if missing_nmi_methods:
                issues.append(f"they are missing required NMI methods {missing_nmi_methods}")

            available_decoherence_methods = (
                set(
                    _df_existing.loc[
                        _df_existing["benchmark_stat"].astype(str) == _BENCHMARK_DECOHERENCE_RADIUS_STAT_KEY,
                        "method",
                    ].astype(str)
                )
                if {"benchmark_stat", "method"}.issubset(_df_existing.columns)
                else set()
            )
            missing_decoherence_methods = sorted(
                {"gse_embedding_decoherence", _UMAP_DECOHERENCE_METHOD_KEY} - available_decoherence_methods
            )
            if missing_decoherence_methods:
                issues.append(f"they are missing required decoherence methods {missing_decoherence_methods}")

            layout_versions = (
                pd.to_numeric(_df_existing["benchmark_layout_version"], errors="coerce").dropna().astype(int).unique()
                if "benchmark_layout_version" in _df_existing.columns else np.asarray([], dtype=int)
            )
            if layout_versions.size != 1 or int(layout_versions[0]) != required_benchmark_layout_version:
                issues.append("they were generated with an older benchmark layout/version")

            existing_false_modes = (
                set(_df_existing["false_mode"].dropna().astype(str).unique())
                if "false_mode" in _df_existing.columns else set()
            )
            if existing_false_modes != {false_mode}:
                issues.append(
                    f"they use false_mode={sorted(existing_false_modes) if existing_false_modes else 'missing'} "
                    f"instead of false_mode={false_mode!r}"
                )
            joint_existing = _df_existing[_df_existing["sweep"].astype(str) == "joint_density"] if "sweep" in _df_existing.columns else _df_existing.iloc[0:0]
            if not _numeric_values_match(_unique_numeric(joint_existing, "false_fraction"), [base_false_frac]):
                issues.append(
                    f"joint_density rows use false_fraction={_unique_numeric(joint_existing, 'false_fraction').tolist() or 'missing'} "
                    f"instead of [{base_false_frac:g}]"
                )

            if run_barrier_sweep:
                barrier_existing = _df_existing[_df_existing["sweep"].astype(str) == "cell_density_barrier_ratio"]
                if not _numeric_values_match(_unique_numeric(barrier_existing, "barrier_ratio"), barrier_ratios):
                    issues.append(
                        f"cell_density_barrier_ratio rows use barrier_ratio={_unique_numeric(barrier_existing, 'barrier_ratio').tolist() or 'missing'} "
                        f"instead of {barrier_ratios}"
                    )
                if not _numeric_values_match(_unique_numeric(barrier_existing, "false_fraction"), [base_false_frac]):
                    issues.append(
                        f"cell_density_barrier_ratio rows use false_fraction={_unique_numeric(barrier_existing, 'false_fraction').tolist() or 'missing'} "
                        f"instead of [{base_false_frac:g}]"
                    )

            if run_false_sweep:
                false_existing = _df_existing[_df_existing["sweep"].astype(str) == "cell_density_false_fraction"]
                if not _numeric_values_match(_unique_numeric(false_existing, "false_fraction"), false_fracs):
                    issues.append(
                        f"cell_density_false_fraction rows use false_fraction={_unique_numeric(false_existing, 'false_fraction').tolist() or 'missing'} "
                        f"instead of {false_fracs}"
                    )

            if issues:
                print(f"[INFO] Found existing benchmark results at: {existing_csv}, but " + issues[0] + "; rerunning benchmark.")
            else:
                rows = _df_existing.to_dict(orient="records")
                reps = 0
                reuse_existing = True
                print(f"[INFO] Found existing benchmark results at: {existing_csv}")
                print("[INFO] Skipping benchmark analysis and regenerating requested heatmaps only.")
        except Exception as e:
            print(f"[WARN] Could not read existing results CSV ({existing_csv}); will rerun analysis. ({e})")

    lo_ncells = int(min(n_cells_list))
    hi_ncells = int(max(n_cells_list))
    lo_sc = float(min(node_scales))
    hi_sc = float(max(node_scales))

    for rep in range(reps):
        seed_rep = base_seed + 1000 * rep + 22
        for nc in n_cells_list:
            seed_geom = int(seed_rep + 37 * int(nc) + 7)
            for sc in node_scales:
                n0 = int(max(10, round(base_n0 * float(sc))))
                n1 = int(max(10, round(base_n1 * float(sc))))
                run_dir = (bench_root / f"joint_density_rep{rep:02d}"
                           / f"ncells_{int(nc)}"
                           / f"scale_{float(sc):.6g}")
                if bool(args.hole):
                    run_dir = run_dir / "hole"
                run_dir = run_dir / _corruption_dir(base_false_frac)

                do_viz = bool(args.viz_isosurfaces) and (rep == 0) and (int(nc) in {lo_ncells, hi_ncells}) and (
                    abs(float(sc) - lo_sc) < 1e-12 or abs(float(sc) - hi_sc) < 1e-12
                )

                giant_frac, scores, umap_hdbscan_nmi = _build_and_score(
                    run_dir=run_dir,
                    n0=n0,
                    n1=n1,
                    n_cells=int(nc),
                    membrane_strength=float(args.membrane_strength),
                    false_edge_frac=base_false_frac,
                    false_edge_targets_nodes=bool(args.false_target_nodes),
                    seed=seed_geom,
                    do_viz=do_viz,
                )

                _append_run_metric_rows(
                    run_dir=run_dir,
                    sweep="joint_density",
                    x=float(n0 + n1),
                    rep=int(rep),
                    scores=scores,
                    umap_hdbscan_nmi=float(umap_hdbscan_nmi),
                    giant_frac=float(giant_frac),
                    n0=int(n0),
                    n1=int(n1),
                    n_cells=int(nc),
                    node_scale=float(sc),
                    barrier_ratio=None,
                    membrane_strength=float(args.membrane_strength),
                    false_fraction=base_false_frac,
                    deco_seed=int(seed_geom) + 50001,
                )

    if run_barrier_sweep:
        for rep in range(reps):
            seed_rep = base_seed + 1000 * rep + 33
            for nc in n_cells_list:
                seed_geom = int(seed_rep + 53 * int(nc) + 11)
                sweep_dir = (bench_root / f"cell_density_barrier_ratio_rep{rep:02d}"
                             / f"ncells_{int(nc)}")
                if bool(args.hole):
                    sweep_dir = sweep_dir / "hole"
                sweep_dir = sweep_dir / _corruption_dir(base_false_frac)
                sweep_dir.mkdir(parents=True, exist_ok=True)

                pos_csv_rep = sweep_dir / "positions_generated.csv"
                centers_rep = sweep_dir / "voronoi_centers_generated.npy"

                run_dirs = [sweep_dir / f"br_{float(br):.6g}" for br in barrier_ratios]
                need_geometry = any(not _run_has_reusable_optimops_results(rd, args) for rd in run_dirs)

                centers_u: np.ndarray | None = None
                if need_geometry:
                    centers_u = _load_or_generate_fixed_geometry(
                        pos_csv_path=pos_csv_rep,
                        centers_path=centers_rep,
                        n0=base_n0,
                        n1=base_n1,
                        n_cells=int(nc),
                        seed=seed_geom,
                    )

                for br in barrier_ratios:
                    beta = _membrane_strength_for_barrier_ratio(
                        barrier_ratio=float(br),
                        D_in=float(args.D_in),
                        D_out=float(args.D_out),
                    )
                    run_dir = sweep_dir / f"br_{float(br):.6g}"

                    giant_frac, scores, umap_hdbscan_nmi, n0_out, n1_out = _build_from_fixed_geometry_and_score(
                        run_dir=run_dir,
                        pos_csv=pos_csv_rep,
                        n_cells=int(nc),
                        membrane_strength=float(beta),
                        false_edge_frac=base_false_frac,
                        false_edge_targets_nodes=bool(args.false_target_nodes),
                        seed=int(seed_geom) + 123,
                        voronoi_centers=centers_u,
                        centers_path=centers_rep,
                    )

                    _append_run_metric_rows(
                        run_dir=run_dir,
                        sweep="cell_density_barrier_ratio",
                        x=float(br),
                        rep=int(rep),
                        scores=scores,
                        umap_hdbscan_nmi=float(umap_hdbscan_nmi),
                        giant_frac=float(giant_frac),
                        n0=int(n0_out),
                        n1=int(n1_out),
                        n_cells=int(nc),
                        node_scale=1.0,
                        barrier_ratio=float(br),
                        membrane_strength=float(beta),
                        false_fraction=base_false_frac,
                        deco_seed=int(seed_geom) + 50001,
                    )

    if run_false_sweep:
        for rep in range(reps):
            seed_rep = base_seed + 1000 * rep + 44
            for nc in n_cells_list:
                seed_geom = int(seed_rep + 53 * int(nc) + 11)
                sweep_dir = (bench_root / f"cell_density_false_fraction_rep{rep:02d}"
                             / f"ncells_{int(nc)}")
                if bool(args.hole):
                    sweep_dir = sweep_dir / "hole"
                sweep_dir = sweep_dir / _corruption_dir()
                sweep_dir.mkdir(parents=True, exist_ok=True)

                pos_csv_rep = sweep_dir / "positions_generated.csv"
                centers_rep = sweep_dir / "voronoi_centers_generated.npy"

                run_dirs = [sweep_dir / f"ff_{float(ff):.6g}" for ff in false_fracs]
                need_geometry = any(not _run_has_reusable_optimops_results(rd, args) for rd in run_dirs)

                centers_u: np.ndarray | None = None
                if need_geometry:
                    centers_u = _load_or_generate_fixed_geometry(
                        pos_csv_path=pos_csv_rep,
                        centers_path=centers_rep,
                        n0=base_n0,
                        n1=base_n1,
                        n_cells=int(nc),
                        seed=seed_geom,
                    )

                for ff in false_fracs:
                    run_dir = sweep_dir / f"ff_{float(ff):.6g}"

                    giant_frac, scores, umap_hdbscan_nmi, n0_out, n1_out = _build_from_fixed_geometry_and_score(
                        run_dir=run_dir,
                        pos_csv=pos_csv_rep,
                        n_cells=int(nc),
                        membrane_strength=float(args.membrane_strength),
                        false_edge_frac=float(ff),
                        false_edge_targets_nodes=bool(args.false_target_nodes),
                        seed=int(seed_geom) + 123,
                        voronoi_centers=centers_u,
                        centers_path=centers_rep,
                    )

                    _append_run_metric_rows(
                        run_dir=run_dir,
                        sweep="cell_density_false_fraction",
                        x=float(ff),
                        rep=int(rep),
                        scores=scores,
                        umap_hdbscan_nmi=float(umap_hdbscan_nmi),
                        giant_frac=float(giant_frac),
                        n0=int(n0_out),
                        n1=int(n1_out),
                        n_cells=int(nc),
                        node_scale=1.0,
                        barrier_ratio=None,
                        membrane_strength=float(args.membrane_strength),
                        false_fraction=float(ff),
                        deco_seed=int(seed_geom) + 50001,
                    )

    df = pd.DataFrame(rows)
    out_csv = outdir / "benchmark_results.csv"
    if not reuse_existing:
        df.to_csv(out_csv, index=False)
        try:
            df.to_csv(bench_root / "benchmark_results.csv", index=False)
        except Exception:
            pass
    else:
        out_csv = existing_csv if existing_csv is not None else out_csv

    def _plot_benchmark_heatmaps(dsub: pd.DataFrame,
                                 out_png: Path,
                                 *,
                                 metric_col: str,
                                 metric_label: str,
                                 x_col: str,
                                 y_col: str,
                                 x_values: list[Any],
                                 y_values: list[Any],
                                 xticklabels: list[str],
                                 yticklabels: list[str],
                                 xlabel: str,
                                 ylabel: str,
                                 title: str,
                                 vmin: float | None = None,
                                 vmax: float | None = None,
                                 norm: "matplotlib.colors.Normalize | None" = None,
                                 cmap: str = "cividis",
                                 annot_fmt: str = ".2f",
                                 method_order: list[str] | None = None) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe

        _mpl_set_arial_fonts(base_size=13.0)
        if dsub.empty:
            return

        vals = pd.to_numeric(dsub[metric_col], errors="coerce")
        dsub = dsub.loc[np.isfinite(vals)].copy()
        dsub[metric_col] = vals.loc[dsub.index].astype(float)
        if dsub.empty:
            return

        g = (dsub.groupby([y_col, x_col, "method"], as_index=False)[metric_col]
             .agg(mean="mean"))
        if g.empty:
            return

        finite_vals = pd.to_numeric(g["mean"], errors="coerce")
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        if finite_vals.empty:
            return

        unique_methods = list(g["method"].astype(str).unique())
        if method_order:
            methods_sorted = [m for m in method_order if m in unique_methods]
            methods_sorted.extend([m for m in sorted(unique_methods) if m not in methods_sorted])
        else:
            methods_sorted = list(sorted(unique_methods))
        n_methods = max(1, len(methods_sorted))
        n_rows = len(y_values) if y_values else 1
        n_cols = len(x_values) if x_values else 1
        cell_w, cell_h = 0.65, 0.65
        panel_w = max(3.0, cell_w * n_cols)
        panel_h = max(2.0, cell_h * n_rows)
        fig, axes = plt.subplots(
            1, n_methods,
            figsize=(panel_w * n_methods + 1.6, panel_h + 1.8),
            squeeze=False,
            gridspec_kw={"wspace": 0.30},
        )

        vmin_eff = float(vmin) if vmin is not None else float(finite_vals.min())
        vmax_eff = float(vmax) if vmax is not None else float(finite_vals.max())
        if not np.isfinite(vmin_eff):
            vmin_eff = float(finite_vals.min())
        if not np.isfinite(vmax_eff):
            vmax_eff = float(finite_vals.max())
        if abs(vmax_eff - vmin_eff) <= 1e-12:
            delta = max(1e-6, 0.05 * max(1.0, abs(vmax_eff)))
            vmin_eff -= delta
            vmax_eff += delta

        if norm is not None:
            norm.vmin = vmin_eff
            norm.vmax = vmax_eff

        annot_fs = float(plt.rcParams.get("xtick.labelsize", 11.0))
        peff = [pe.Stroke(linewidth=3, foreground="black", alpha=0.35), pe.Normal()]
        x_to_j = {_benchmark_value_key(x): j for j, x in enumerate(x_values)}
        y_to_i = {_benchmark_value_key(y): i for i, y in enumerate(y_values)}

        last_im = None
        for ax, mname in zip(axes[0], methods_sorted):
            mat = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
            dd = g[g["method"].astype(str) == str(mname)]
            for _, row in dd.iterrows():
                xv_key = _benchmark_value_key(row[x_col])
                yv_key = _benchmark_value_key(row[y_col])
                if xv_key in x_to_j and yv_key in y_to_i:
                    mat[y_to_i[yv_key], x_to_j[xv_key]] = float(row["mean"])

            im_kwargs = dict(
                origin="lower",
                aspect="equal",
                cmap=str(cmap),
                interpolation="nearest",
            )
            if norm is not None:
                im_kwargs["norm"] = norm
            else:
                im_kwargs["vmin"] = vmin_eff
                im_kwargs["vmax"] = vmax_eff

            im = ax.imshow(mat, **im_kwargs)
            last_im = im

            ax.set_title(_m_label(mname), fontsize=9)
            ax.set_xticks(range(len(x_values)))
            ax.set_xticklabels(xticklabels, rotation=0)
            ax.set_yticks(range(len(y_values)))
            ax.set_yticklabels(yticklabels)

            ax.set_xlabel(xlabel, fontsize=9)
            if ax is axes[0][0]:
                ax.set_ylabel(ylabel, fontsize=9)

            if len(y_values) * len(x_values) <= 80:
                for i in range(len(y_values)):
                    for j in range(len(x_values)):
                        if np.isfinite(mat[i, j]):
                            ax.text(
                                j,
                                i,
                                format(float(mat[i, j]), annot_fmt),
                                ha="center",
                                va="center",
                                fontsize=annot_fs,
                                color="white",
                                path_effects=peff,
                            )

        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.90, pad=0.02)
            cbar.set_label(f"mean {metric_label}")

        fig.suptitle(title, y=1.02)
        fig.savefig(str(out_png), dpi=300, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)


    def _plot_register_zf_metric_heatmaps(dsub: pd.DataFrame,
                                          out_prefix: Path,
                                          *,
                                          x_col: str,
                                          y_col: str,
                                          x_values: list[Any],
                                          y_values: list[Any],
                                          xticklabels: list[str],
                                          yticklabels: list[str],
                                          xlabel: str,
                                          ylabel: str,
                                          title_prefix: str) -> None:
        if "benchmark_stat" not in dsub.columns:
            return
        align_df = dsub[dsub["benchmark_stat"].astype(str) == _REGZF_ALIGNMENT_STAT_KEY].copy()
        if align_df.empty:
            return
        if "regzf_final_postrank_spearman_mean" in align_df.columns:
            _plot_benchmark_heatmaps(
                align_df,
                Path(str(out_prefix) + "_register_zf_alignment_spearman.png"),
                metric_col="regzf_final_postrank_spearman_mean",
                metric_label="register_zf final-vs-slice post-rank Spearman",
                x_col=x_col,
                y_col=y_col,
                x_values=x_values,
                y_values=y_values,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title_prefix + " — register_zf spatial feature agreement",
                vmin=-1.0,
                vmax=1.0,
                cmap="coolwarm",
                annot_fmt=".2f",
                method_order=["register_zf_alignment"],
            )
        if "regzf_fraction_reassigned_by_graph_refinement" in align_df.columns:
            _plot_benchmark_heatmaps(
                align_df,
                Path(str(out_prefix) + "_register_zf_refinement_fraction.png"),
                metric_col="regzf_fraction_reassigned_by_graph_refinement",
                metric_label="fraction reassigned by graph refinement",
                x_col=x_col,
                y_col=y_col,
                x_values=x_values,
                y_values=y_values,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title_prefix + " — register_zf graph-refinement movement",
                vmin=0.0,
                vmax=1.0,
                cmap="magma",
                annot_fmt=".2f",
                method_order=["register_zf_alignment"],
            )
        if "regzf_ensemble_coord_sd_mean" in align_df.columns:
            _plot_benchmark_heatmaps(
                align_df,
                Path(str(out_prefix) + "_register_zf_ensemble_spread.png"),
                metric_col="regzf_ensemble_coord_sd_mean",
                metric_label="mean ensemble coordinate SD",
                x_col=x_col,
                y_col=y_col,
                x_values=x_values,
                y_values=y_values,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title_prefix + " — register_zf ensemble assignment spread",
                cmap="viridis",
                annot_fmt=".2g",
                method_order=["register_zf_alignment"],
            )

    nmi_stat_groups = {_BENCHMARK_METHOD_STAT_KEY, _BENCHMARK_UMAP_HDBSCAN_STAT_KEY}

    # --- Global color scale for decoherence radius across all sweep plots ---
    all_deco_df = df[df["benchmark_stat"] == _BENCHMARK_DECOHERENCE_RADIUS_STAT_KEY].copy()
    _deco_vals = pd.to_numeric(all_deco_df[_BENCHMARK_DECOHERENCE_RADIUS_KEY], errors="coerce")
    _deco_vals = _deco_vals[np.isfinite(_deco_vals)]
    if _deco_vals.empty:
        _deco_vmin, _deco_vmax = 0.0, 1.0
    else:
        _deco_vmin = np.floor(10*float(np.percentile(_deco_vals, 10)))/10.0
        _deco_vmax = np.ceil(10*float(np.percentile(_deco_vals, 90)))/10.0

    joint_df = df[df["sweep"] == "joint_density"].copy()
    joint_nmi_df = joint_df[joint_df["benchmark_stat"].isin(nmi_stat_groups)].copy()
    joint_deco_df = joint_df[joint_df["benchmark_stat"] == _BENCHMARK_DECOHERENCE_RADIUS_STAT_KEY].copy()
    joint_node_scales = list(node_scales)
    joint_n_cells = list(n_cells_list)
    joint_xticklabels = []
    for sc in joint_node_scales:
        total_nodes = int(max(10, round(base_n0 * float(sc))) + max(10, round(base_n1 * float(sc))))
        joint_xticklabels.append(f"{float(sc):g}×\n({total_nodes})")

    _plot_benchmark_heatmaps(
        joint_nmi_df,
        outdir / "benchmark_plot_joint_density_NMI.png",
        metric_col=_BENCHMARK_METRIC_KEY,
        metric_label=_BENCHMARK_METRIC_LABEL,
        x_col="node_scale",
        y_col="n_cells",
        x_values=joint_node_scales,
        y_values=joint_n_cells,
        xticklabels=joint_xticklabels,
        yticklabels=[str(int(nc)) for nc in joint_n_cells],
        xlabel="Node density (scale × total nodes)",
        ylabel="Cell density (n_cells)",
        title=f"Node-density × cell-density sweep — mean {_BENCHMARK_METRIC_KEY}",
        vmin=0.0,
        vmax=1.0,
        cmap="cividis",
        annot_fmt=".2f",
        method_order=required_nmi_methods,
    )
    _plot_benchmark_heatmaps(
        joint_deco_df,
        outdir / "benchmark_plot_joint_density_decoherence_radius.png",
        metric_col=_BENCHMARK_DECOHERENCE_RADIUS_KEY,
        metric_label=_BENCHMARK_DECOHERENCE_RADIUS_LABEL,
        x_col="node_scale",
        y_col="n_cells",
        x_values=joint_node_scales,
        y_values=joint_n_cells,
        xticklabels=joint_xticklabels,
        yticklabels=[str(int(nc)) for nc in joint_n_cells],
        xlabel="Node density (scale × total nodes)",
        ylabel="Cell density (n_cells)",
        title=(
            "Node-density × cell-density sweep — decoherence radius "
            "(mean RMSD > 1/sqrt(avg counts per node))"
        ),
        cmap="turbo",
        annot_fmt=".2g",
        vmin=_deco_vmin,
        vmax=_deco_vmax,
    )
    _plot_register_zf_metric_heatmaps(
        joint_df,
        outdir / "benchmark_plot_joint_density",
        x_col="node_scale",
        y_col="n_cells",
        x_values=joint_node_scales,
        y_values=joint_n_cells,
        xticklabels=joint_xticklabels,
        yticklabels=[str(int(nc)) for nc in joint_n_cells],
        xlabel="Node density (scale × total nodes)",
        ylabel="Cell density (n_cells)",
        title_prefix="Node-density × cell-density sweep",
    )

    if run_barrier_sweep:
        cell_barrier_df = df[df["sweep"] == "cell_density_barrier_ratio"].copy()
        cell_barrier_nmi_df = cell_barrier_df[cell_barrier_df["benchmark_stat"].isin(nmi_stat_groups)].copy()
        cell_barrier_deco_df = cell_barrier_df[
            cell_barrier_df["benchmark_stat"] == _BENCHMARK_DECOHERENCE_RADIUS_STAT_KEY
        ].copy()
        barrier_values = list(barrier_ratios)
        barrier_n_cells = list(n_cells_list)

        _plot_benchmark_heatmaps(
            cell_barrier_nmi_df,
            outdir / "benchmark_plot_cell_density_barrier_ratio_NMI.png",
            metric_col=_BENCHMARK_METRIC_KEY,
            metric_label=_BENCHMARK_METRIC_LABEL,
            x_col="barrier_ratio",
            y_col="n_cells",
            x_values=barrier_values,
            y_values=barrier_n_cells,
            xticklabels=[f"{float(br):g}" for br in barrier_values],
            yticklabels=[str(int(nc)) for nc in barrier_n_cells],
            xlabel="Membrane/Intracellular diffusivity ratio (D_mem / D_in)",
            ylabel="Cell density (n_cells)",
            title=f"Cell-density × barrier-ratio sweep — mean {_BENCHMARK_METRIC_KEY}",
            vmin=0.0,
            vmax=1.0,
            cmap="cividis",
            annot_fmt=".2f",
            method_order=required_nmi_methods,
        )
        _plot_benchmark_heatmaps(
            cell_barrier_deco_df,
            outdir / "benchmark_plot_cell_density_barrier_ratio_decoherence_radius.png",
            metric_col=_BENCHMARK_DECOHERENCE_RADIUS_KEY,
            metric_label=_BENCHMARK_DECOHERENCE_RADIUS_LABEL,
            x_col="barrier_ratio",
            y_col="n_cells",
            x_values=barrier_values,
            y_values=barrier_n_cells,
            xticklabels=[f"{float(br):g}" for br in barrier_values],
            yticklabels=[str(int(nc)) for nc in barrier_n_cells],
            xlabel="Membrane/Intracellular diffusivity ratio (D_mem / D_in)",
            ylabel="Cell density (n_cells)",
            title=(
                "Cell-density × barrier-ratio sweep — decoherence radius "
                "(mean RMSD > 1/sqrt(avg counts per node))"
            ),
            cmap="turbo",
            annot_fmt=".2g",
            vmin=_deco_vmin,
            vmax=_deco_vmax,
        )
        _plot_register_zf_metric_heatmaps(
            cell_barrier_df,
            outdir / "benchmark_plot_cell_density_barrier_ratio",
            x_col="barrier_ratio",
            y_col="n_cells",
            x_values=barrier_values,
            y_values=barrier_n_cells,
            xticklabels=[f"{float(br):g}" for br in barrier_values],
            yticklabels=[str(int(nc)) for nc in barrier_n_cells],
            xlabel="Membrane/Intracellular diffusivity ratio (D_mem / D_in)",
            ylabel="Cell density (n_cells)",
            title_prefix="Cell-density × barrier-ratio sweep",
        )

    if run_false_sweep:
        cell_false_df = df[df["sweep"] == "cell_density_false_fraction"].copy()
        cell_false_nmi_df = cell_false_df[cell_false_df["benchmark_stat"].isin(nmi_stat_groups)].copy()
        cell_false_deco_df = cell_false_df[
            cell_false_df["benchmark_stat"] == _BENCHMARK_DECOHERENCE_RADIUS_STAT_KEY
        ].copy()
        false_values = list(false_fracs)
        false_n_cells = list(n_cells_list)
        false_xlabel = "False-node fraction" if bool(args.false_target_nodes) else "False-edge fraction"
        false_title = (
            f"Cell-density × false-node-fusion sweep — mean {_BENCHMARK_METRIC_KEY}"
            if bool(args.false_target_nodes)
            else f"Cell-density × false-edge sweep — mean {_BENCHMARK_METRIC_KEY}"
        )
        false_deco_title = (
            "Cell-density × false-node-fusion sweep — decoherence radius "
            "(mean RMSD > 1/sqrt(avg counts per node))"
            if bool(args.false_target_nodes)
            else "Cell-density × false-edge sweep — decoherence radius "
                 "(mean RMSD > 1/sqrt(avg counts per node))"
        )

        _plot_benchmark_heatmaps(
            cell_false_nmi_df,
            outdir / "benchmark_plot_cell_density_false_fraction_NMI.png",
            metric_col=_BENCHMARK_METRIC_KEY,
            metric_label=_BENCHMARK_METRIC_LABEL,
            x_col="false_fraction",
            y_col="n_cells",
            x_values=false_values,
            y_values=false_n_cells,
            xticklabels=[f"{float(ff):g}" for ff in false_values],
            yticklabels=[str(int(nc)) for nc in false_n_cells],
            xlabel=false_xlabel,
            ylabel="Cell density (n_cells)",
            title=false_title,
            vmin=0.0,
            vmax=1.0,
            cmap="cividis",
            annot_fmt=".2f",
            method_order=required_nmi_methods,
        )
        _plot_benchmark_heatmaps(
            cell_false_deco_df,
            outdir / "benchmark_plot_cell_density_false_fraction_decoherence_radius.png",
            metric_col=_BENCHMARK_DECOHERENCE_RADIUS_KEY,
            metric_label=_BENCHMARK_DECOHERENCE_RADIUS_LABEL,
            x_col="false_fraction",
            y_col="n_cells",
            x_values=false_values,
            y_values=false_n_cells,
            xticklabels=[f"{float(ff):g}" for ff in false_values],
            yticklabels=[str(int(nc)) for nc in false_n_cells],
            xlabel=false_xlabel,
            ylabel="Cell density (n_cells)",
            title=false_deco_title,
            cmap="turbo",
            annot_fmt=".2g",
            vmin=_deco_vmin,
            vmax=_deco_vmax,
        )
        _plot_register_zf_metric_heatmaps(
            cell_false_df,
            outdir / "benchmark_plot_cell_density_false_fraction",
            x_col="false_fraction",
            y_col="n_cells",
            x_values=false_values,
            y_values=false_n_cells,
            xticklabels=[f"{float(ff):g}" for ff in false_values],
            yticklabels=[str(int(nc)) for nc in false_n_cells],
            xlabel=false_xlabel,
            ylabel="Cell density (n_cells)",
            title_prefix=false_title.rsplit(" — ", 1)[0],
        )

    print(f"[INFO] Wrote benchmark results: {out_csv}")
    print(f"[INFO] Wrote plots into: {outdir}")

# ─────────────────────────────── CLI ───────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build a simulated UEI association graph (graph.npz) and benchmark segmentation methods from node positions."
    )

    ap.add_argument("-o", "--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--build-from-posfile", help="CSV with columns id,label,x[,y[,z...]]")

    ap.add_argument("--n0", type=int, help="If --build-from-posfile is omitted, generate this many type-0 nodes.")
    ap.add_argument("--n1", type=int, help="If --build-from-posfile is omitted, generate this many type-1 nodes.")
    ap.add_argument("--bbox-lo", type=float, nargs=3, metavar=("X0", "Y0", "Z0"),
                    help="Lower corner of 3D bbox for generated positions (default: 0 0 0).")
    ap.add_argument("--bbox-hi", type=float, nargs=3, metavar=("X1", "Y1", "Z1"),
                    help="Upper corner of 3D bbox for generated positions (default: 10 10 10).")
    ap.add_argument("--boundary-margin-frac", type=float, default=0.20,
                    help="Reject generated points near the Voronoi boundary (fraction of half NN spacing).")
    ap.add_argument("--generated-positions-csv", default="",
                    help="Optional path to write the generated positions CSV (default: <outdir>/positions_generated.csv).")
    ap.add_argument("--hole", action="store_true",
                    help="When auto-generating positions, exclude a centered cylindrical hole running through the full z-span of the bbox. "
                         "Voronoi centers and diffusion remain unchanged; only node placement is affected.")

    ap.add_argument("--render-isosurface",
                    action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Render Voronoi boundary isosurface into <outdir>/voronoi_isosurface.png "
                         "(default: on; use --no-render-isosurface to disable).")
    ap.add_argument("--isosurface-grid-res", type=int, default=110)
    ap.add_argument("--isosurface-show-points", action="store_true")
    ap.add_argument("--isosurface-point-size", type=float, default=4.0)
    ap.add_argument("--isosurface-panel-size", type=int, nargs=2, default=(1400, 1100), metavar=("W", "H"))

    ap.add_argument("--rescale", type=float, default=2.0)
    ap.add_argument("--mperPt", type=float, default=50.0)
    ap.add_argument("--amp-dispersion", type=float, default=0.0)

    ap.add_argument("--n-cells", type=int, default=300)
    ap.add_argument("--cell-jitter", type=float, default=0.25)
    ap.add_argument("--D-in", dest="D_in", type=float, default=1.0)
    ap.add_argument("--D-out", dest="D_out", type=float, default=0.0)
    ap.add_argument("--ecs-width", dest="ecs_width", type=float, default=0.0)
    ap.add_argument("--ecs-width-frac", dest="ecs_width_frac", type=float, default=0.0)
    ap.add_argument("--P-mem", dest="P_mem", type=float, default=0.0)
    ap.add_argument("--D-min", dest="D_min", type=float, default=0.05)
    ap.add_argument("--D-max", dest="D_max", type=float, default=10.0)
    ap.add_argument("--k-capture", type=float, default=1.0)
    ap.add_argument("--pi-short", type=float, default=0.35)
    ap.add_argument("--sigma-s", type=float, default=0.35)
    ap.add_argument("--short-trunc", type=float, default=4.0)
    ap.add_argument("--long-trunc", type=float, default=4.0)
    ap.add_argument("--long-eps", type=float, default=0.05)
    ap.add_argument("--path-samples", type=int, default=5)
    ap.add_argument("--path-mode", choices=["sample", "boundaryaware", "endpoint"], default="sample")
    ap.add_argument("--max-nbrs-per-source", type=int, default=0)

    ap.add_argument("--qp-modes", type=int, default=6)
    ap.add_argument("--qp-wavelength", type=float, default=0.0)
    ap.add_argument("--qp-amp", type=float, default=0.7)
    ap.add_argument("--qp-mode", choices=["global", "cell"], default="cell")
    ap.add_argument("--cell-sigma", type=float, default=0.5)
    ap.add_argument("--cell-q-corr", type=float, default=0.6)
    ap.add_argument("--membrane-width", type=float, default=0.0)
    ap.add_argument("--membrane-width-frac", type=float, default=0.15)
    ap.add_argument("--membrane-strength", type=float, default=2.0)

    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--graph-chunk-size", type=int, default=0)
    ap.add_argument(
        "--false-edge-frac",
        type=float,
        default=0.0,
        help=(
            "Fraction of corruption to inject. By default this is interpreted as the fraction of "
            "total weighted edges to rewire. With --false-target-nodes, it is interpreted as the "
            "fraction of nodes on each bipartite side to fuse to random same-side donors."
        ),
    )
    ap.add_argument(
        "--false-target-nodes",
        action="store_true",
        help=(
            "Interpret --false-edge-frac as a node-fusion fraction instead of an edge-rewiring fraction. "
            "Targeted nodes copy the donor's graph representation and become indistinguishable from it."
        ),
    )
    ap.add_argument("--seed", type=int)

    ap.add_argument("--benchmark", action="store_true",
                    help="Run a segmentation benchmark sweep.")
    ap.add_argument("--segmentation-module", default="",
                    help="Python module name or path to a .py file defining method1(graph), method2(graph), method3(graph).")

    ap.add_argument("--use-optimops", action="store_true",
                    help="Use optimOps.py as the segmentation backend. With the attached optimOps.py, "
                         "cluster_labels.npy currently contains 2 clusterings: hdbscan and infomap. "
                         "You can either set --use-optimops and (optionally) --optimops-module, "
                         "or simply set --segmentation-module to optimOps.py.")
    ap.add_argument("--optimops-module", default="",
                    help="Python module name or path to optimOps.py (used when --use-optimops is set). "
                         "If omitted, --segmentation-module is used.")
    ap.add_argument("--optimops-inference-dim", type=int, default=2,
                    help="optimOps: -inference_dim (embedding dimension).")
    ap.add_argument("--optimops-inference-eignum", type=int, default=30,
                    help="optimOps: -inference_eignum (number of eigenvectors in the inference basis).")
    ap.add_argument("--optimops-final-eignum", type=int, default=100,
                    help="optimOps: -final_eignum (used by the 2-pass solve; keep at default unless you know why).")
    ap.add_argument("--optimops-scales", type=int, default=1,
                    help="optimOps: -scales (number of main outer iterations for the final scalar solve).")
    ap.add_argument("--optimops-output-name", default="GSEoutput.txt",
                    help="optimOps: output filename passed as output_name to run_GSE() (written inside each run dir).")
    ap.add_argument("--optimops-force", action="store_true",
                    help="optimOps: force re-run even if cluster_labels.npy already exists.")
    ap.add_argument("--optimops-render-roc", action="store_true",
                    help="optimOps: also write a pairwise ROC curve plot (same Voronoi cell vs embedding distance) "
                         "as GSE_embedding_pairwise_ROC.png when ground_truth_cells.csv is available.")
    ap.add_argument("--optimops-coarsen-infomap", action="store_true",
                    help="optimOps: enable -coarsen_infomap. In the attached backend this is only supported together with --optimops-register-zf and is auto-enabled for that route.")
    ap.add_argument("--optimops-coarsen-annotation-binary-threshold", type=int, default=None,
                    help="optimOps: pass -coarsen_annotation_binary_threshold when building coarsened annotation matrices.")
    ap.add_argument("--optimops-register-zf", "--optimops-register-zf-new",
                    dest="optimops_register_zf", default="",
                    help="optimOps: pass the register_zf time flag (one of 12hpf, 18hpf, 24hpf) to enable the coarsen-and-align branch.")
    ap.add_argument("--optimops-slice-path", default="",
                    help="optimOps: pass -slice_path. If omitted with --optimops-register-zf, each run uses a generated register_zf_slice.h5ad.")
    ap.add_argument("--optimops-generate-register-zf-fixture", "--optimops-zf-synthetic-fixtures",
                    dest="optimops_zf_synthetic_fixtures", action="store_true",
                    help="Prewrite synthetic final.h5ad and register_zf_slice.h5ad fixtures for optimOps/register_zf tests. This is auto-enabled when --optimops-register-zf is set and --optimops-slice-path is omitted.")
    ap.add_argument("--optimops-register-zf-fixture-slice-n", "--optimops-zf-slice-n",
                    dest="optimops_zf_slice_n", type=int, default=0,
                    help="Number of raw slice observations in generated register_zf_slice.h5ad; 0 uses all simulated nodes.")
    ap.add_argument("--optimops-register-zf-fixture-pairs", "--optimops-zf-num-pole-pairs",
                    dest="optimops_zf_num_pole_pairs", type=int, default=3,
                    help="Number of synthetic A/B pole marker pairs to generate for register_zf.")
    ap.add_argument("--optimops-register-zf-fixture-genes-per-pole", "--optimops-zf-genes-per-pole",
                    dest="optimops_zf_genes_per_pole", type=int, default=3,
                    help="Number of synthetic genes per pole side; keep this >= register_zf genes_per_pole.")
    ap.add_argument("--optimops-register-zf-match-lam-dir", type=float, default=None,
                    help="Forward to current optimOps.py as -register_zf_match_lam_dir.")
    ap.add_argument("--optimops-register-zf-match-refine-iter", type=int, default=None,
                    help="Forward to current optimOps.py as -register_zf_match_refine_iter.")
    ap.add_argument("--optimops-register-zf-ensemble-size", type=int, default=None,
                    help="Forward to current optimOps.py as -register_zf_ensemble_size.")
    ap.add_argument("--optimops-register-zf-ensemble-n-jobs", type=int, default=None,
                    help="Set REGISTER_ZF_ENSEMBLE_N_JOBS around the current optimOps.py call.")
    ap.add_argument("--optimops-register-zf-ensemble-threads-per-worker", type=int, default=None,
                    help="Set REGISTER_ZF_ENSEMBLE_THREADS_PER_WORKER around the current optimOps.py call.")
    ap.add_argument("--optimops-register-zf-ensemble-mp-start-method", default=None,
                    help="Set REGISTER_ZF_ENSEMBLE_MP_START_METHOD around the current optimOps.py call.")
    ap.add_argument("--viz-register-zf-alignment",
                    dest="viz_register_zf_alignment", action=argparse.BooleanOptionalAction, default=True,
                    help="When --optimops-register-zf is enabled, write per-run register_zf spatial alignment diagnostics beside match_result_<ZF_FLAG>.")
    ap.add_argument("--viz-register-zf-alignment-max-pairs",
                    dest="viz_register_zf_alignment_max_pairs", type=int, default=2,
                    help="Maximum number of register_zf pole-pair channels to visualize per run.")
    ap.add_argument("--viz-register-zf-alignment-point-size",
                    dest="viz_register_zf_alignment_point_size", type=float, default=7.0,
                    help="Scatter point size for register_zf alignment diagnostic plots.")
    ap.add_argument("--viz-register-zf-alignment-robust-quantile",
                    dest="viz_register_zf_alignment_robust_quantile", type=float, default=0.995,
                    help="Robust quantile for symmetric residual color limits in register_zf alignment diagnostic plots.")
    ap.add_argument("--method-input", choices=["csr", "dict"], default="csr",
                    help="What object is passed to the segmentation methods: CSR adjacency, or a dict with adjacency+n0+n1.")
    ap.add_argument("--sweep-reps", type=int, default=1,
                    help="Number of replicate simulations per sweep value.")
    ap.add_argument("--benchmark-second-sweep",
                    choices=["barrier_ratio", "false_fraction", "both"],
                    default="barrier_ratio",
                    help="Select the second benchmark sweep axis: barrier_ratio, false_fraction, or both.")
    ap.add_argument("--sweep-barrier-ratios", default="",
                    help="Comma-separated list of membrane/intracellular diffusion ratios (D_mem/D_in) to sweep.")
    ap.add_argument("--sweep-false-fracs", "--sweep-false-edge-fracs",
                    dest="sweep_false_fracs",
                    default="",
                    help=("Comma-separated list of false-fraction values for the benchmark corruption sweep. "
                          "With --false-target-nodes, these become node-fusion fractions."))
    ap.add_argument("--sweep-node-scales", default="",
                    help="Comma-separated list of scale factors applied to n0 and n1 to sweep node sampling density.")
    ap.add_argument("--sweep-ncells", default="",
                    help="Comma-separated list of n_cells values to sweep cell spatial density.")
    ap.add_argument("--viz-isosurfaces", action="store_true",
                    help="In benchmark mode, render boundary isosurfaces + diffusion slices for low/high n_cells (replicate 0 only).")
    ap.add_argument("--viz-isosurface-grid-res", type=int, default=85,
                    help="Grid resolution per axis for marching-cubes isosurface rendering (benchmark viz).")
    ap.add_argument("--viz-isosurface-show-points", action="store_true",
                    help="Overlay simulated node positions on the benchmark isosurface plots.")
    ap.add_argument("--viz-isosurface-point-size", type=float, default=1.0,
                    help="Point size for benchmark isosurface plots (if --viz-isosurface-show-points).")

    args = ap.parse_args()
    _sync_optimops_register_zf_aliases(args)
    if not (0.0 <= float(args.false_edge_frac) <= 1.0):
        raise SystemExit("--false-edge-frac must lie in [0, 1].")
    if _optimops_register_zf_requested(args):
        zf_flag = _optimops_register_zf_flag(args)
        if zf_flag not in {"12hpf", "18hpf", "24hpf"}:
            raise SystemExit("--optimops-register-zf must be one of: 12hpf, 18hpf, 24hpf.")
        args.optimops_register_zf = zf_flag
        if not bool(getattr(args, "optimops_coarsen_infomap", False)):
            print("[INFO] --optimops-register-zf uses optimOps' coarsened Infomap route; enabling --optimops-coarsen-infomap.")
            args.optimops_coarsen_infomap = True
        if not str(getattr(args, "optimops_slice_path", "") or "").strip():
            print("[INFO] No --optimops-slice-path supplied; enabling synthetic register_zf fixture generation.")
            args.optimops_zf_synthetic_fixtures = True
        if int(getattr(args, "optimops_inference_dim", 2)) not in (2, 3):
            raise SystemExit("--optimops-register-zf requires --optimops-inference-dim 2 or 3.")

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.benchmark:
        run_segmentation_benchmark(args)
        return

    pos_csv = args.build_from_posfile
    voronoi_centers = None

    if pos_csv and bool(args.hole):
        print("[WARN] --hole only affects auto-generated positions; ignoring because --build-from-posfile was provided.")

    if not pos_csv:
        if args.n0 is None or args.n1 is None:
            raise SystemExit("Provide --build-from-posfile, or (to auto-generate positions) specify --n0 and --n1.")

        centers_u, points_u, labels_u = generate_3d_voronoi_positions(
            n0=args.n0,
            n1=args.n1,
            n_cells=args.n_cells,
            cell_jitter=args.cell_jitter,
            bbox_lo=(tuple(args.bbox_lo) if args.bbox_lo is not None else None),
            bbox_hi=(tuple(args.bbox_hi) if args.bbox_hi is not None else None),
            boundary_margin_frac=args.boundary_margin_frac,
            hole=bool(args.hole),
            threads=int(args.threads),
            seed=args.seed,
        )
        voronoi_centers = centers_u

        gen_csv = args.generated_positions_csv or str(outdir / "positions_generated.csv")
        arr_gen = positions_to_csv_array(points_u, labels_u)
        np.savetxt(gen_csv, arr_gen, delimiter=',', fmt='%.6g')
        print(f"[INFO] Generated positions CSV: {gen_csv}")
        pos_csv = gen_csv

    graph_path, n0, n1 = build_graph_from_positions_cellular(
        pos_csv=str(pos_csv),
        outdir=outdir,
        rescale=float(args.rescale),
        mperPt=float(args.mperPt),
        pi_short=float(args.pi_short),
        sigma_s=float(args.sigma_s),
        short_trunc=float(args.short_trunc),
        k_capture=float(args.k_capture),
        long_trunc=float(args.long_trunc),
        long_eps=float(args.long_eps),
        path_samples=int(args.path_samples),
        path_mode=str(args.path_mode),
        max_nbrs_per_source=int(args.max_nbrs_per_source),
        n_cells=int(args.n_cells),
        cell_jitter=float(args.cell_jitter),
        D_in=float(args.D_in),
        D_out=float(args.D_out),
        ecs_width=float(args.ecs_width),
        ecs_width_frac=float(args.ecs_width_frac),
        P_mem=float(args.P_mem),
        D_min=float(args.D_min),
        D_max=float(args.D_max),
        qp_modes=int(args.qp_modes),
        qp_wavelength=(None if float(args.qp_wavelength) <= 0 else float(args.qp_wavelength)),
        qp_amp=float(args.qp_amp),
        cell_sigma=float(args.cell_sigma),
        cell_q_corr=float(args.cell_q_corr),
        membrane_width=float(args.membrane_width),
        membrane_width_frac=float(args.membrane_width_frac),
        membrane_strength=float(args.membrane_strength),
        qp_mode=str(args.qp_mode),
        amp_dispersion=float(args.amp_dispersion),
        false_edge_frac=float(args.false_edge_frac),
        false_edge_targets_nodes=bool(args.false_target_nodes),
        write_synthetic_zf_fixtures=bool(getattr(args, "optimops_zf_synthetic_fixtures", False) or _optimops_register_zf_requested(args)),
        synthetic_zf_flag=str(_optimops_register_zf_flag(args) or "18hpf"),
        synthetic_zf_slice_path=(
            None if not str(getattr(args, "optimops_slice_path", "") or "").strip()
            else str(getattr(args, "optimops_slice_path"))
        ),
        synthetic_zf_write_slice=bool(getattr(args, "optimops_zf_synthetic_fixtures", False) or not str(getattr(args, "optimops_slice_path", "") or "").strip()),
        synthetic_zf_slice_n=int(getattr(args, "optimops_zf_slice_n", 0)),
        synthetic_zf_num_pole_pairs=int(getattr(args, "optimops_zf_num_pole_pairs", 3)),
        synthetic_zf_genes_per_pole=int(getattr(args, "optimops_zf_genes_per_pole", 3)),
        threads=int(args.threads),
        graph_chunk_size=int(args.graph_chunk_size),
        seed=args.seed,
        voronoi_centers=voronoi_centers,
        render_isosurface=bool(args.render_isosurface),
        isosurface_grid_res=int(args.isosurface_grid_res),
        isosurface_show_points=bool(args.isosurface_show_points),
        isosurface_point_size=float(args.isosurface_point_size),
        isosurface_panel_size=tuple(int(x) for x in args.isosurface_panel_size),
    )

    print(f"[INFO] Wrote graph: {graph_path}  (n0={n0}, n1={n1}, n={n0+n1})")
    print(f"[INFO] Wrote ground truth files into: {outdir}")


if __name__ == "__main__":
    main()

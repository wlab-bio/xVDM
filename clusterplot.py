"""Segmentation helpers for xVDM/GSE outputs.

The production GSE route uses HDBSCAN on final coordinates and Infomap on a
sparse final-coordinate diffusion-transport graph. Both routes can be curated
with the raw adjacency graph and split by raw connected components.

Experimental/back-compatibility helpers for transport-HDBSCAN, Leiden,
sparsest-cut, and dual-wrapper calls are kept below, but they are not the
cell-calling route used by ``optimOps.run_GSE()``.
"""

from __future__ import annotations

import functools
import math
import os
import shutil
import subprocess
from collections import deque
from typing import Literal, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix, load_npz, triu
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, eigsh


# -----------------------------------------------------------------------------
# Optional sysOps integration (keeps drop-in compatibility with the original code)
# -----------------------------------------------------------------------------

try:
    import sysOps  # type: ignore

    def _status(msg: str) -> None:
        # Original pipeline uses sysOps.throw_status
        if hasattr(sysOps, "throw_status"):
            sysOps.throw_status(msg)
        else:
            print(msg, flush=True)

    def _global_data_path() -> Optional[str]:
        return getattr(sysOps, "globaldatapath", None)

except Exception:  # pragma: no cover
    sysOps = None  # type: ignore

    def _status(msg: str) -> None:
        print(msg, flush=True)

    def _global_data_path() -> Optional[str]:
        return None


# -----------------------------------------------------------------------------
# Numba (optional but strongly recommended for large graphs)
# -----------------------------------------------------------------------------

try:
    from numba import jit, prange

except Exception:  # pragma: no cover
    # Fallback: keep correctness (but will be slow)
    def jit(*_args, **_kwargs):  # type: ignore
        def _wrap(fn):
            return fn

        return _wrap

    prange = range  # type: ignore


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _as_csr(mat: Union[np.ndarray, csr_matrix]) -> csr_matrix:
    if isinstance(mat, csr_matrix):
        return mat
    return csr_matrix(mat)


def _remap_contiguous(labels: np.ndarray) -> np.ndarray:
    """Remap non-negative labels to contiguous 0..K-1; preserve -1."""
    labels = np.asarray(labels)
    out = np.full(labels.shape, -1, dtype=np.int32)
    good = labels >= 0
    if not np.any(good):
        return out
    uniq = np.unique(labels[good])
    # uniq is sorted, so searchsorted gives the new contiguous id
    out[good] = np.searchsorted(uniq, labels[good]).astype(np.int32)
    return out


def _normalize_which(which: str) -> Literal["hdbscan", "infomap", "both"]:
    w = (which or "").strip().lower()
    if w in {"hdbscan", "orig", "original", "graph1", "method1", "1"}:
        return "hdbscan"
    if w in {"infomap", "transport", "transformed", "graph2", "method2", "2"}:
        return "infomap"
    if w in {"both", "all", "dual", "hdbscan+infomap", "infomap+hdbscan"}:
        return "both"

# Keep public defaults aligned with optimOps final clustering
_DEFAULT_MIN_CLUSTER_SIZE = 10
_DEFAULT_HDBSCAN_MIN_SAMPLES = 10
_DEFAULT_INFOMAP_MARKOV_TIME = 1.0

@jit(nopython=True, parallel=True)
def curate_labels_numba(
    labels: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
    """Majority-vote reassignment for noise points (-1) using CSR neighbors.

    A noise point is reassigned iff a strict majority (> 50%) of its neighbors
    (restricted to non-noise labels) share the same label.
    """
    revised_labels = labels.copy()

    for idx in prange(len(neighbor_indices)):
        point_idx = neighbor_indices[idx]
        start = indptr[point_idx]
        end = indptr[point_idx + 1]

        neighbor_labs = labels[indices[start:end]]
        valid_neighbors = neighbor_labs[neighbor_labs >= 0]

        if len(valid_neighbors) == 0:
            continue

        # Find mode (most frequent label) in valid_neighbors
        sorted_neighbors = np.sort(valid_neighbors)
        current_label = sorted_neighbors[0]
        current_count = 1
        max_count = 1
        max_label = current_label

        for i in range(1, len(sorted_neighbors)):
            if sorted_neighbors[i] == current_label:
                current_count += 1
            else:
                if current_count > max_count:
                    max_count = current_count
                    max_label = current_label
                current_label = sorted_neighbors[i]
                current_count = 1

        if current_count > max_count:
            max_count = current_count
            max_label = current_label

        if max_count > len(valid_neighbors) / 2:
            revised_labels[point_idx] = max_label

    return revised_labels


def curate_labels_with_graph(labels: np.ndarray, adjacency: Union[np.ndarray, csr_matrix]) -> np.ndarray:
    """Run one curation pass over all noise points (-1) using the given graph."""
    adjacency_csr = _as_csr(adjacency)
    adjacency_csr.sum_duplicates()
    adjacency_csr.eliminate_zeros()

    noise_indices = np.where(labels == -1)[0]
    if noise_indices.size == 0:
        return labels

    revised_labels = curate_labels_numba(labels, adjacency_csr.indices, adjacency_csr.indptr, noise_indices)
    return revised_labels


def _ensure_square(adjacency: csr_matrix) -> None:
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Adjacency must be square; got shape={adjacency.shape}.")


def _largest_connected_component_mask(adjacency: Union[np.ndarray, csr_matrix]) -> np.ndarray:
    """Boolean mask for nodes in the (undirected) largest connected component.

    The mask is computed on the *unpruned* graph and is intended for "LCC-protected"
    pruning/renormalization, where we preserve transition weights for nodes that
    already live in the dominant connected component as much as possible.

    For large graphs this is O(nnz) and typically dominated by downstream steps
    (e.g. Infomap itself).
    """
    adj = _as_csr(adjacency)
    _ensure_square(adj)

    try:
        n_comp, labels = connected_components(adj, directed=False, return_labels=True)
    except Exception:
        # Conservative fallback: treat everything as one component.
        return np.ones(adj.shape[0], dtype=bool)

    if int(n_comp) <= 1:
        return np.ones(adj.shape[0], dtype=bool)

    sizes = np.bincount(labels.astype(np.int64), minlength=int(n_comp))
    largest = int(np.argmax(sizes))
    return labels == largest


def _resolve_transformed_matrix_path(transformed_matrix_path: Optional[str]) -> str:
    """Resolve transformed_matrix.npz path using sysOps.globaldatapath when needed."""
    tm_path = transformed_matrix_path
    if tm_path is None:
        gdp = _global_data_path()
        if gdp is None:
            raise ValueError(
                "transformed_matrix_path was not provided and sysOps.globaldatapath is unavailable. "
                "Pass transformed_matrix_path explicitly."
            )
        tm_path = os.path.join(gdp, "transformed_matrix.npz")

    if not os.path.exists(tm_path):
        raise FileNotFoundError(f"Expected transformed transport graph at {tm_path} but it does not exist.")

    return tm_path


@jit(nopython=True)
def _uf_find(parent: np.ndarray, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@jit(nopython=True)
def _uf_union(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> None:
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


@jit(nopython=True)
def _union_same_label_edges(indptr, indices, labels, parent, rank):
    n = indptr.shape[0] - 1
    for i in range(n):
        li = labels[i]
        if li < 0:
            continue
        start = indptr[i]
        end = indptr[i + 1]
        for p in range(start, end):
            j = indices[p]
            if j <= i:
                continue
            if labels[j] == li:
                _uf_union(parent, rank, i, j)


@jit(nopython=True)
def _compress_roots(parent):
    n = parent.shape[0]
    roots = np.empty(n, dtype=np.int32)
    for i in range(n):
        roots[i] = _uf_find(parent, int(i))
    return roots


def split_clusters_by_raw_connected_components(
    labels: np.ndarray,
    raw_adjacency: Union[np.ndarray, csr_matrix],
    *,
    min_component_size: Optional[int] = 1,
    route_name: str = "cluster",
) -> np.ndarray:
    """Split labels into connected components of the raw link graph.

    Every raw connected component inside a label is preserved as its own
    candidate cluster; nothing is reduced to only the largest component.
    If ``min_component_size`` is provided, the same size threshold is applied to
    every resulting component independently.
    """
    labels = np.asarray(labels, dtype=np.int32)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D; got shape={labels.shape}")

    valid = labels >= 0
    if not np.any(valid):
        return labels.astype(np.int32, copy=True)

    raw = _as_csr(raw_adjacency)
    _ensure_square(raw)
    if raw.shape[0] != labels.shape[0]:
        raise ValueError(
            f"raw_adjacency shape {raw.shape} does not match label length {labels.shape[0]}"
        )

    raw = raw.maximum(raw.T).tocsr()
    raw.setdiag(0)
    raw.eliminate_zeros()

    parent = np.arange(raw.shape[0], dtype=np.int32)
    rank = np.zeros(raw.shape[0], dtype=np.int16)
    _union_same_label_edges(raw.indptr, raw.indices, labels, parent, rank)
    roots = _compress_roots(parent)

    valid_idx = np.flatnonzero(valid)
    keys = np.empty(valid_idx.size, dtype=[("label", np.int32), ("root", np.int32)])
    keys["label"] = labels[valid_idx]
    keys["root"] = roots[valid_idx]
    _, inv = np.unique(keys, return_inverse=True)

    out = np.full(labels.shape, -1, dtype=np.int32)
    out[valid_idx] = inv.astype(np.int32, copy=False)

    n_before = int(np.unique(labels[valid]).size)
    n_after_split = int(np.unique(out[out >= 0]).size)
    _status(f"({route_name}) Raw-graph connected-component split: {n_before} -> {n_after_split} clusters.")

    mcs = 1 if min_component_size is None else int(min_component_size)
    if mcs > 1:
        valid_out = out >= 0
        if np.any(valid_out):
            sizes = np.bincount(out[valid_out])
            small = np.where(sizes < mcs)[0]
            if small.size > 0:
                out = out.copy()
                out[np.isin(out, small)] = -1
                kept = int(np.unique(out[out >= 0]).size)
                _status(
                    f"({route_name}) Raw-graph component size filter min_component_size={mcs}: "
                    f"kept {kept} clusters; dropped {int(small.size)} small components."
                )

    out = _remap_contiguous(out)
    return out.astype(np.int32, copy=False)


@jit(nopython=True, parallel=True)
def _transport_neglog_distance_data(indptr, indices, data, row_strength, eps):
    out = np.empty(data.shape[0], dtype=np.float64)
    n = indptr.shape[0] - 1
    for i in prange(n):
        si = row_strength[i]
        if si < eps:
            si = eps
        sqrt_si = math.sqrt(si)
        start = indptr[i]
        end = indptr[i + 1]
        for p in range(start, end):
            j = indices[p]
            sj = row_strength[j]
            if sj < eps:
                sj = eps
            a = float(data[p]) / (sqrt_si * math.sqrt(sj))
            if a > 1.0:
                a = 1.0
            if a < eps:
                a = eps
            d = -math.log(a)
            if d < eps:
                d = eps
            out[p] = d
    return out


def _transport_similarity_to_distance_csr(
    P: csr_matrix,
    row_strength: np.ndarray,
    *,
    distance_eps: float = 1e-12,
) -> csr_matrix:
    """Convert a sparse transport similarity graph into sparse precomputed distances."""
    if float(distance_eps) <= 0.0:
        raise ValueError("distance_eps must be > 0")

    D = P.tocsr(copy=True)
    D.data = _transport_neglog_distance_data(
        D.indptr,
        D.indices,
        D.data.astype(np.float64, copy=False),
        np.asarray(row_strength, dtype=np.float64),
        float(distance_eps),
    )
    return D


def _add_self_distance_diagonal(distance_graph: csr_matrix, *, self_distance: float) -> csr_matrix:
    """Store an explicit positive self-distance on the diagonal.

    `sklearn.cluster.HDBSCAN` counts sparse stored distances per row when
    checking the `min_samples` neighborhood requirement. Since sklearn's
    `min_samples` includes the point itself, sparse precomputed inputs need an
    explicit diagonal entry to represent that self-neighbor.
    """
    if float(self_distance) <= 0.0:
        raise ValueError("self_distance must be > 0")

    D = distance_graph.tocsr(copy=True)
    D.setdiag(float(self_distance))
    D.eliminate_zeros()
    return D


def _augment_distance_graph_with_fallback_knn(
    distance_graph: csr_matrix,
    fallback_knn_indices: np.ndarray,
    *,
    min_stored_neighbors: int,
    fill_value: float,
    route_name: str = "hdbscan transport",
    component_labels: Optional[np.ndarray] = None,
) -> csr_matrix:
    """Add weak fallback edges from a coordinate-kNN graph until each row stores enough distances.

    The added edges only exist to satisfy sklearn's sparse-precomputed neighbor-count
    requirement; they are assigned a large fallback distance and therefore act as the
    weakest possible ties.
    """
    D = distance_graph.tocsr(copy=True)
    fb = np.asarray(fallback_knn_indices)
    if fb.ndim != 2 or fb.shape[0] != D.shape[0]:
        raise ValueError(
            f"fallback_knn_indices must have shape (n_nodes, k); got {fb.shape} for n_nodes={D.shape[0]}"
        )
    need_n = int(min_stored_neighbors)
    if need_n < 1:
        return D
    if float(fill_value) <= 0.0:
        raise ValueError("fill_value must be > 0")

    counts = np.diff(D.indptr)
    deficient_rows = np.flatnonzero(counts < need_n)
    if deficient_rows.size == 0:
        return D

    add_rows = []
    add_cols = []
    add_vals = []
    comp = None if component_labels is None else np.asarray(component_labels)
    if comp is not None and comp.shape[0] != D.shape[0]:
        raise ValueError("component_labels has incompatible shape")

    for i in deficient_rows:
        row_start = D.indptr[i]
        row_end = D.indptr[i + 1]
        existing = set(int(x) for x in D.indices[row_start:row_end])
        needed = need_n - len(existing)
        if needed <= 0:
            continue

        comp_i = None if comp is None else int(comp[i])
        for j in fb[i]:
            j = int(j)
            if j < 0 or j == i or j in existing:
                continue
            if comp_i is not None and int(comp[j]) != comp_i:
                continue
            add_rows.append(int(i))
            add_cols.append(j)
            add_vals.append(float(fill_value))
            existing.add(j)
            needed -= 1
            if needed <= 0:
                break

    if len(add_rows) == 0:
        return D

    A = csr_matrix(
        (
            np.asarray(add_vals, dtype=np.float64),
            (np.asarray(add_rows, dtype=np.int64), np.asarray(add_cols, dtype=np.int64)),
        ),
        shape=D.shape,
    )
    D = (D + A + A.T).tocsr()
    D.sum_duplicates()
    D.eliminate_zeros()

    remaining = int(np.sum(np.diff(D.indptr) < need_n))
    _status(
        f"({route_name}) Added {int(len(add_rows))} fallback kNN edges (plus symmetric counterparts); "
        f"remaining deficient rows = {remaining}."
    )
    return D


def _fit_hdbscan_sparse_precomputed_by_component(
    distance_graph: csr_matrix,
    connectivity_graph: csr_matrix,
    *,
    min_cluster_size: int,
    min_samples: int,
    allow_single_cluster: bool = True,
    cluster_selection_method: str = "eom",
    route_name: str = "hdbscan transport",
) -> np.ndarray:
    """Fit sklearn.cluster.HDBSCAN(metric='precomputed') on each connected component."""
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

    D = distance_graph.tocsr()
    G = connectivity_graph.tocsr()
    _ensure_square(D)
    _ensure_square(G)
    if D.shape != G.shape:
        raise ValueError(f"distance_graph shape {D.shape} != connectivity_graph shape {G.shape}")

    n = int(D.shape[0])
    labels = np.full(n, -1, dtype=np.int32)
    n_comp, comp_ids = connected_components(G, directed=False, return_labels=True)
    _status(f"({route_name}) Fitting sparse-precomputed HDBSCAN across {int(n_comp)} connected transport components.")

    next_label = 0
    sk_min_samples_global = max(2, int(min_samples) + 1)
    min_cluster_size = int(min_cluster_size)

    for comp_id in range(int(n_comp)):
        nodes = np.flatnonzero(comp_ids == comp_id)
        size = int(nodes.size)
        if size <= 0:
            continue
        if size < min_cluster_size:
            continue

        D_sub = D[nodes][:, nodes].tocsr()
        if D_sub.nnz == 0:
            continue

        sk_min_samples_sub = min(sk_min_samples_global, size)
        row_counts_sub = np.diff(D_sub.indptr)
        min_row_count_sub = int(np.min(row_counts_sub)) if row_counts_sub.size > 0 else 0
        if min_row_count_sub < 2:
            continue
        if min_row_count_sub < int(sk_min_samples_sub):
            _status(
                f"({route_name}) connected transport component {comp_id} only stores "
                f"{min_row_count_sub} sparse distances per row at minimum; reducing effective "
                f"min_samples from {int(sk_min_samples_sub) - 1} to {min_row_count_sub - 1}."
            )
            sk_min_samples_sub = max(2, min_row_count_sub)
        max_distance_sub = float(np.max(D_sub.data)) if D_sub.data.size > 0 else 1.0
        if not np.isfinite(max_distance_sub) or max_distance_sub <= 0.0:
            max_distance_sub = 1.0
        else:
            max_distance_sub *= 1.05

        clusterer = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=int(sk_min_samples_sub),
            metric="precomputed",
            algorithm="brute",
            allow_single_cluster=bool(allow_single_cluster),
            cluster_selection_method=str(cluster_selection_method),
            metric_params={"max_distance": float(max_distance_sub)},
            copy=False,
        )
        labs_sub = np.asarray(clusterer.fit_predict(D_sub), dtype=np.int32)
        labs_sub[labs_sub < 0] = -1

        good = labs_sub >= 0
        if np.any(good):
            n_local = int(np.max(labs_sub[good])) + 1
            labs_off = labs_sub.copy()
            labs_off[good] += int(next_label)
            labels[nodes] = labs_off
            next_label += n_local

    return labels


# -----------------------------------------------------------------------------
# Version 1: HDBSCAN + original adjacency curation
# -----------------------------------------------------------------------------


def _perform_hdbscan(
    points: np.ndarray,
    min_cluster_size: int = _DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = _DEFAULT_HDBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    """Run HDBSCAN on Nx3 (or NxD) point coordinates."""
    import hdbscan  # local import keeps this module importable without hdbscan

    _status("Performing HDBSCAN clustering (hdbscan)...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(points)
    _status("HDBSCAN clustering complete (hdbscan).")
    return labels.astype(np.int32, copy=False)


def execute_clusters_hdbscan(
    Xpts: np.ndarray,
    adjacency: Union[np.ndarray, csr_matrix],
    min_cluster_size: int = _DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = _DEFAULT_HDBSCAN_MIN_SAMPLES,
    curation_tol: float = 1e-4,
    max_curation_iter: int = 50,
    symmetrize: bool = True,
) -> np.ndarray:
    """Version 1 segmentation: HDBSCAN + iterative curation on original adjacency."""
    if Xpts is None:
        raise ValueError("hdbscan requires Xpts (point coordinates).")
    if adjacency is None:
        raise ValueError("hdbscan requires adjacency (original graph).")

    adj = _as_csr(adjacency)
    _ensure_square(adj)
    if symmetrize:
        # Many pipelines store upper triangle only.
        adj = (adj + adj.T).tocsr()
        adj.eliminate_zeros()

    labels = _perform_hdbscan(Xpts, min_cluster_size=min_cluster_size, min_samples=min_samples)

    # Curate labels until convergence.
    for it in range(int(max_curation_iter)):
        revised = curate_labels_with_graph(labels, adj)
        modified_frac = float(np.mean(revised != labels))
        _status(f"(hdbscan) Modified fraction on iteration {it} = {modified_frac:.6g}")
        labels = revised
        if modified_frac < float(curation_tol):
            break

    labels = _remap_contiguous(labels)
    return labels


def execute_clusters_hdbscan_transport(
    min_cluster_size: int = _DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = _DEFAULT_HDBSCAN_MIN_SAMPLES,
    transformed_matrix_path: Optional[str] = None,
    k_per_row: Optional[int] = None,
    weight_fraction_per_row: Optional[float] = None,
    min_edges_per_row: int = 3,
    curation_tol: float = 1e-4,
    max_curation_iter: int = 50,
    distance_eps: float = 1e-12,
) -> np.ndarray:
    """Run sparse-precomputed HDBSCAN directly on a transformed transport graph.

    When ``k_per_row`` and ``weight_fraction_per_row`` are both left as ``None``,
    the raw sparse ``transformed_matrix.npz`` is reused as the neighborhood graph.
    This is the intended route for the earlier higher-dimensional transport matrix,
    where the matrix itself already encodes the source kNN structure.
    """
    tm_path = _resolve_transformed_matrix_path(transformed_matrix_path)
    _status(f"Loading transformed transport graph (hdbscan transport): {tm_path}")
    T = load_npz(tm_path).tocsr()
    n = int(T.shape[0])
    _ensure_square(T)

    T = T.maximum(T.T).tocsr()
    T.setdiag(0)
    T.eliminate_zeros()

    if T.nnz == 0:
        _status("(hdbscan transport) Transport graph is empty; returning all-noise labels.")
        return np.full(n, -1, dtype=np.int32)

    if weight_fraction_per_row is None and k_per_row is None:
        P = T
        _status(
            "(hdbscan transport) Using raw transformed_matrix.npz as the sparse neighbor graph "
            "for precomputed HDBSCAN distances."
        )
    else:
        if k_per_row is None:
            k = max(int(min_edges_per_row), int(min_samples) + 1)
        else:
            k = int(k_per_row)
            if k < 1:
                raise ValueError("k_per_row must be >= 1")

        if weight_fraction_per_row is not None:
            frac = float(weight_fraction_per_row)
            if not (0.0 <= frac <= 1.0):
                raise ValueError("weight_fraction_per_row must be in [0, 1].")
            _status(
                f"(hdbscan transport) Pruning transformed graph by per-row cumulative weight fraction={frac:.6g} "
                f"with cap k_max={k} and min_edges_per_row={int(min_edges_per_row)}."
            )
            P = _prune_cumweight_csr(T, k_max=k, weight_fraction=frac, min_keep=int(min_edges_per_row))
        else:
            _status(f"(hdbscan transport) Pruning transformed graph to ~{k} strongest edges per node.")
            P = _prune_topk_csr(T, k)

        P = P.maximum(P.T).tocsr()
        P.setdiag(0)
        P.eliminate_zeros()

    if P.nnz == 0:
        _status("(hdbscan transport) Prepared transport graph is empty; returning all-noise labels.")
        return np.full(n, -1, dtype=np.int32)

    row_strength = np.asarray(P.sum(axis=1)).reshape(-1).astype(np.float64, copy=False)
    D = _transport_similarity_to_distance_csr(P, row_strength, distance_eps=float(distance_eps))
    D = _add_self_distance_diagonal(D, self_distance=max(float(distance_eps), 1e-12))

    labels = _fit_hdbscan_sparse_precomputed_by_component(
        D,
        P,
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        route_name="hdbscan transport",
    )

    for it in range(int(max_curation_iter)):
        revised = curate_labels_with_graph(labels, P)
        modified_frac = float(np.mean(revised != labels))
        _status(f"(hdbscan transport) Modified fraction on iteration {it} = {modified_frac:.6g}")
        labels = revised
        if modified_frac < float(curation_tol):
            break

    labels = _remap_contiguous(labels)
    _status("HDBSCAN clustering complete (hdbscan transport).")
    return labels.astype(np.int32, copy=False)


# -----------------------------------------------------------------------------
# Version 2: Infomap on transformed transport graph
# -----------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _infomap_help_text(binary: str) -> str:
    """Cache Infomap --help output so we can feature-detect flags."""
    try:
        p = subprocess.run(
            [binary, "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return p.stdout or ""
    except Exception:
        return ""


def _find_infomap_binary() -> str:
    """Find an Infomap executable on PATH."""
    for cand in ("infomap", "Infomap"):
        if shutil.which(cand):
            return cand
    raise RuntimeError("Infomap executable not found on PATH (tried 'infomap' and 'Infomap').")


def _run_infomap_link_list(
    link_path: str,
    out_dir: str,
    out_name: str = "tm_modules",
    seed: int = 1,
    num_trials: int = 3,
    silent: bool = True,
    markov_time: float = 1.0,
    *,
    num_threads: int = 1,
    env: Optional[dict] = None,
) -> str:
    """Run Infomap on a .txt link list and return the produced .clu path."""
    binary = _find_infomap_binary()
    help_txt = _infomap_help_text(binary)

    cmd = [binary, link_path, out_dir]

    # Some Infomap builds support --input-format; others infer from extension.
    if ("--input-format" in help_txt) or ("-i<" in help_txt) or ("-i " in help_txt):
        cmd += ["--input-format", "link-list"]

    # Two-level partition (flat segmentation) + .clu output
    cmd += ["--two-level", "--clu", "--seed", str(int(seed)), "-N", str(int(num_trials)), "--out-name", out_name, "--markov-time", str(markov_time)]

    if silent:
        cmd += ["--silent"]

    try:
        if env is None:
            env_run = os.environ.copy()
        else:
            env_run = dict(env)
        for k in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMBA_NUM_THREADS",
            "TBB_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            env_run[k] = str(num_threads)  # <-- CHANGED: Use parameterized thread count
        env_run.setdefault("MKL_DYNAMIC", "FALSE")
        env_run.setdefault("OMP_DYNAMIC", "FALSE")
        env_run.setdefault("KMP_BLOCKTIME", "0")

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env_run)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Infomap failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}"
        ) from e

    clu_path = os.path.join(out_dir, f"{out_name}.clu")
    if not os.path.exists(clu_path):
        cand = [fn for fn in os.listdir(out_dir) if fn.endswith(".clu")]
        raise RuntimeError(f"Infomap did not produce expected {clu_path}. Found .clu files: {cand}")

    return clu_path


def _read_infomap_clu(clu_path: str, n_nodes: int) -> np.ndarray:
    """Read Infomap .clu output and remap module ids to contiguous 0..K-1."""
    raw = np.full(int(n_nodes), -1, dtype=np.int64)
    seq_i = 0

    with open(clu_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s[0] in ("*", "%"):
                # e.g. "*Vertices 123"
                continue

            parts = s.split()
            if len(parts) == 1:
                # Pajek-style: one module id per node, in order
                if seq_i < n_nodes:
                    raw[seq_i] = int(parts[0])
                seq_i += 1
            else:
                # Fallback: "node module"
                node = int(parts[0])
                mod = int(parts[1])
                if 1 <= node <= n_nodes:
                    raw[node - 1] = mod

    return _remap_contiguous(raw.astype(np.int64))


def _write_link_list_upper_txt(
    mat: csr_matrix,
    link_path: str,
    one_based: bool = True,
    include_diagonal: bool = False,
    chunk_edges: int = 1_000_000,
) -> None:
    """Write upper-triangle edges of a symmetric CSR matrix to a weighted link-list file."""
    coo = triu(mat, k=0 if include_diagonal else 1).tocoo()
    rows = coo.row
    cols = coo.col
    data = coo.data

    if one_based:
        rows = rows + 1
        cols = cols + 1

    os.makedirs(os.path.dirname(link_path) or ".", exist_ok=True)

    with open(link_path, "w") as f:
        n = int(data.size)
        for start in range(0, n, int(chunk_edges)):
            end = min(start + int(chunk_edges), n)

            block = np.empty((end - start, 3), dtype=np.float64)
            block[:, 0] = rows[start:end]
            block[:, 1] = cols[start:end]
            block[:, 2] = data[start:end]

            np.savetxt(f, block, fmt=["%d", "%d", "%.12g"])


# Infomap graph pruning utilities. The production Infomap route prunes the
# transformed transport graph before writing a weighted link-list. The default
# policy is top-k per row; an optional cumulative-weight policy keeps the
# strongest row entries until their mass reaches a requested fraction, subject
# to a row cap and minimum edge count.

# ---- scalable pruning kernels (numba) ----


@jit(nopython=True, parallel=True)
def _topk_per_row(indptr, indices, data, k):
    """Return per-row top-k (by weight) with row sums.

    Notes
    -----
    * For rows with deg <= k, we still sort the returned values (descending)
      so downstream cumulative-weight pruning can assume order.
    * Returned weights are float32 for compactness.
    """

    n = indptr.shape[0] - 1
    top_idx = np.full((n, k), -1, np.int32)
    top_w = np.full((n, k), np.float32(-np.inf))
    counts = np.zeros(n, np.int32)
    row_sum = np.zeros(n, np.float64)

    for i in prange(n):
        start = indptr[i]
        end = indptr[i + 1]
        deg = end - start
        if deg <= 0:
            continue

        # Total incident weight (needed for per-row cumulative fraction pruning)
        s = 0.0

        if deg <= k:
            counts[i] = deg
            for t in range(deg):
                w0 = data[start + t]
                s += w0
                top_idx[i, t] = np.int32(indices[start + t])
                top_w[i, t] = np.float32(w0)
        else:
            # init
            for t in range(k):
                w0 = data[start + t]
                s += w0
                top_idx[i, t] = np.int32(indices[start + t])
                top_w[i, t] = np.float32(w0)
            counts[i] = k

            # scan remaining
            for p in range(start + k, end):
                w0 = data[p]
                s += w0
                w = np.float32(w0)

                # find current min
                minpos = 0
                minw = top_w[i, 0]
                for t in range(1, k):
                    if top_w[i, t] < minw:
                        minw = top_w[i, t]
                        minpos = t

                if w > minw:
                    top_w[i, minpos] = w
                    top_idx[i, minpos] = np.int32(indices[p])

        row_sum[i] = s

        # Always sort the kept portion descending by weight.
        c = counts[i]
        if c > 1:
            # selection sort on the kept prefix (cheap when k is small)
            for a in range(c - 1):
                maxpos = a
                maxw = top_w[i, a]
                for b in range(a + 1, c):
                    if top_w[i, b] > maxw:
                        maxw = top_w[i, b]
                        maxpos = b
                if maxpos != a:
                    tmpw = top_w[i, a]
                    top_w[i, a] = top_w[i, maxpos]
                    top_w[i, maxpos] = tmpw
                    tmpj = top_idx[i, a]
                    top_idx[i, a] = top_idx[i, maxpos]
                    top_idx[i, maxpos] = tmpj

    return top_idx, top_w, counts, row_sum


@jit(nopython=True)
def _prefix_indptr(counts):
    n = counts.shape[0]
    indptr = np.empty(n + 1, np.int64)
    indptr[0] = 0
    for i in range(n):
        indptr[i + 1] = indptr[i] + counts[i]
    return indptr


@jit(nopython=True, parallel=True)
def _flatten_topk(top_idx, top_w, counts, indptr):
    n, k = top_idx.shape
    nnz = indptr[-1]
    out_idx = np.empty(nnz, np.int32)
    out_w = np.empty(nnz, np.float32)
    for i in prange(n):
        base = indptr[i]
        c = counts[i]
        for t in range(c):
            out_idx[base + t] = top_idx[i, t]
            out_w[base + t] = top_w[i, t]
    return out_idx, out_w


def _prune_topk_csr(mat: csr_matrix, k: int) -> csr_matrix:
    mat = mat.tocsr()
    mat.eliminate_zeros()
    top_idx, top_w, counts, _row_sum = _topk_per_row(
        mat.indptr,
        mat.indices.astype(np.int32),
        mat.data,
        int(k),
    )
    indptr = _prefix_indptr(counts)
    out_idx, out_w = _flatten_topk(top_idx, top_w, counts, indptr)
    pruned = csr_matrix((out_w, out_idx, indptr), shape=mat.shape)
    pruned.sum_duplicates()
    pruned.eliminate_zeros()
    return pruned


@jit(nopython=True, parallel=True)
def _cumweight_keep_counts(top_w, counts, row_sum, frac, min_keep):
    """Compute how many edges to keep per row given a cumulative weight fraction."""
    n = counts.shape[0]
    out = np.zeros(n, np.int32)

    for i in prange(n):
        c = counts[i]
        if c <= 0:
            out[i] = 0
            continue

        # Clamp min_keep to [1, c]
        mk = min_keep
        if mk < 1:
            mk = 1
        if mk > c:
            mk = c

        if frac >= 1.0:
            out[i] = c
            continue
        if frac <= 0.0:
            out[i] = mk
            continue

        total = row_sum[i]
        if total <= 0.0:
            out[i] = mk
            continue

        target = frac * total
        cum = 0.0
        m = 0

        # top_w is assumed sorted descending on the kept prefix.
        for t in range(c):
            cum += float(top_w[i, t])
            m += 1
            if cum >= target and m >= mk:
                break

        if m < mk:
            m = mk
        if m > c:
            m = c
        out[i] = m

    return out


def _prune_cumweight_csr(
    mat: csr_matrix,
    k_max: int,
    weight_fraction: float,
    *,
    min_keep: int = 3,
) -> csr_matrix:
    """Prune CSR by keeping the strongest edges until per-row weight fraction is reached.

    Implementation details
    ----------------------
    1) Compute per-row top-k_max edges (sorted descending) and full row sums.
    2) For each row, keep the smallest prefix whose cumulative weight reaches
       `weight_fraction * row_sum`, with a floor of `min_keep` edges.
    3) Materialize a CSR matrix from the kept edges.

    The per-row cap (k_max) is critical to keep runtime and output size bounded.
    If the target fraction cannot be met within k_max edges, we keep all k_max.
    """
    mat = mat.tocsr()
    mat.eliminate_zeros()

    top_idx, top_w, counts, row_sum = _topk_per_row(
        mat.indptr,
        mat.indices.astype(np.int32),
        mat.data,
        int(k_max),
    )

    keep_counts = _cumweight_keep_counts(
        top_w,
        counts,
        row_sum,
        float(weight_fraction),
        int(min_keep),
    )

    indptr = _prefix_indptr(keep_counts)
    out_idx, out_w = _flatten_topk(top_idx, top_w, keep_counts, indptr)
    pruned = csr_matrix((out_w, out_idx, indptr), shape=mat.shape)
    pruned.sum_duplicates()
    pruned.eliminate_zeros()
    return pruned


def execute_clusters_infomap(
    min_cluster_size: int = _DEFAULT_MIN_CLUSTER_SIZE,
    transformed_matrix_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    out_name: str = "tm_modules",
    seed: int = 1,
    num_trials: int = 10,
    silent: bool = True,
    k_per_row: Optional[int] = None,
    enforce_min_cluster_size: bool = True,
    infomap_markov_time: float = _DEFAULT_INFOMAP_MARKOV_TIME,
    weight_fraction_per_row: Optional[float] = None,
    min_edges_per_row: int = 3,
    protect_lcc: bool = True,
    num_threads: int = 1,
    curation_tol: float = 1e-4,
    max_curation_iter: int = 50,
) -> np.ndarray:

    # Resolve transformed_matrix.npz path
    tm_path = transformed_matrix_path
    if tm_path is None:
        gdp = _global_data_path()
        if gdp is None:
            raise ValueError(
                "transformed_matrix_path was not provided and sysOps.globaldatapath is unavailable. "
                "Pass transformed_matrix_path explicitly."
            )
        tm_path = os.path.join(gdp, "transformed_matrix.npz")

    if not os.path.exists(tm_path):
        raise FileNotFoundError(f"Expected transformed transport graph at {tm_path} but it does not exist.")

    _status(f"Loading transformed transport graph (infomap): {tm_path}")
    T = load_npz(tm_path).tocsr()
    n = int(T.shape[0])
    _ensure_square(T)

    # Remove self-loops
    T.setdiag(0)
    T.eliminate_zeros()

    # Optionally protect the largest connected component (computed pre-prune).
    # This is used to minimize changes to nodes that are already in the dominant
    # component when we later prune edges aggressively.
    lcc_mask = None
    row_sum_full = None
    if bool(protect_lcc):
        _status("(infomap) Detecting largest connected component (pre-prune).")
        lcc_mask = _largest_connected_component_mask(T)
        row_sum_full = np.asarray(T.sum(axis=1)).reshape(-1)

    # Choose k (top edges per row).
    # If weight_fraction_per_row is provided, k is treated as a *cap* (k_max).
    if k_per_row is None:
        k = 10 # will bound the weights
    else:
        k = int(k_per_row)
        if k < 1:
            raise ValueError("k_per_row must be >= 1")

    if weight_fraction_per_row is not None:
        frac = float(weight_fraction_per_row)
        if not (0.0 <= frac <= 1.0):
            raise ValueError("weight_fraction_per_row must be in [0, 1].")
        _status(
            f"(infomap) Pruning transformed graph by per-row cumulative weight fraction={frac:.6g} "
            f"with cap k_max={k} and min_edges_per_row={int(min_edges_per_row)}."
        )
        P = _prune_cumweight_csr(T, k_max=k, weight_fraction=frac, min_keep=int(min_edges_per_row))
    else:
        _status(f"(infomap) Pruning transformed graph to ~{k} strongest edges per node.")
        P = _prune_topk_csr(T, k)

    # Symmetrize after pruning.
    # Use union (max) instead of averaging so edges retained by either endpoint
    # keep their original weight (important when preserving row-mass via self-loops).
    P = P.maximum(P.T)
    P.eliminate_zeros()

    # LCC-protected PruneRenorm:
    #   keep retained weights unchanged for nodes in the pre-prune LCC by
    #   allocating dropped row-mass to a self-loop.
    # This preserves transition probabilities on the kept edges under Infomap's
    # internal row-normalization.
    include_diagonal = False
    if bool(protect_lcc) and (lcc_mask is not None) and (row_sum_full is not None):
        row_sum_kept = np.asarray(P.sum(axis=1)).reshape(-1)
        slack = row_sum_full - row_sum_kept
        # Numerical guard (should be >=0 for symmetric inputs + max-symmetrize)
        slack[slack < 0.0] = 0.0

        diag = np.zeros(int(n), dtype=np.float32)
        diag[lcc_mask] = slack[lcc_mask].astype(np.float32, copy=False)

        # Insert only the diagonal slack (off-diagonal weights remain unchanged).
        P.setdiag(diag)
        P.eliminate_zeros()
        include_diagonal = True

    # Resolve output directory for Infomap artifacts
    if out_dir is None:
        gdp = _global_data_path()
        if gdp is not None:
            out_dir = os.path.join(gdp, "tmp", "infomap_tm")
        else:
            out_dir = os.path.join(os.getcwd(), "tmp", "infomap_tm")
    os.makedirs(out_dir, exist_ok=True)

    link_path = os.path.join(out_dir, "tm_links.txt")
    _status("(infomap) Writing link-list file for Infomap (upper triangle, weighted).")
    _write_link_list_upper_txt(P, link_path, one_based=True, include_diagonal=include_diagonal)

    _status("(infomap) Running Infomap (two-level + clu output).")
    clu_path = _run_infomap_link_list(
        link_path=link_path,
        out_dir=out_dir,
        out_name=out_name,
        seed=seed,
        num_trials=num_trials,
        silent=silent,
        markov_time=infomap_markov_time,
        num_threads=num_threads,    
    )

    labels = _read_infomap_clu(clu_path, n_nodes=n).astype(np.int32, copy=False)

    # Mirror the HDBSCAN flow: treat the initial Infomap partition as the
    # starting labeling, optionally convert undersized modules to noise (-1),
    # then iteratively curate noise labels until convergence.
    if enforce_min_cluster_size and min_cluster_size is not None and int(min_cluster_size) > 1:
        valid = labels >= 0
        if np.any(valid):
            sizes = np.bincount(labels[valid])
            small = np.where(sizes < int(min_cluster_size))[0]
            if small.size > 0:
                labels = labels.copy()
                labels[np.isin(labels, small)] = -1

    # Curate labels until convergence, matching execute_clusters_hdbscan()'s
    # majority-vote refinement pattern while preserving the Infomap graph path.
    for it in range(int(max_curation_iter)):
        revised = curate_labels_with_graph(labels, P)
        modified_frac = float(np.mean(revised != labels))
        _status(f"(infomap) Modified fraction on iteration {it} = {modified_frac:.6g}")
        labels = revised
        if modified_frac < float(curation_tol):
            break

    labels = _remap_contiguous(labels)

    _status("Infomap clustering complete (infomap).")
    return labels


def execute_clusters_leiden(
    min_cluster_size: int = _DEFAULT_MIN_CLUSTER_SIZE,
    transformed_matrix_path: Optional[str] = None,
    seed: int = 1,
    k_per_row: Optional[int] = None,
    weight_fraction_per_row: Optional[float] = None,
    min_edges_per_row: int = 3,
    protect_lcc: bool = True,
    curation_tol: float = 1e-4,
    max_curation_iter: int = 50,
    leiden_resolution: float = 1.0,
    partition_type: str = "rb",
) -> np.ndarray:
    """Run Leiden on the same transformed transport graph used for final Infomap.

    The graph-loading and pruning path intentionally mirrors
    :func:`execute_clusters_infomap`, but omits the Infomap-specific diagonal
    slack/self-loop trick before partitioning.  Small clusters are converted to
    noise (-1) and then majority-vote curated on the pruned transport graph,
    matching the existing Infomap post-processing flow.
    """
    try:
        import igraph as ig  # type: ignore
        import leidenalg as la  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Leiden clustering requires python-igraph and leidenalg to be installed."
        ) from e

    tm_path = _resolve_transformed_matrix_path(transformed_matrix_path)
    _status(f"Loading transformed transport graph (leiden): {tm_path}")
    T = load_npz(tm_path).tocsr()
    n = int(T.shape[0])
    _ensure_square(T)

    T.setdiag(0)
    T.eliminate_zeros()

    if bool(protect_lcc):
        _status("(leiden) Detecting largest connected component (pre-prune).")
        _ = _largest_connected_component_mask(T)

    if k_per_row is None:
        k = 10
    else:
        k = int(k_per_row)
        if k < 1:
            raise ValueError("k_per_row must be >= 1")

    if weight_fraction_per_row is not None:
        frac = float(weight_fraction_per_row)
        if not (0.0 <= frac <= 1.0):
            raise ValueError("weight_fraction_per_row must be in [0, 1].")
        _status(
            f"(leiden) Pruning transformed graph by per-row cumulative weight fraction={frac:.6g} "
            f"with cap k_max={k} and min_edges_per_row={int(min_edges_per_row)}."
        )
        P = _prune_cumweight_csr(T, k_max=k, weight_fraction=frac, min_keep=int(min_edges_per_row))
    else:
        _status(f"(leiden) Pruning transformed graph to ~{k} strongest edges per node.")
        P = _prune_topk_csr(T, k)

    P = P.maximum(P.T).tocsr()
    P.setdiag(0)
    P.eliminate_zeros()

    if P.nnz == 0:
        _status("(leiden) Pruned graph is empty; returning singleton labels.")
        labels = np.arange(n, dtype=np.int32)
    else:
        edge_coo = triu(P, k=1).tocoo()
        edges = list(zip(edge_coo.row.tolist(), edge_coo.col.tolist()))
        weights = edge_coo.data.astype(float).tolist()

        graph = ig.Graph(n=n, edges=edges, directed=False)
        graph.es["weight"] = weights

        part_key = str(partition_type).strip().lower()
        if part_key == "cpm":
            partition_cls = la.CPMVertexPartition
        elif part_key in {"rb", "rbconfiguration", "modularity"}:
            partition_cls = la.RBConfigurationVertexPartition
        else:
            raise ValueError("partition_type must be one of {'rb', 'cpm'}")

        partition = la.find_partition(
            graph,
            partition_cls,
            weights="weight",
            seed=int(seed),
            resolution_parameter=float(leiden_resolution),
        )
        labels = np.asarray(partition.membership, dtype=np.int32)

    if min_cluster_size is not None and int(min_cluster_size) > 1:
        valid = labels >= 0
        if np.any(valid):
            sizes = np.bincount(labels[valid])
            small = np.where(sizes < int(min_cluster_size))[0]
            if small.size > 0:
                labels = labels.copy()
                labels[np.isin(labels, small)] = -1

    for it in range(int(max_curation_iter)):
        revised = curate_labels_with_graph(labels, P)
        modified_frac = float(np.mean(revised != labels))
        _status(f"(leiden) Modified fraction on iteration {it} = {modified_frac:.6g}")
        labels = revised
        if modified_frac < float(curation_tol):
            break

    labels = _remap_contiguous(labels)
    _status("Leiden clustering complete (leiden).")
    return labels.astype(np.int32, copy=False)


# -----------------------------------------------------------------------------
# Version 3: recursive sparsest-cut on transformed transport graph
# -----------------------------------------------------------------------------


@jit(nopython=True, parallel=True)
def _weighted_mean_square_edge_dispersion(indptr, indices, data, coords):
    sigma2 = np.empty(indptr.shape[0] - 1, dtype=np.float64)
    n = indptr.shape[0] - 1
    dims = coords.shape[1]
    for i in prange(n):
        start = indptr[i]
        end = indptr[i + 1]
        weighted_sum = 0.0
        weight_total = 0.0
        for p in range(start, end):
            j = indices[p]
            w = float(data[p])
            if w <= 0.0:
                continue
            dist2 = 0.0
            for d in range(dims):
                diff = coords[i, d] - coords[j, d]
                dist2 += diff * diff
            weighted_sum += w * dist2
            weight_total += w
        if weight_total > 0.0:
            sigma2[i] = weighted_sum / weight_total
        else:
            sigma2[i] = np.nan
    return sigma2


@jit(nopython=True, parallel=True)
def _adaptive_gaussian_reweight_csr_data(indptr, indices, data, coords, sigma2, log_prefactor, min_sigma2_sum):
    out = np.empty(data.shape[0], dtype=np.float64)
    n = indptr.shape[0] - 1
    dims = coords.shape[1]
    for i in prange(n):
        start = indptr[i]
        end = indptr[i + 1]
        sigma_i2 = sigma2[i]
        for p in range(start, end):
            j = indices[p]
            sigma_sum = 0.5*(sigma_i2 + sigma2[j])
            if sigma_sum < min_sigma2_sum:
                sigma_sum = min_sigma2_sum
            dist2 = 0.0
            for d in range(dims):
                diff = coords[i, d] - coords[j, d]
                dist2 += diff * diff
            out[p] = float(data[p]) * math.exp(log_prefactor * math.log(sigma_sum) - dist2 / sigma_sum)
    return out


def _apply_adaptive_gaussian_edge_reweighting(
    mat: csr_matrix,
    coords: np.ndarray,
) -> csr_matrix:
    """Multiply each edge weight by an adaptive Gaussian factor.

    The multiplier is

        (sigma_i^2 + sigma_j^2)^(-d/2) * exp(-||x_i - x_j||^2 / (sigma_i^2 + sigma_j^2))

    where ``d`` is the coordinate dimension and ``sigma_i^2`` is the weighted
    mean squared edge dispersion of node ``i`` over its incident graph edges.
    """
    A = mat.tocsr(copy=True)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError(f"Xpts must be 2D; got shape={coords.shape}")
    if coords.shape[0] != A.shape[0]:
        raise ValueError(
            f"Xpts has {coords.shape[0]} rows but the transport graph has {A.shape[0]} nodes."
        )

    sigma2 = _weighted_mean_square_edge_dispersion(
        A.indptr,
        A.indices,
        A.data.astype(np.float64, copy=False),
        coords,
    )
    good = np.isfinite(sigma2) & (sigma2 > 0.0)
    if np.any(good):
        fallback_sigma2 = float(np.mean(sigma2[good]))
        if (not np.isfinite(fallback_sigma2)) or fallback_sigma2 <= 0.0:
            fallback_sigma2 = 1.0
    else:
        fallback_sigma2 = 1.0
    sigma2 = sigma2.copy()
    sigma2[~good] = fallback_sigma2

    min_sigma2_sum = max(1e-12, 1e-12 * float(fallback_sigma2))
    log_prefactor = -0.5 * float(coords.shape[1])
    A.data = _adaptive_gaussian_reweight_csr_data(
        A.indptr,
        A.indices,
        A.data.astype(np.float64, copy=False),
        coords,
        sigma2,
        float(log_prefactor),
        float(min_sigma2_sum),
    )
    A.eliminate_zeros()
    return A


def _component_groups_from_labels(labels: np.ndarray) -> list[np.ndarray]:
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    order = np.argsort(labels, kind="mergesort")
    labels_ord = labels[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(labels_ord)) + 1, [labels_ord.size]))
    return [order[starts[i] : starts[i + 1]] for i in range(starts.size - 1)]


def _make_normalized_adjacency_operator(A: csr_matrix, degree: np.ndarray) -> tuple[LinearOperator, np.ndarray]:
    invsqrt = np.zeros_like(degree, dtype=np.float64)
    good = degree > 0.0
    invsqrt[good] = 1.0 / np.sqrt(degree[good])

    def _mv(x: np.ndarray) -> np.ndarray:
        y = invsqrt * np.asarray(x, dtype=np.float64)
        y = A @ y
        return invsqrt * np.asarray(y, dtype=np.float64).reshape(-1)

    op = LinearOperator(A.shape, matvec=_mv, rmatvec=_mv, dtype=np.float64)
    return op, invsqrt


def _power_second_normalized_eigenvector(
    op: LinearOperator,
    trivial_vec: np.ndarray,
    *,
    max_iter: int,
    tol: float,
    random_state: int,
) -> np.ndarray:
    n = int(trivial_vec.shape[0])
    rng = np.random.default_rng(int(random_state))

    x = rng.standard_normal(n)
    x = x - trivial_vec * float(np.dot(trivial_vec, x))
    norm_x = float(np.linalg.norm(x))
    if (not np.isfinite(norm_x)) or norm_x <= 0.0:
        x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        x = x - trivial_vec * float(np.dot(trivial_vec, x))
        norm_x = float(np.linalg.norm(x))
        if norm_x <= 0.0:
            return np.linspace(-1.0, 1.0, n, dtype=np.float64)
    x /= norm_x

    for _ in range(max(1, int(max_iter))):
        y = np.asarray(op.matvec(x), dtype=np.float64).reshape(-1)
        y = y - trivial_vec * float(np.dot(trivial_vec, y))
        norm_y = float(np.linalg.norm(y))
        if (not np.isfinite(norm_y)) or norm_y <= 0.0:
            break
        y /= norm_y
        delta = min(float(np.linalg.norm(y - x)), float(np.linalg.norm(y + x)))
        x = y
        if delta <= float(tol):
            break

    return x


def _second_normalized_eigenvector(
    A: csr_matrix,
    degree: np.ndarray,
    *,
    exact_eig_max_nodes: int,
    exact_eig_max_edges: int,
    power_max_iter: int,
    power_tol: float,
    eig_tol: float,
    eig_maxiter: Optional[int],
    random_state: int,
) -> np.ndarray:
    n = int(A.shape[0])
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    if n == 2:
        return np.asarray([-1.0, 1.0], dtype=np.float64)

    trivial = np.sqrt(np.maximum(degree, 0.0))
    trivial_norm = float(np.linalg.norm(trivial))
    if trivial_norm <= 0.0:
        return np.linspace(-1.0, 1.0, n, dtype=np.float64)
    trivial /= trivial_norm

    op, invsqrt = _make_normalized_adjacency_operator(A, degree)

    if n <= 4:
        dense = A.toarray().astype(np.float64, copy=False)
        dense = (invsqrt[:, None] * dense) * invsqrt[None, :]
        evals, evecs = np.linalg.eigh(dense)
        order = np.argsort(evals)[::-1]
        best = None
        best_overlap = np.inf
        for idx in order:
            vec = np.asarray(evecs[:, idx], dtype=np.float64)
            overlap = abs(float(np.dot(vec, trivial)))
            if overlap < best_overlap:
                best_overlap = overlap
                best = vec
        if best is None:
            best = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        best = best - trivial * float(np.dot(trivial, best))
        norm_best = float(np.linalg.norm(best))
        if norm_best > 0.0:
            return best / norm_best
        return np.linspace(-1.0, 1.0, n, dtype=np.float64)

    use_exact = (n <= int(exact_eig_max_nodes)) and (int(A.nnz) <= int(exact_eig_max_edges))
    if use_exact:
        try:
            evals, evecs = eigsh(
                op,
                k=2,
                which="LA",
                tol=float(eig_tol),
                maxiter=eig_maxiter,
            )
            overlaps = np.abs(np.asarray(evecs.T @ trivial, dtype=np.float64).reshape(-1))
            idx = int(np.argmin(overlaps))
            vec = np.asarray(evecs[:, idx], dtype=np.float64).reshape(-1)
            vec = vec - trivial * float(np.dot(trivial, vec))
            norm_vec = float(np.linalg.norm(vec))
            if np.isfinite(norm_vec) and norm_vec > 0.0:
                return vec / norm_vec
        except Exception as exc:
            _status(
                f"(sparsest transport) eigsh failed on block size={n}, nnz={int(A.nnz)}; "
                f"falling back to power iteration. {exc}"
            )

    return _power_second_normalized_eigenvector(
        op,
        trivial,
        max_iter=int(power_max_iter),
        tol=float(power_tol),
        random_state=int(random_state),
    )


@jit(nopython=True)
def _sweep_best_conductance_cut(indptr, indices, data, order, degree, min_side_size):
    n = order.shape[0]
    total_volume = 0.0
    for i in range(n):
        total_volume += degree[i]

    in_left = np.zeros(n, dtype=np.uint8)
    left_volume = 0.0
    cut_weight = 0.0
    cut_edges = 0
    best_cond = np.inf
    best_edges = -1
    best_idx = -1

    for t in range(n):
        node = order[t]
        row_start = indptr[node]
        row_end = indptr[node + 1]
        if t == 0:
            cut_weight = degree[node]
            cut_edges = row_end - row_start
            in_left[node] = 1
            left_volume = degree[node]
        else:
            for p in range(row_start, row_end):
                j = indices[p]
                w = data[p]
                if in_left[j] == 1:
                    cut_weight -= w
                    cut_edges -= 1
                else:
                    cut_weight += w
                    cut_edges += 1
            in_left[node] = 1
            left_volume += degree[node]

        left_size = t + 1
        right_size = n - left_size
        if left_size < min_side_size or right_size < min_side_size:
            continue

        right_volume = total_volume - left_volume
        if left_volume <= 0.0 or right_volume <= 0.0:
            continue

        denom = left_volume if left_volume < right_volume else right_volume
        cond = cut_weight / denom
        if cond < best_cond:
            best_cond = cond
            best_edges = cut_edges
            best_idx = t

    mask = np.zeros(n, dtype=np.bool_)
    if best_idx >= 0:
        for t in range(best_idx + 1):
            mask[order[t]] = True

    return best_cond, best_edges, mask


def _weighted_kcore_mask(A: csr_matrix, threshold: float) -> np.ndarray:
    n = int(A.shape[0])
    if float(threshold) <= 0.0 or n == 0:
        return np.ones(n, dtype=bool)

    indptr = A.indptr
    indices = A.indices
    data = A.data.astype(np.float64, copy=False)
    degree = np.asarray(A.sum(axis=1)).reshape(-1).astype(np.float64, copy=False)
    alive = np.ones(n, dtype=bool)
    q: deque[int] = deque(int(i) for i in np.flatnonzero(degree < float(threshold)))

    while q:
        i = int(q.popleft())
        if not alive[i]:
            continue
        alive[i] = False
        for p in range(indptr[i], indptr[i + 1]):
            j = int(indices[p])
            if alive[j]:
                degree[j] -= float(data[p])
                if degree[j] < float(threshold):
                    q.append(j)

    return alive


def _pruned_connected_children(
    nodes_global: np.ndarray,
    A_local: csr_matrix,
    *,
    min_within_weight: float,
) -> list[tuple[np.ndarray, csr_matrix]]:
    out: list[tuple[np.ndarray, csr_matrix]] = []
    n = int(nodes_global.shape[0])
    if n == 0:
        return out
    if n == 1:
        out.append((nodes_global.astype(np.int64, copy=False), csr_matrix((1, 1), dtype=A_local.dtype)))
        return out

    alive = _weighted_kcore_mask(A_local, float(min_within_weight))
    dropped = nodes_global[~alive]
    for node in dropped:
        out.append((np.asarray([int(node)], dtype=np.int64), csr_matrix((1, 1), dtype=A_local.dtype)))

    if not np.any(alive):
        return out

    kept_nodes = nodes_global[alive].astype(np.int64, copy=False)
    A_kept = A_local[alive][:, alive].tocsr()
    if kept_nodes.size == 1:
        out.append((kept_nodes, csr_matrix((1, 1), dtype=A_kept.dtype)))
        return out

    n_comp, comp_ids = connected_components(A_kept, directed=False, return_labels=True)
    if int(n_comp) <= 1:
        out.append((kept_nodes, A_kept))
        return out

    for loc in _component_groups_from_labels(comp_ids):
        child_nodes = kept_nodes[loc].astype(np.int64, copy=False)
        child_A = A_kept[loc][:, loc].tocsr()
        out.append((child_nodes, child_A))

    return out


def _recursive_sparsest_cut_transport_labels(
    A: csr_matrix,
    *,
    min_cluster_size: int,
    stopping_conductance: Optional[float],
    stopping_assoc: Optional[int],
    max_cluster_size: Optional[int],
    min_within_weight: float,
    exact_eig_max_nodes: int,
    exact_eig_max_edges: int,
    power_max_iter: int,
    power_tol: float,
    eig_tol: float,
    eig_maxiter: Optional[int],
    random_state: int,
) -> np.ndarray:
    A = A.tocsr(copy=True)
    A.sum_duplicates()
    A.setdiag(0)
    A.eliminate_zeros()
    _ensure_square(A)

    n = int(A.shape[0])
    if n == 0:
        return np.empty(0, dtype=np.int32)
    if A.nnz == 0:
        return np.arange(n, dtype=np.int32)

    labels = np.full(n, -1, dtype=np.int32)
    next_label = 0
    processed_blocks = 0
    accepted_splits = 0

    n_root_comp, root_comp_ids = connected_components(A, directed=False, return_labels=True)
    root_groups = _component_groups_from_labels(root_comp_ids)
    root_groups.sort(key=lambda idx: int(idx.size))

    stack: list[tuple[np.ndarray, csr_matrix, int]] = []
    for nodes in root_groups:
        nodes = nodes.astype(np.int64, copy=False)
        stack.append((nodes, A[nodes][:, nodes].tocsr(), 0))

    while stack:
        nodes_global, A_local, depth = stack.pop()
        processed_blocks += 1
        block_n = int(nodes_global.shape[0])

        if block_n == 0:
            continue
        if block_n == 1:
            labels[int(nodes_global[0])] = int(next_label)
            next_label += 1
            continue

        A_local = A_local.tocsr(copy=False)
        A_local.sum_duplicates()
        A_local.setdiag(0)
        A_local.eliminate_zeros()

        if A_local.nnz == 0:
            for node in nodes_global:
                labels[int(node)] = int(next_label)
                next_label += 1
            continue

        if block_n < max(2, 2 * int(min_cluster_size)):
            labels[nodes_global] = int(next_label)
            next_label += 1
            continue

        degree = np.asarray(A_local.sum(axis=1)).reshape(-1).astype(np.float64, copy=False)
        if np.count_nonzero(degree > 0.0) < 2:
            labels[nodes_global] = int(next_label)
            next_label += 1
            continue

        cut_vec = _second_normalized_eigenvector(
            A_local,
            degree,
            exact_eig_max_nodes=int(exact_eig_max_nodes),
            exact_eig_max_edges=int(exact_eig_max_edges),
            power_max_iter=int(power_max_iter),
            power_tol=float(power_tol),
            eig_tol=float(eig_tol),
            eig_maxiter=eig_maxiter,
            random_state=int(random_state) + int(depth) + int(block_n),
        )
        order = np.argsort(cut_vec).astype(np.int64, copy=False)
        cut_cond, cut_edges, left_mask = _sweep_best_conductance_cut(
            A_local.indptr,
            A_local.indices,
            A_local.data.astype(np.float64, copy=False),
            order,
            degree,
            max(1, int(min_cluster_size)),
        )

        no_valid_cut = (not np.isfinite(cut_cond)) or (cut_edges < 0) or (not np.any(left_mask)) or bool(np.all(left_mask))
        stop_here = no_valid_cut
        if not stop_here:
            stop_here = (
                (stopping_conductance is None or float(cut_cond) >= float(stopping_conductance))
                and (stopping_assoc is None or int(cut_edges) >= int(stopping_assoc))
                and (max_cluster_size is None or block_n <= int(max_cluster_size))
            )

        if stop_here:
            labels[nodes_global] = int(next_label)
            next_label += 1
            if block_n >= 100000 or processed_blocks % 50 == 0:
                _status(
                    f"(sparsest transport) finalized block size={block_n}, depth={depth}, "
                    f"cut_cond={float(cut_cond):.6g}, cut_edges={int(cut_edges)}."
                )
            continue

        accepted_splits += 1
        left_nodes = nodes_global[left_mask].astype(np.int64, copy=False)
        right_nodes = nodes_global[~left_mask].astype(np.int64, copy=False)
        A_left = A_local[left_mask][:, left_mask].tocsr()
        A_right = A_local[~left_mask][:, ~left_mask].tocsr()

        children = _pruned_connected_children(
            left_nodes,
            A_left,
            min_within_weight=float(min_within_weight),
        )
        children.extend(
            _pruned_connected_children(
                right_nodes,
                A_right,
                min_within_weight=float(min_within_weight),
            )
        )

        if len(children) <= 1:
            labels[nodes_global] = int(next_label)
            next_label += 1
            continue

        children.sort(key=lambda item: int(item[0].shape[0]))
        if block_n >= 50000 or accepted_splits % 25 == 0:
            child_sizes = [int(child_nodes.shape[0]) for child_nodes, _ in children]
            _status(
                f"(sparsest transport) split block size={block_n}, depth={depth}, "
                f"cut_cond={float(cut_cond):.6g}, cut_edges={int(cut_edges)}, children={child_sizes[:8]}"
                + ("..." if len(child_sizes) > 8 else "")
            )

        for child_nodes, child_A in children:
            stack.append((child_nodes, child_A, depth + 1))

    if np.any(labels < 0):
        missing = np.flatnonzero(labels < 0)
        for node in missing:
            labels[int(node)] = int(next_label)
            next_label += 1

    return _remap_contiguous(labels).astype(np.int32, copy=False)


def execute_clusters_sparsest_transport(
    min_cluster_size: int = _DEFAULT_MIN_CLUSTER_SIZE,
    transformed_matrix_path: Optional[str] = None,
    k_per_row: Optional[int] = None,
    weight_fraction_per_row: Optional[float] = 0.90,
    min_edges_per_row: Optional[int] = None,
    *,
    Xpts: Optional[np.ndarray] = None,
    transport_adjacency: Optional[Union[np.ndarray, csr_matrix]] = None,
    reweight_by_gaussian: bool = True,
    stopping_conductance: Optional[float] = 0.25,
    stopping_assoc: Optional[int] = None,
    max_cluster_size: Optional[int] = None,
    min_within_weight: float = 0.0,
    exact_eig_max_nodes: int = 50000,
    exact_eig_max_edges: int = 5000000,
    power_max_iter: int = 32,
    power_tol: float = 1e-4,
    eig_tol: float = 1e-3,
    eig_maxiter: Optional[int] = None,
    random_state: int = 1,
) -> np.ndarray:
    """Cluster the transformed transport graph with recursive spectral sparsest cuts.

    Notes
    -----
    * This is the primary transport clustering route. Scalability comes from:
        1) optional per-row pruning before recursion,
        2) depth-first processing of connected components,
        3) approximate second-eigenvector power iteration on large blocks,
        4) exact `eigsh` only on smaller blocks, and
        5) connected-component / weighted-k-core cleanup after every cut.
    * If ``reweight_by_gaussian=True`` (default), each edge weight is multiplied by

          (sigma_i^2 + sigma_j^2)^(-d/2) * exp(-||x_i - x_j||^2 / (sigma_i^2 + sigma_j^2))

      where ``d`` is the coordinate dimension and ``sigma_i^2`` is the weighted
      mean squared edge dispersion of node ``i`` over its incident edges.
    """
    if min_cluster_size is None or int(min_cluster_size) < 1:
        raise ValueError("min_cluster_size must be >= 1")

    if transport_adjacency is None:
        tm_path = _resolve_transformed_matrix_path(transformed_matrix_path)
        _status(f"Loading transformed transport graph (sparsest transport): {tm_path}")
        T = load_npz(tm_path).tocsr()
    else:
        T = _as_csr(transport_adjacency).tocsr()
        _status("Loading transformed transport graph (sparsest transport) from in-memory adjacency.")

    _ensure_square(T)
    T += T.T
    n = int(T.shape[0])
    if n == 0:
        return np.empty(0, dtype=np.int32)

    T.setdiag(0)
    T.eliminate_zeros()
    if np.any(T.data < 0):
        _status("(sparsest transport) Warning: negative weights detected; clipping to zero.")
        T = T.copy()
        T.data = np.maximum(T.data, 0.0)
        T.eliminate_zeros()

    if bool(reweight_by_gaussian):
        if Xpts is None:
            raise ValueError(
                "reweight_by_gaussian=True requires Xpts so adaptive Gaussian edge weights can be applied."
            )
        _status("(sparsest transport) Applying adaptive Gaussian edge reweighting from nodewise edge dispersions.")
        T = _apply_adaptive_gaussian_edge_reweighting(T, np.asarray(Xpts))

    if T.nnz == 0:
        _status("(sparsest transport) Graph is empty after preprocessing; returning singleton labels.")
        return np.arange(n, dtype=np.int32)

    if min_edges_per_row is None:
        min_keep = 3
    else:
        min_keep = max(1, int(min_edges_per_row))

    if k_per_row is None:
        k_eff = max(64, min_keep)
    else:
        k_eff = max(int(k_per_row), min_keep)

    if weight_fraction_per_row is not None:
        frac = float(weight_fraction_per_row)
        if not (0.0 <= frac <= 1.0):
            raise ValueError("weight_fraction_per_row must be in [0, 1].")
        _status(
            f"(sparsest transport) Pruning transformed graph by per-row cumulative weight fraction={frac:.6g} "
            f"with cap k_max={k_eff} and min_edges_per_row={int(min_keep)}."
        )
        P = _prune_cumweight_csr(T, k_max=k_eff, weight_fraction=frac, min_keep=min_keep)
    else:
        _status(f"(sparsest transport) Pruning transformed graph to ~{k_eff} strongest edges per node.")
        P = _prune_topk_csr(T, k_eff)

    P = P.maximum(P.T).tocsr()
    P.setdiag(0)
    P.eliminate_zeros()

    if P.nnz == 0:
        _status("(sparsest transport) Pruned graph is empty; returning singleton labels.")
        return np.arange(n, dtype=np.int32)

    _status(
        f"(sparsest transport) Beginning recursive spectral cuts on n={n} nodes, nnz={int(P.nnz)} stored edges."
    )
    labels = _recursive_sparsest_cut_transport_labels(
        P,
        min_cluster_size=int(min_cluster_size),
        stopping_conductance=stopping_conductance,
        stopping_assoc=stopping_assoc,
        max_cluster_size=max_cluster_size,
        min_within_weight=float(min_within_weight),
        exact_eig_max_nodes=int(exact_eig_max_nodes),
        exact_eig_max_edges=int(exact_eig_max_edges),
        power_max_iter=int(power_max_iter),
        power_tol=float(power_tol),
        eig_tol=float(eig_tol),
        eig_maxiter=eig_maxiter,
        random_state=int(random_state),
    )
    _status("Transport-graph recursive sparsest-cut clustering complete (sparsest transport).")
    return labels.astype(np.int32, copy=False)


# -----------------------------------------------------------------------------
# Infomap multi-seed runner (reuses the same pruned graph + link-list)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Dual runner (returns Nx2 labels)
# -----------------------------------------------------------------------------


def execute_clusters(
    Xpts: Optional[np.ndarray],
    adjacency: Optional[Union[np.ndarray, csr_matrix]],
    min_cluster_size: int = _DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = _DEFAULT_HDBSCAN_MIN_SAMPLES,
    which: str = "both",
    # hdbscan tuning
    hdbscan_curation_tol: float = 1e-4,
    hdbscan_max_curation_iter: int = 50,
    hdbscan_symmetrize: bool = True,
    # infomap tuning
    transformed_matrix_path: Optional[str] = None,
    infomap_out_dir: Optional[str] = None,
    infomap_out_name: str = "tm_modules",
    infomap_seed: int = 1,
    infomap_num_trials: int = 10,
    infomap_silent: bool = True,
    infomap_k_per_row: Optional[int] = None,
    infomap_enforce_min_cluster_size: bool = True,
    infomap_markov_time: float = _DEFAULT_INFOMAP_MARKOV_TIME,
    infomap_weight_fraction_per_row: Optional[float] = None,
    infomap_min_edges_per_row: int = 3,
    infomap_protect_lcc: bool = True,
    infomap_curation_tol: float = 1e-4,
    infomap_max_curation_iter: int = 50,
) -> np.ndarray:
    """Compute segmentation labels using hdbscan, infomap, or both.

    Returns
    -------
    labels_dual : np.ndarray, shape (N, 2), dtype int32
        Column 0 = hdbscan labels (or -1 if not computed)
        Column 1 = infomap labels (or -1 if not computed)
    """
    w = _normalize_which(which)

    labels_hdbscan: Optional[np.ndarray] = None
    labels_infomap: Optional[np.ndarray] = None

    n: Optional[int] = None
    if Xpts is not None:
        n = int(Xpts.shape[0])
    elif adjacency is not None:
        n = int(_as_csr(adjacency).shape[0])

    if w in {"hdbscan", "both"}:
        if Xpts is None:
            raise ValueError("which includes hdbscan but Xpts is None")
        if adjacency is None:
            raise ValueError("which includes hdbscan but adjacency is None")
        labels_hdbscan = execute_clusters_hdbscan(
            Xpts=Xpts,
            adjacency=adjacency,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            curation_tol=hdbscan_curation_tol,
            max_curation_iter=hdbscan_max_curation_iter,
            symmetrize=hdbscan_symmetrize,
        )
        n = int(labels_hdbscan.shape[0])

    if w in {"infomap", "both"}:
        labels_infomap = execute_clusters_infomap(
            min_cluster_size=min_cluster_size,
            transformed_matrix_path=transformed_matrix_path,
            out_dir=infomap_out_dir,
            out_name=infomap_out_name,
            seed=infomap_seed,
            num_trials=infomap_num_trials,
            silent=infomap_silent,
            k_per_row=infomap_k_per_row,
            enforce_min_cluster_size=infomap_enforce_min_cluster_size,
            infomap_markov_time=infomap_markov_time,
            weight_fraction_per_row=infomap_weight_fraction_per_row,
            min_edges_per_row=infomap_min_edges_per_row,
            protect_lcc=infomap_protect_lcc,
            curation_tol=infomap_curation_tol,
            max_curation_iter=infomap_max_curation_iter,
        )
        n = int(labels_infomap.shape[0])

    if n is None:
        raise ValueError("Unable to infer N. Provide Xpts and/or adjacency.")

    out = np.full((n, 2), -1, dtype=np.int32)
    if labels_hdbscan is not None:
        if labels_hdbscan.shape[0] != n:
            raise ValueError(f"hdbscan label length {labels_hdbscan.shape[0]} != N={n}")
        out[:, 0] = labels_hdbscan
    if labels_infomap is not None:
        if labels_infomap.shape[0] != n:
            raise ValueError(f"infomap label length {labels_infomap.shape[0]} != N={n}")
        out[:, 1] = labels_infomap
    return out


# Backwards-compatible alias: earlier versions of the pipeline exposed
# `execute_clusters_dual` as the main entry point.
def execute_clusters_dual(*args, **kwargs) -> np.ndarray:
    return execute_clusters(*args, **kwargs)



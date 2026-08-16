#!/usr/bin/env python3
"""
Focused infomap/kNN exact-2 pair analysis for matrix-backed AnnData/H5AD files.

This cleaned version keeps only the raw-data path needed to produce:

  infomap_cluster_knn_relative_cd3_vs_igd_map_k1_knn30.png
  knn_neighborhood_zscore_summary_d1.png

plus the CSV/JSON tables that directly back those plots.  Rows are hubs/nodes,
columns are insert/sequence features, and a positive matrix entry means that an
insert is present on a hub.  Only sequence columns with exactly two recognized
epitope motif hits become analytic insert-pair labels.  The requested plots are
computed inside one exact hub insert-count stratum, default d=1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


DEFAULT_LAYER = "auto"
DEFAULT_SEQUENCE_VAR = "auto"
DEFAULT_X_COL = "auto"
DEFAULT_Y_COL = "auto"
DEFAULT_KNN_KS = "15,30,50,100,200"
DEFAULT_N_PERMUTATIONS = 10000
DEFAULT_RANDOM_SEED = 13
DEFAULT_EXACT_HUB_INSERT_COUNT = 1
DEFAULT_INFOMAP_CLUSTER_COL = "cluster_infomap"
DEFAULT_INFOMAP_CELLTYPE_KNN_K = 30
DEFAULT_LOG2_RELATIVE_CAP = 3.0
DEFAULT_MIN_UNITS = 5
DEFAULT_CHUNK_SIZE = 50_000
DEFAULT_NULL_SLOT_BUDGET = 400_000_000

SEQUENCE_VAR_CANDIDATES = (
    "sequence", "seq", "insert_sequence", "consensus_sequence", "cdr3_sequence", "feature_sequence",
)
LAYER_CANDIDATES = ("seq", "counts", "raw_counts")
COORD_OBS_CANDIDATE_PAIRS = (
    ("GSE_1", "GSE_2"), ("gse_1", "gse_2"), ("x", "y"), ("X", "Y"),
    ("coord_x", "coord_y"), ("x_coord", "y_coord"), ("spatial_x", "spatial_y"),
    ("UMAP_1", "UMAP_2"), ("umap_1", "umap_2"),
)
COORD_OBSM_CANDIDATES = ("spatial", "X_spatial", "X_gse", "X_umap", "X_pca")

TARGET_MOTIFS = {
    "CD9": ["GACTGATC", "GAAGTTGG"],
    "IgD": ["CGGTTGTT"],
    "CD3": ["TAAGGCTC"],
    "CD24": ["TCCGAGAT"],
}
ALIAS_TO_CANONICAL = {
    "CD9": "CD9", "CD9-2": "CD9",
    "IgD": "IgD", "IgD2": "IgD", "IgD-2": "IgD",
    "CD3": "CD3", "CD3-2": "CD3",
    "CD24": "CD24",
}
PREFERRED_MARKER_ORDER = ("CD3", "CD9", "IgD", "CD24")
BACKGROUND_DOT_SIZE = 1.2
INFOMAP_RELATIVE_DOT_SIZE = 2.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
})


@dataclass(frozen=True)
class Headers:
    matrix_source: str
    sequence_var: str
    coord_source: str
    infomap_cluster_col: str


@dataclass(frozen=True)
class PairBundle:
    node_idx: np.ndarray
    pair_idx: np.ndarray
    coords: np.ndarray | None
    pair_names: list[str]
    pair_tuples: list[tuple[str, str]]
    column_pair_idx: np.ndarray
    labeled_cols: np.ndarray
    hub_insert_count: np.ndarray
    hub_pair_counts: np.ndarray
    sequence_tagcount_hist: dict[str, int]
    n_sequence_columns: int
    n_exact2_sequence_columns: int
    n_exact2_columns_with_positive_entries: int


def canonical_target(name: str) -> str:
    return ALIAS_TO_CANONICAL.get(name, name)


def marker_order() -> list[str]:
    out: list[str] = []
    for name in (*PREFERRED_MARKER_ORDER, *TARGET_MOTIFS):
        canon = canonical_target(name)
        if canon not in out:
            out.append(canon)
    return out


MARKER_ORDER = marker_order()
MARKER_RANK = {m: i for i, m in enumerate(MARKER_ORDER)}


def marker_sort_key(marker: str) -> tuple[int, str]:
    return MARKER_RANK.get(marker, len(MARKER_RANK)), marker


def pair_sort_key(pair: tuple[str, str]) -> tuple[tuple[int, str], tuple[int, str]]:
    return marker_sort_key(pair[0]), marker_sort_key(pair[1])


def format_pair(pair: tuple[str, str]) -> str:
    return f"{pair[0]}/{pair[1]}"


def safe_token(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value)).strip("_") or "x"


def parse_int_list(spec: str | Sequence[int]) -> list[int]:
    if isinstance(spec, str):
        values = [int(x.strip()) for x in spec.replace(";", ",").split(",") if x.strip()]
    else:
        values = [int(x) for x in spec]
    if any(v < 1 for v in values):
        raise ValueError("k values must be positive integers")
    return sorted(set(values))


def choose_var_column(var, explicit: str, candidates: Sequence[str]) -> str:
    text = str(explicit).strip()
    if text in {"var_names", "var.index", "index"}:
        return "__var_names__"
    if text and text != "auto":
        if text not in var.columns:
            raise KeyError(f"Requested sequence var column {text!r} was not found")
        return text
    for candidate in candidates:
        if candidate in var.columns:
            return candidate
    return "__var_names__"


def resolve_matrix(adata, layer_spec: str):
    def check(X, source: str):
        if X.shape != (adata.n_obs, adata.n_vars):
            raise ValueError(f"{source} has shape {X.shape}, expected {(adata.n_obs, adata.n_vars)}")
        return X, source
    if layer_spec == "auto":
        for layer in LAYER_CANDIDATES:
            if layer in adata.layers:
                return check(adata.layers[layer], f"layers[{layer!r}]")
        return check(adata.X, "X")
    if layer_spec == "X":
        return check(adata.X, "X")
    if layer_spec not in adata.layers:
        raise KeyError(f"Requested layer {layer_spec!r} was not found")
    return check(adata.layers[layer_spec], f"layers[{layer_spec!r}]")


def resolve_coords(adata, x_col: str, y_col: str) -> tuple[np.ndarray, str]:
    if (x_col and x_col != "auto") or (y_col and y_col != "auto"):
        if not x_col or not y_col or x_col == "auto" or y_col == "auto":
            raise ValueError("Pass both --x-col and --y-col, or leave both as auto")
        if x_col not in adata.obs.columns or y_col not in adata.obs.columns:
            raise KeyError(f"Coordinate columns {x_col!r}, {y_col!r} were not both found")
        return np.column_stack([adata.obs[x_col].to_numpy(float), adata.obs[y_col].to_numpy(float)]), f"obs[{x_col!r}], obs[{y_col!r}]"
    for xc, yc in COORD_OBS_CANDIDATE_PAIRS:
        if xc in adata.obs.columns and yc in adata.obs.columns:
            return np.column_stack([adata.obs[xc].to_numpy(float), adata.obs[yc].to_numpy(float)]), f"obs[{xc!r}], obs[{yc!r}]"
    for key in COORD_OBSM_CANDIDATES:
        if key in adata.obsm and adata.obsm[key].shape[1] >= 2:
            return np.asarray(adata.obsm[key][:, :2], dtype=float), f"obsm[{key!r}][:, :2]"
    raise KeyError("No 2-D coordinates were found. Pass --x-col and --y-col explicitly.")


def binarize_feature_matrix(X):
    if sparse.issparse(X):
        X = X.tocsr(copy=True)
        X.data[:] = 1
        X.eliminate_zeros()
        return X
    return (np.asarray(X) != 0).astype(np.uint8, copy=False)


def count_motif(seq: str, motif: str) -> int:
    count = start = 0
    while True:
        pos = seq.find(motif, start)
        if pos == -1:
            return count
        count += 1
        start = pos + len(motif)


def epitope_hits(seq: object) -> list[str]:
    if seq is None:
        return []
    text = str(seq).upper()
    if not text or text in {"NAN", "NONE", "<NA>"}:
        return []
    hits: list[str] = []
    for target, motifs in TARGET_MOTIFS.items():
        canon = canonical_target(target)
        for motif in motifs:
            hits.extend([canon] * count_motif(text, motif))
    return hits


def label_sequence_columns(sequences: Sequence[object]) -> tuple[list[tuple[str, str] | None], dict[str, int]]:
    labels: list[tuple[str, str] | None] = []
    hist: Counter = Counter()
    for seq in sequences:
        hits = epitope_hits(seq)
        hist[str(len(hits)) if len(hits) < 3 else "3+"] += 1
        labels.append(tuple(sorted(hits, key=marker_sort_key)) if len(hits) == 2 else None)
    for key in ("0", "1", "2", "3+"):
        hist.setdefault(key, 0)
    return labels, {str(k): int(v) for k, v in hist.items()}


def build_pair_bundle(X, coords: np.ndarray, seq_pairs: Sequence[tuple[str, str] | None], tag_hist: dict[str, int]) -> PairBundle:
    labeled_cols = np.asarray([i for i, p in enumerate(seq_pairs) if p is not None], dtype=np.int64)
    if labeled_cols.size == 0:
        raise RuntimeError("No sequence columns contained exactly two recognized epitope motif hits")
    labeled_pairs = [seq_pairs[int(i)] for i in labeled_cols]
    unique_pairs = sorted({p for p in labeled_pairs if p is not None}, key=pair_sort_key)
    pair_names = [format_pair(p) for p in unique_pairs]
    pair_to_idx = {p: i for i, p in enumerate(unique_pairs)}
    column_pair_idx = np.asarray([pair_to_idx[p] for p in labeled_pairs], dtype=np.int16)

    if sparse.issparse(X):
        X_labeled = X[:, labeled_cols].tocoo()
        node_idx = X_labeled.row.astype(np.int64, copy=False)
        pair_idx = column_pair_idx[X_labeled.col].astype(np.int16, copy=False)
        cols_with_positive = int(np.unique(X_labeled.col).size)
    else:
        X_labeled = np.asarray(X[:, labeled_cols])
        rows, cols = np.nonzero(X_labeled)
        node_idx = rows.astype(np.int64, copy=False)
        pair_idx = column_pair_idx[cols].astype(np.int16, copy=False)
        cols_with_positive = int(np.unique(cols).size)
    if node_idx.size == 0:
        raise RuntimeError("Exact-2 sequence columns were found, but none had positive matrix entries")

    n_hubs, n_pairs = X.shape[0], len(unique_pairs)
    hub_insert_count = np.bincount(node_idx, minlength=n_hubs).astype(np.int64)
    key = node_idx * n_pairs + pair_idx.astype(np.int64, copy=False)
    hub_pair_counts = np.bincount(key, minlength=n_hubs * n_pairs).reshape(n_hubs, n_pairs).astype(np.int32)
    return PairBundle(
        node_idx=node_idx,
        pair_idx=pair_idx,
        coords=coords[node_idx],
        pair_names=pair_names,
        pair_tuples=unique_pairs,
        column_pair_idx=column_pair_idx,
        labeled_cols=labeled_cols,
        hub_insert_count=hub_insert_count,
        hub_pair_counts=hub_pair_counts,
        sequence_tagcount_hist=tag_hist,
        n_sequence_columns=len(seq_pairs),
        n_exact2_sequence_columns=int(labeled_cols.size),
        n_exact2_columns_with_positive_entries=cols_with_positive,
    )


def categorical_codes(values: Sequence[object]) -> tuple[np.ndarray, list[str]]:
    labels: list[str] = []
    label_to_code: dict[str, int] = {}
    codes = np.full(len(values), -1, dtype=np.int64)
    for i, value in enumerate(values):
        text = str(value)
        if text.lower() in {"nan", "none", "<na>"}:
            continue
        if text not in label_to_code:
            label_to_code[text] = len(labels)
            labels.append(text)
        codes[i] = label_to_code[text]
    return codes, labels


def build_full_neighbor_index(coords: np.ndarray, k: int, chunk_size: int) -> tuple[np.ndarray, int]:
    n_points = coords.shape[0]
    if n_points == 0:
        return np.empty((0, 0), dtype=np.int32), 0
    k_eff = max(1, min(int(k), n_points))
    nn = np.empty((n_points, k_eff), dtype=np.int32)
    tree = cKDTree(coords)
    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        try:
            _, chunk = tree.query(coords[start:end], k=k_eff, workers=-1)
        except TypeError:
            _, chunk = tree.query(coords[start:end], k=k_eff)
        if k_eff == 1:
            chunk = chunk[:, None]
        nn[start:end] = chunk.astype(np.int32, copy=False)
    return nn, k_eff


def smoothed_relative_score(observed_cd3: float, observed_igd: float, expected_cd3: float, expected_igd: float) -> tuple[float, float, float]:
    pc = 0.5
    cd3_fold = (observed_cd3 + pc) / (expected_cd3 + pc) if np.isfinite(expected_cd3) else math.nan
    igd_fold = (observed_igd + pc) / (expected_igd + pc) if np.isfinite(expected_igd) else math.nan
    score = float(np.log2(cd3_fold / igd_fold)) if cd3_fold > 0 and igd_fold > 0 and np.isfinite(cd3_fold) and np.isfinite(igd_fold) else math.nan
    return float(cd3_fold), float(igd_fold), score


def compute_infomap_relative_enrichment(
    bundle: PairBundle,
    coords: np.ndarray,
    cluster_codes: np.ndarray,
    cluster_labels: Sequence[str],
    *,
    exact_count: int,
    smoothing_k: int,
    min_labeled_hubs: int,
    chunk_size: int,
) -> list[dict[str, object]]:
    """Score each infomap cluster by kNN-smoothed CD3/CD3-vs-IgD/IgD enrichment within exact d."""
    pair_to_idx = {name: i for i, name in enumerate(bundle.pair_names)}
    cd3_idx, igd_idx = pair_to_idx.get("CD3/CD3"), pair_to_idx.get("IgD/IgD")
    missing = [name for name, idx in (("CD3/CD3", cd3_idx), ("IgD/IgD", igd_idx)) if idx is None]
    if missing:
        warnings.warn("Missing target pair(s), so infomap relative map has no scored clusters: " + ", ".join(missing))
        return []
    cd3_idx, igd_idx = int(cd3_idx), int(igd_idx)
    finite = np.isfinite(coords).all(axis=1)
    source_mask = (bundle.hub_insert_count == int(exact_count)) & finite & (cluster_codes >= 0)
    source_idx = np.flatnonzero(source_mask).astype(np.int64)
    if source_idx.size == 0:
        return []

    subcoords = coords[source_idx]
    sub_clusters = cluster_codes[source_idx].astype(np.int64, copy=False)
    sub_counts = bundle.hub_pair_counts[source_idx].astype(float, copy=False)
    slots_per_hub = sub_counts.sum(axis=1)
    total_slots = float(slots_per_hub.sum())
    if total_slots <= 0:
        return []
    global_cd3 = float(sub_counts[:, cd3_idx].sum()) / total_slots
    global_igd = float(sub_counts[:, igd_idx].sum()) / total_slots
    nn, k_eff = build_full_neighbor_index(subcoords, smoothing_k, chunk_size)

    neighbor_cd3 = sub_counts[nn, cd3_idx].sum(axis=1)
    neighbor_igd = sub_counts[nn, igd_idx].sum(axis=1)
    neighbor_slots = slots_per_hub[nn].sum(axis=1)
    records: list[dict[str, object]] = []
    for cluster_code, cluster in enumerate(cluster_labels):
        centers = np.flatnonzero(sub_clusters == cluster_code)
        if centers.size == 0:
            continue
        total_neighbor_slots = float(neighbor_slots[centers].sum())
        if total_neighbor_slots <= 0:
            continue
        observed_cd3 = float(neighbor_cd3[centers].sum())
        observed_igd = float(neighbor_igd[centers].sum())
        expected_cd3 = total_neighbor_slots * global_cd3
        expected_igd = total_neighbor_slots * global_igd
        cd3_fold, igd_fold, score = smoothed_relative_score(observed_cd3, observed_igd, expected_cd3, expected_igd)
        records.append({
            "analysis_level": "infomap_cluster_knn_relative_enrichment",
            "infomap_cluster": str(cluster),
            "exact_hub_insert_count": int(exact_count),
            "smoothing_k_requested": int(smoothing_k),
            "smoothing_k_effective": int(k_eff),
            "n_labeled_hubs_in_cluster_exact_k": int(centers.size),
            "n_labeled_hubs_in_exact_k_stratum": int(source_idx.size),
            "min_labeled_hubs_per_cluster": int(min_labeled_hubs),
            "is_eligible_by_labeled_hub_count": bool(centers.size >= min_labeled_hubs),
            "n_knn_center_hubs_in_cluster": int(centers.size),
            "n_knn_neighbor_hub_visits": int(centers.size * k_eff),
            "n_knn_neighbor_exact2_insert_slots": total_neighbor_slots,
            "cd3_pair": "CD3/CD3",
            "igd_pair": "IgD/IgD",
            "observed_cd3_pair_count_knn": observed_cd3,
            "observed_igd_pair_count_knn": observed_igd,
            "expected_cd3_pair_count_knn": expected_cd3,
            "expected_igd_pair_count_knn": expected_igd,
            "global_cd3_pair_fraction_exact_k": global_cd3,
            "global_igd_pair_fraction_exact_k": global_igd,
            "cd3_fold_enrichment_knn": cd3_fold,
            "igd_fold_enrichment_knn": igd_fold,
            "log2_relative_enrichment_cd3_vs_igd": score,
            "direction": "CD3/CD3-relative" if np.isfinite(score) and score > 0 else "IgD/IgD-relative" if np.isfinite(score) and score < 0 else "neutral_or_undefined",
        })
    return records


def combo_definitions(n_labels: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_labels) for j in range(i, n_labels)]


def labels_and_units_from_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=np.int64)
    n_units, n_labels = counts.shape
    labels = np.repeat(np.tile(np.arange(n_labels, dtype=np.int16), n_units), counts.ravel()).astype(np.int16, copy=False)
    units = np.repeat(np.repeat(np.arange(n_units, dtype=np.int64), n_labels), counts.ravel()).astype(np.int64, copy=False)
    return labels, units


def counts_by_unit_from_labels(labels: np.ndarray, unit_ids: np.ndarray, n_units: int, n_labels: int) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((n_units, n_labels), dtype=np.int32)
    key = unit_ids.astype(np.int64, copy=False) * n_labels + labels.astype(np.int64, copy=False)
    return np.bincount(key, minlength=n_units * n_labels).reshape(n_units, n_labels).astype(np.int32, copy=False)


def event_counts_from_unit_counts(unit_counts: np.ndarray, combos: Sequence[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, int]:
    counts = np.asarray(unit_counts, dtype=np.int64)
    row_sums = counts.sum(axis=1)
    possible = int(np.sum(row_sums * (row_sums - 1) // 2))
    events = np.zeros(len(combos), dtype=np.int64)
    unit_hits = np.zeros(len(combos), dtype=np.int64)
    for ci, (a, b) in enumerate(combos):
        ca = counts[:, a]
        if a == b:
            per_unit = ca * (ca - 1) // 2
            hit = ca >= 2
        else:
            cb = counts[:, b]
            per_unit = ca * cb
            hit = (ca > 0) & (cb > 0)
        events[ci] = int(per_unit.sum())
        unit_hits[ci] = int(np.count_nonzero(hit))
    return events, unit_hits, possible


def event_counts_from_knn_hub_counts(hub_counts: np.ndarray, nn: np.ndarray, combos: Sequence[tuple[int, int]], chunk_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    events = np.zeros(len(combos), dtype=np.int64)
    unit_hits = np.zeros(len(combos), dtype=np.int64)
    possible = 0
    n_labels = hub_counts.shape[1]
    for start in range(0, nn.shape[0], chunk_size):
        end = min(start + chunk_size, nn.shape[0])
        chunk_counts = hub_counts[nn[start:end].reshape(-1)].reshape(end - start, nn.shape[1], n_labels).sum(axis=1)
        ev, uh, poss = event_counts_from_unit_counts(chunk_counts, combos)
        events += ev
        unit_hits += uh
        possible += poss
    return events, unit_hits, int(possible)


def empirical_tail_p(null_values: np.ndarray, observed: float, tail: str) -> float:
    if null_values.size == 0 or not np.isfinite(observed):
        return math.nan
    if tail == "upper":
        return float((1 + np.count_nonzero(null_values >= observed)) / (null_values.size + 1))
    if tail == "lower":
        return float((1 + np.count_nonzero(null_values <= observed)) / (null_values.size + 1))
    raise ValueError(tail)


def mean_sd_z(null_values: np.ndarray, observed: float) -> tuple[float, float, float]:
    if null_values.size == 0 or not np.isfinite(observed):
        return math.nan, math.nan, math.nan
    mean = float(np.mean(null_values))
    sd = float(np.std(null_values, ddof=1)) if null_values.size > 1 else 0.0
    return mean, sd, float((observed - mean) / sd) if sd > 0 else math.nan


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    pv = p[valid]
    order = np.argsort(pv)
    adjusted = pv[order] * len(pv) / np.arange(1, len(pv) + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(pv)
    restored[order] = adjusted
    q[np.flatnonzero(valid)] = restored
    return q


def effective_permutation_count(n_slots: int, requested: int, budget: int, context: str) -> int:
    if requested < 1:
        raise ValueError("--n-permutations must be at least 1")
    if budget <= 0 or n_slots * requested <= budget:
        return int(requested)
    reduced = max(1, int(budget // max(1, n_slots)))
    warnings.warn(f"{context}: reducing permutations {requested} -> {reduced} for slot budget")
    return min(int(requested), reduced)


def run_knn_neighborhood_records(
    bundle: PairBundle,
    coords: np.ndarray,
    *,
    exact_count: int,
    ks: Sequence[int],
    n_permutations: int,
    rng: np.random.Generator,
    min_units: int,
    chunk_size: int,
    null_slot_budget: int,
) -> list[dict[str, object]]:
    """Permutation z-score table for kNN neighborhoods inside one exact-d stratum."""
    finite = np.isfinite(coords).all(axis=1)
    source_mask = (bundle.hub_insert_count == int(exact_count)) & finite
    hub_counts = bundle.hub_pair_counts[source_mask].astype(np.int32, copy=False)
    subcoords = coords[source_mask]
    if hub_counts.shape[0] < min_units:
        return []
    combos = combo_definitions(len(bundle.pair_names))
    slot_labels, hub_slot_ids = labels_and_units_from_counts(hub_counts)
    eff = effective_permutation_count(int(slot_labels.size), n_permutations, null_slot_budget, "kNN neighborhood null")
    records: list[dict[str, object]] = []

    for k in ks:
        nn, k_eff = build_full_neighbor_index(subcoords, int(k), chunk_size)
        observed_events, observed_units, possible = event_counts_from_knn_hub_counts(hub_counts, nn, combos, chunk_size)
        if possible <= 0:
            continue
        null_events = np.zeros((eff, len(combos)), dtype=np.int64)
        null_units = np.zeros((eff, len(combos)), dtype=np.int64)
        work = slot_labels.copy()
        for pi in range(eff):
            rng.shuffle(work)
            perm_counts = counts_by_unit_from_labels(work, hub_slot_ids, hub_counts.shape[0], len(bundle.pair_names))
            ev, uh, _ = event_counts_from_knn_hub_counts(perm_counts, nn, combos, chunk_size)
            null_events[pi] = ev
            null_units[pi] = uh
        for ci, (a, b) in enumerate(combos):
            obs_event_fraction = float(observed_events[ci] / possible)
            null_event_fraction = null_events[:, ci].astype(float) / float(possible)
            event_mean, event_sd, event_z = mean_sd_z(null_event_fraction, obs_event_fraction)
            obs_unit_rate = float(observed_units[ci] / nn.shape[0])
            null_unit_rate = null_units[:, ci].astype(float) / float(nn.shape[0])
            unit_mean, unit_sd, unit_z = mean_sd_z(null_unit_rate, obs_unit_rate)
            records.append({
                "analysis_level": "knn_neighborhood",
                "exact_hub_insert_count": int(exact_count),
                "neighborhood_k": int(k_eff),
                "n_source_hubs": int(hub_counts.shape[0]),
                "n_units": int(nn.shape[0]),
                "n_shuffle_slots": int(slot_labels.size),
                "n_possible_insert_pairs": int(possible),
                "label_a": bundle.pair_names[a],
                "label_b": bundle.pair_names[b],
                "label_combo": f"{bundle.pair_names[a]} + {bundle.pair_names[b]}",
                "is_same_label_combo": bool(a == b),
                "observed_event_count": int(observed_events[ci]),
                "observed_event_fraction": obs_event_fraction,
                "null_event_fraction_mean": event_mean,
                "null_event_fraction_sd": event_sd,
                "z_score_event_fraction": event_z,
                "p_event_enriched": empirical_tail_p(null_event_fraction, obs_event_fraction, "upper"),
                "p_event_depleted": empirical_tail_p(null_event_fraction, obs_event_fraction, "lower"),
                "observed_units_with_combo": int(observed_units[ci]),
                "observed_unit_rate": obs_unit_rate,
                "null_unit_rate_mean": unit_mean,
                "null_unit_rate_sd": unit_sd,
                "z_score_unit_rate": unit_z,
                "p_unit_enriched": empirical_tail_p(null_unit_rate, obs_unit_rate, "upper"),
                "p_unit_depleted": empirical_tail_p(null_unit_rate, obs_unit_rate, "lower"),
                "n_permutations": int(eff),
            })
    add_knn_fdr(records)
    return records


def add_knn_fdr(records: list[dict[str, object]]) -> None:
    if not records:
        return
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[(int(rec["exact_hub_insert_count"]), int(rec["neighborhood_k"]))].append(i)
    for idxs in groups.values():
        for p_col, q_col in (("p_event_enriched", "q_event_enriched"), ("p_event_depleted", "q_event_depleted"), ("p_unit_enriched", "q_unit_enriched"), ("p_unit_depleted", "q_unit_depleted")):
            q = bh_fdr([float(records[i].get(p_col, math.nan)) for i in idxs])
            for i, qi in zip(idxs, q):
                records[i][q_col] = float(qi) if np.isfinite(qi) else math.nan


def equal_quantile_limits(coords: np.ndarray, q_low: float = 0.001, q_high: float = 0.999):
    finite = np.isfinite(coords).all(axis=1)
    data = coords[finite]
    if data.shape[0] == 0:
        return (-1, 1), (-1, 1)
    xq = np.quantile(data[:, 0], [q_low, q_high])
    yq = np.quantile(data[:, 1], [q_low, q_high])
    mid = (0.5 * (xq[0] + xq[1]), 0.5 * (yq[0] + yq[1]))
    half = 0.5 * max(xq[1] - xq[0], yq[1] - yq[0])
    half = float(half) if np.isfinite(half) and half > 0 else 1.0
    return (mid[0] - half, mid[0] + half), (mid[1] - half, mid[1] + half)


def style_map_axes(ax, xlim, ylim, title: str) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(True, color="#a8a8a8", alpha=0.36, linewidth=0.65)
    ax.set_xlabel("coordinate 1", color="white")
    ax.set_ylabel("coordinate 2", color="white")
    ax.set_title(title, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")


def plot_infomap_relative_map(
    coords: np.ndarray,
    cluster_codes: np.ndarray,
    cluster_labels: Sequence[str],
    records: Sequence[dict[str, object]],
    *,
    exact_count: int,
    output_png: Path,
    log2_cap: float,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    finite = np.isfinite(coords).all(axis=1)
    xlim, ylim = equal_quantile_limits(coords)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("black")
    ax.scatter(coords[finite, 0], coords[finite, 1], s=BACKGROUND_DOT_SIZE, color="white", alpha=0.035, edgecolors="none", rasterized=True)

    selected = [r for r in records if int(r.get("exact_hub_insert_count", -1)) == int(exact_count) and np.isfinite(float(r.get("log2_relative_enrichment_cd3_vs_igd", math.nan)))]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("igd_to_cd3_relative", ["#b2182b", "#ffffbf", "#1a9850"])
    cap = float(log2_cap) if np.isfinite(log2_cap) and log2_cap > 0 else 3.0
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-cap, vcenter=0.0, vmax=cap)
    label_to_code = {str(label): i for i, label in enumerate(cluster_labels)}

    if selected:
        for rec in selected:
            code = label_to_code.get(str(rec["infomap_cluster"]))
            if code is None:
                continue
            mask = finite & (cluster_codes == code)
            if not mask.any():
                continue
            score = float(rec["log2_relative_enrichment_cd3_vs_igd"])
            ax.scatter(coords[mask, 0], coords[mask, 1], s=INFOMAP_RELATIVE_DOT_SIZE, color=cmap(norm(float(np.clip(score, -cap, cap)))), alpha=0.72, edgecolors="none", rasterized=True)
    else:
        ax.text(0.5, 0.5, f"No finite CD3/CD3-vs-IgD/IgD cluster scores at exact d={exact_count}", ha="center", va="center", color="white", transform=ax.transAxes)

    style_map_axes(ax, xlim, ylim, f"Infomap clusters colored by kNN-smoothed relative enrichment, exact d={exact_count}\nred = IgD/IgD-relative; green = CD3/CD3-relative")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log2 relative enrichment: CD3/CD3 vs IgD/IgD", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    cbar.outline.set_edgecolor("white")
    ax.scatter([], [], s=78, facecolors="#1a9850", edgecolors="none", label="CD3/CD3-relative")
    ax.scatter([], [], s=78, facecolors="#ffffbf", edgecolors="none", label="near neutral")
    ax.scatter([], [], s=78, facecolors="#b2182b", edgecolors="none", label="IgD/IgD-relative")
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=3, frameon=True, facecolor="black", edgecolor="white")
    for text in legend.get_texts():
        text.set_color("white")
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def plot_knn_zscore_summary(records: Sequence[dict[str, object]], output_png: Path, exact_count: int) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    sub = [r for r in records if int(r.get("exact_hub_insert_count", -1)) == int(exact_count) and np.isfinite(float(r.get("z_score_event_fraction", math.nan)))]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    if sub:
        preferred = {
            tuple(sorted(("CD3/CD3", "IgD/IgD"))),
            tuple(sorted(("CD3/CD3", "CD9/CD9"))),
            tuple(sorted(("IgD/IgD", "CD9/CD9"))),
            tuple(sorted(("CD24/CD24", "IgD/IgD"))),
        }
        filtered = [r for r in sub if tuple(sorted((str(r["label_a"]), str(r["label_b"])))) in preferred]
        if not filtered:
            filtered = sorted(sub, key=lambda r: abs(float(r.get("z_score_event_fraction", 0.0))), reverse=True)[:40]
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for rec in filtered:
            grouped[str(rec["label_combo"])].append(rec)
        for combo, series in grouped.items():
            series = sorted(series, key=lambda r: int(r["neighborhood_k"]))
            ax.plot([int(r["neighborhood_k"]) for r in series], [float(r["z_score_event_fraction"]) for r in series], marker="o", linewidth=1.8, label=combo)
        if grouped:
            legend = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, facecolor="white", edgecolor="black")
            fig.subplots_adjust(right=0.76)
    else:
        ax.text(0.5, 0.5, f"No finite kNN z-scores at exact d={exact_count}", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0.0, linestyle="--", linewidth=1, color="#555555")
    ax.axhline(1.96, linestyle=":", linewidth=1, color="#555555")
    ax.axhline(-1.96, linestyle=":", linewidth=1, color="#555555")
    ax.set_axisbelow(True)
    ax.grid(True, color="#b0b0b0", alpha=0.35, linewidth=0.7)
    ax.set_xlabel("k nearest same-degree hubs")
    ax.set_ylabel("Permutation z-score, event fraction")
    ax.set_title(f"kNN neighborhood co-localization, exact hub insert count d={exact_count}")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for rec in records:
        for key in rec:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_pair_inventory(bundle: PairBundle, outdir: Path) -> None:
    col_counts = np.bincount(bundle.column_pair_idx.astype(np.int64), minlength=len(bundle.pair_names))
    ann_counts = np.bincount(bundle.pair_idx.astype(np.int64), minlength=len(bundle.pair_names))
    records = []
    for i, name in enumerate(bundle.pair_names):
        a, b = bundle.pair_tuples[i]
        records.append({
            "pair_label": name,
            "marker_a": a,
            "marker_b": b,
            "is_homo_epitope_pair": bool(a == b),
            "n_exact2_sequence_columns": int(col_counts[i]),
            "n_positive_hub_insert_annotations": int(ann_counts[i]),
        })
    write_csv(outdir / "pair_label_inventory.csv", records)


def write_hub_count_summary(bundle: PairBundle, outdir: Path) -> None:
    records = []
    for d in sorted(int(x) for x in np.unique(bundle.hub_insert_count) if int(x) > 0):
        mask = bundle.hub_insert_count == d
        rec = {"exact_hub_insert_count": d, "n_hubs": int(mask.sum()), "n_qualifying_insert_annotations": int(bundle.hub_insert_count[mask].sum())}
        counts = bundle.hub_pair_counts[mask].sum(axis=0)
        for i, name in enumerate(bundle.pair_names):
            rec[f"pair_count__{safe_token(name)}"] = int(counts[i])
        records.append(rec)
    write_csv(outdir / "hub_exact_insert_count_summary.csv", records)


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.h5ad!r} …", flush=True)
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError("Install anndata to read .h5ad files: pip install anndata") from exc
    adata = ad.read_h5ad(args.h5ad)
    if args.infomap_cluster_col not in adata.obs.columns:
        raise KeyError(f"--infomap-cluster-col {args.infomap_cluster_col!r} was not found in adata.obs")

    X_raw, matrix_source = resolve_matrix(adata, args.layer)
    seq_var = choose_var_column(adata.var, args.sequence_var, SEQUENCE_VAR_CANDIDATES)
    coords, coord_source = resolve_coords(adata, args.x_col, args.y_col)
    headers = Headers(matrix_source, seq_var, coord_source, args.infomap_cluster_col)

    sequences = np.asarray(adata.var_names, dtype=object) if seq_var == "__var_names__" else adata.var[seq_var].to_numpy(dtype=object, copy=False)
    seq_pairs, tag_hist = label_sequence_columns(sequences)
    X = binarize_feature_matrix(X_raw)
    bundle = build_pair_bundle(X, coords, seq_pairs, tag_hist)
    exact_count = int(args.exact_hub_insert_count)
    if not np.any(bundle.hub_insert_count == exact_count):
        raise RuntimeError(f"No hubs have exact qualifying insert count d={exact_count}")

    cluster_codes, cluster_labels = categorical_codes(adata.obs[args.infomap_cluster_col].to_numpy(dtype=object, copy=False))
    print(f"Scoring infomap relative CD3/CD3-vs-IgD/IgD map at d={exact_count}, kNN={args.infomap_celltype_knn_k} …", flush=True)
    relative_records = compute_infomap_relative_enrichment(
        bundle, coords, cluster_codes, cluster_labels,
        exact_count=exact_count,
        smoothing_k=args.infomap_celltype_knn_k,
        min_labeled_hubs=args.min_labeled_hubs_per_infomap_cluster,
        chunk_size=args.chunk_size,
    )
    relative_csv = outdir / f"infomap_cluster_knn_relative_cd3_vs_igd_enrichment_k{safe_token(exact_count)}_knn{safe_token(args.infomap_celltype_knn_k)}.csv"
    write_csv(relative_csv, relative_records)
    if not args.no_plots:
        plot_infomap_relative_map(
            coords,
            cluster_codes,
            cluster_labels,
            relative_records,
            exact_count=exact_count,
            output_png=outdir / f"infomap_cluster_knn_relative_cd3_vs_igd_map_k{safe_token(exact_count)}_knn{safe_token(args.infomap_celltype_knn_k)}.png",
            log2_cap=args.celltype_log2_relative_enrichment_cap,
        )

    ks = parse_int_list(args.knn_ks)
    print(f"Running kNN neighborhood z-score summary at d={exact_count}; k values={ks} …", flush=True)
    knn_records = run_knn_neighborhood_records(
        bundle,
        coords,
        exact_count=exact_count,
        ks=ks,
        n_permutations=args.n_permutations,
        rng=rng,
        min_units=args.min_units_per_stratum,
        chunk_size=args.chunk_size,
        null_slot_budget=args.null_slot_budget,
    )
    knn_csv = outdir / f"knn_neighborhood_zscore_summary_source_d{safe_token(exact_count)}.csv"
    write_csv(knn_csv, knn_records)
    if not args.no_plots:
        plot_knn_zscore_summary(knn_records, outdir / f"knn_neighborhood_zscore_summary_d{safe_token(exact_count)}.png", exact_count)

    write_pair_inventory(bundle, outdir)
    write_hub_count_summary(bundle, outdir)
    metadata = {
        "h5ad": str(args.h5ad),
        "n_hubs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "headers": {
            "matrix_source": headers.matrix_source,
            "sequence_var": headers.sequence_var,
            "coord_source": headers.coord_source,
            "infomap_cluster_col": headers.infomap_cluster_col,
        },
        "selected_analysis": {
            "exact_hub_insert_count": exact_count,
            "infomap_relative_knn_k": int(args.infomap_celltype_knn_k),
            "knn_zscore_k_values": ks,
            "n_permutations": int(args.n_permutations),
            "seed": int(args.seed),
        },
        "sequence_tagcount_distribution": bundle.sequence_tagcount_hist,
        "n_exact2_sequence_columns": int(bundle.n_exact2_sequence_columns),
        "n_exact2_columns_with_positive_entries": int(bundle.n_exact2_columns_with_positive_entries),
        "n_positive_exact2_hub_insert_annotations": int(bundle.node_idx.size),
        "n_hubs_with_selected_exact_count": int(np.count_nonzero(bundle.hub_insert_count == exact_count)),
        "pair_labels": bundle.pair_names,
        "target_motifs": TARGET_MOTIFS,
        "outputs": {
            "infomap_relative_csv": str(relative_csv),
            "knn_zscore_summary_csv": str(knn_csv),
        },
    }
    write_json(outdir / "parse_summary.json", metadata)
    print(f"Done. Focused outputs written to {outdir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Focused infomap/kNN exact-2 pair outputs only.")
    p.add_argument("--h5ad", default="data_dir/uei_grp0/final.h5ad")
    p.add_argument("--outdir", default="out_dir")
    p.add_argument("--layer", default=DEFAULT_LAYER)
    p.add_argument("--sequence-var", default=DEFAULT_SEQUENCE_VAR)
    p.add_argument("--x-col", default=DEFAULT_X_COL)
    p.add_argument("--y-col", default=DEFAULT_Y_COL)
    p.add_argument("--infomap-cluster-col", default=DEFAULT_INFOMAP_CLUSTER_COL)
    p.add_argument("--exact-hub-insert-count", type=int, default=DEFAULT_EXACT_HUB_INSERT_COUNT, help="Default d=1, producing *_k1_* and *_d1 outputs.")
    p.add_argument("--infomap-celltype-knn-k", type=int, default=DEFAULT_INFOMAP_CELLTYPE_KNN_K)
    p.add_argument("--celltype-log2-relative-enrichment-cap", type=float, default=DEFAULT_LOG2_RELATIVE_CAP)
    p.add_argument("--knn-ks", default=DEFAULT_KNN_KS)
    p.add_argument("--n-permutations", type=int, default=DEFAULT_N_PERMUTATIONS)
    p.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--min-units-per-stratum", type=int, default=DEFAULT_MIN_UNITS)
    p.add_argument("--min-labeled-hubs-per-infomap-cluster", type=int, default=10)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--null-slot-budget", type=int, default=DEFAULT_NULL_SLOT_BUDGET)
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.exact_hub_insert_count < 1:
        parser.error("--exact-hub-insert-count must be at least 1")
    if args.infomap_celltype_knn_k < 1:
        parser.error("--infomap-celltype-knn-k must be at least 1")
    if args.n_permutations < 1:
        parser.error("--n-permutations must be at least 1")
    if args.min_units_per_stratum < 1:
        parser.error("--min-units-per-stratum must be at least 1")
    if args.celltype_log2_relative_enrichment_cap <= 0:
        parser.error("--celltype-log2-relative-enrichment-cap must be positive")
    try:
        run(args)
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Focused cDNA exact-k co-localization pipeline.

This cleaned version keeps only the raw-data path needed to produce:

  conditional_lift_heatmaps_by_exact_hub_insert_count/
      homo_epitope_pair_to_homo_epitope_pair_log2_enrichment_t1_k2.png

  epitope_colocalization_grids_by_exact_hub_insert_count/
      epitope_colocalization_between_insert_log2_enrichment_t1_k2.png

plus the CSV/JSON tables that directly back those plots.  Inserts are analytic
only when their sequence contains exactly two recognized epitope motif hits.
For the selected trust threshold and exact hub insert count, the null shuffles
exact-two insert-pair labels across the fixed insert slots in that same exact-k
hub stratum.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
LEADING_KMER_WINDOW = 12
DEFAULT_NULL_INSERT_BUDGET = 400_000_000

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans"],
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


@dataclass(frozen=True)
class Universe:
    markers: list[str]
    marker_to_idx: dict[str, int]
    pair_names: list[str]
    pair_marker_idx: np.ndarray
    pair_to_idx: dict[tuple[int, int], int]
    homo_pair_idx: np.ndarray


@dataclass(frozen=True)
class ParsedInserts:
    insert_hub: np.ndarray
    insert_trust: np.ndarray
    insert_pair_idx: np.ndarray
    n_hubs: int
    n_inserts_total: int
    n_inserts_with_any_tag: int
    tagcount_hist: dict[int, int]
    malformed_trust_values: int
    length_mismatch_rows: int
    empty_or_missing_rows: int
    max_trust: int
    unrecognized_leading_kmers: Counter


@dataclass(frozen=True)
class ExactStratum:
    threshold: int
    exact_k: int
    hub_ids: np.ndarray
    slot_hub_inv: np.ndarray
    slot_pair_idx: np.ndarray


def canonical_target(name: str) -> str:
    return ALIAS_TO_CANONICAL.get(name, name)


def build_universe() -> Universe:
    markers: list[str] = []
    for name in (*PREFERRED_MARKER_ORDER, *TARGET_MOTIFS):
        canon = canonical_target(name)
        if canon not in markers:
            markers.append(canon)
    marker_to_idx = {m: i for i, m in enumerate(markers)}
    pair_names: list[str] = []
    pair_marker_idx: list[tuple[int, int]] = []
    pair_to_idx: dict[tuple[int, int], int] = {}
    for i, a in enumerate(markers):
        for j, b in enumerate(markers[i:], start=i):
            pair_to_idx[(i, j)] = len(pair_names)
            pair_names.append(f"{a}/{b}")
            pair_marker_idx.append((i, j))
    pair_marker_idx_arr = np.asarray(pair_marker_idx, dtype=np.int16)
    homo = np.flatnonzero(pair_marker_idx_arr[:, 0] == pair_marker_idx_arr[:, 1]).astype(np.int64)
    return Universe(markers, marker_to_idx, pair_names, pair_marker_idx_arr, pair_to_idx, homo)


def count_motif(seq: str, motif: str) -> int:
    n = start = 0
    while True:
        pos = seq.find(motif, start)
        if pos == -1:
            return n
        n += 1
        start = pos + len(motif)


def tags_in_sequence(seq: object, universe: Universe) -> list[int]:
    if not isinstance(seq, str) or not seq:
        return []
    text = seq.upper()
    hits: list[int] = []
    for target, motifs in TARGET_MOTIFS.items():
        idx = universe.marker_to_idx[canonical_target(target)]
        for motif in motifs:
            hits.extend([idx] * count_motif(text, motif))
    return hits


def split_semicolon_field(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value)
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return []
    return text.split(";")


def parse_trust_values(value: object) -> tuple[list[int], int]:
    values, malformed = [], 0
    for part in split_semicolon_field(value):
        try:
            values.append(int(part))
        except (TypeError, ValueError):
            values.append(0)
            malformed += 1
    return values, malformed


def validate_input(ann) -> None:
    missing = [c for c in ("seq_str", "seq_subreads_str") if c not in ann.obs.columns]
    if missing:
        raise KeyError("Input AnnData is missing obs column(s): " + ", ".join(missing))


def parse_exact2_inserts(ann, universe: Universe, *, strict_lengths: bool, report_every: int) -> ParsedInserts:
    """Parse semicolon-delimited cDNA inserts and retain exactly-two-hit inserts only."""
    validate_input(ann)
    seq_rows = ann.obs["seq_str"].to_numpy(dtype=object, copy=False)
    trust_rows = ann.obs["seq_subreads_str"].to_numpy(dtype=object, copy=False)

    hubs: list[int] = []
    trusts: list[int] = []
    pairs: list[int] = []
    tag_hist: Counter = Counter()
    unrecognized: Counter = Counter()
    total = tagged = malformed = mismatched = empty = max_trust = 0

    for hub_i, (seq_entry, trust_entry) in enumerate(zip(seq_rows, trust_rows)):
        seqs = split_semicolon_field(seq_entry)
        trust_values, bad = parse_trust_values(trust_entry)
        malformed += bad
        if not seqs and not trust_values:
            empty += 1
            continue
        if len(seqs) != len(trust_values):
            mismatched += 1
            message = f"row {hub_i}: {len(seqs)} seq_str values but {len(trust_values)} seq_subreads_str values"
            if strict_lengths:
                raise ValueError(message)
            if mismatched <= 5:
                warnings.warn(message + "; truncating to the shorter length")
        n = min(len(seqs), len(trust_values))
        total += n
        for seq, trust in zip(seqs[:n], trust_values[:n]):
            max_trust = max(max_trust, int(trust))
            hits = tags_in_sequence(seq, universe)
            tag_hist[min(len(hits), 3)] += 1
            if hits:
                tagged += 1
            elif isinstance(seq, str) and seq:
                unrecognized[seq[:LEADING_KMER_WINDOW]] += 1
            if len(hits) != 2:
                continue
            a, b = sorted((int(hits[0]), int(hits[1])))
            hubs.append(hub_i)
            trusts.append(int(trust))
            pairs.append(universe.pair_to_idx[(a, b)])
        if report_every > 0 and (hub_i + 1) % report_every == 0:
            print(f"  parsed {hub_i + 1:,}/{ann.n_obs:,} hubs; exact-2 inserts: {len(hubs):,}", flush=True)

    return ParsedInserts(
        np.asarray(hubs, dtype=np.int64),
        np.asarray(trusts, dtype=np.int64),
        np.asarray(pairs, dtype=np.int16),
        int(ann.n_obs), int(total), int(tagged),
        {int(k): int(v) for k, v in sorted(tag_hist.items())},
        int(malformed), int(mismatched), int(empty), int(max_trust), unrecognized,
    )


def parse_thresholds(spec: str) -> list[int]:
    out = []
    for chunk in str(spec).replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            value = int(chunk)
            if value < 1:
                raise ValueError("thresholds must be positive integers")
            out.append(value)
    if not out:
        raise ValueError("no thresholds were parsed")
    return sorted(set(out))


def make_exact_stratum(parsed: ParsedInserts, threshold: int, exact_k: int) -> ExactStratum:
    accepted = parsed.insert_trust >= int(threshold)
    hub_counts = np.bincount(parsed.insert_hub[accepted], minlength=parsed.n_hubs).astype(np.int64)
    hub_ids = np.flatnonzero(hub_counts == int(exact_k)).astype(np.int64)
    hub_mask = np.zeros(parsed.n_hubs, dtype=bool)
    hub_mask[hub_ids] = True
    slot_mask = accepted & hub_mask[parsed.insert_hub]
    hub_to_inv = np.full(parsed.n_hubs, -1, dtype=np.int64)
    hub_to_inv[hub_ids] = np.arange(hub_ids.size, dtype=np.int64)
    slot_hub_inv = hub_to_inv[parsed.insert_hub[slot_mask]]
    slot_pair_idx = parsed.insert_pair_idx[slot_mask].astype(np.int16, copy=False)
    expected = int(hub_ids.size * exact_k)
    if slot_pair_idx.size != expected:
        warnings.warn(f"stratum integrity check: expected {expected:,} slots, found {slot_pair_idx.size:,}")
    return ExactStratum(int(threshold), int(exact_k), hub_ids, slot_hub_inv, slot_pair_idx)


def pair_counts_from_slots(stratum: ExactStratum, n_pairs: int) -> np.ndarray:
    n_hubs = int(stratum.hub_ids.size)
    if n_hubs == 0 or stratum.slot_pair_idx.size == 0:
        return np.zeros((n_hubs, n_pairs), dtype=np.int32)
    key = stratum.slot_hub_inv * n_pairs + stratum.slot_pair_idx.astype(np.int64, copy=False)
    return np.bincount(key, minlength=n_hubs * n_pairs).reshape(n_hubs, n_pairs).astype(np.int32)


def slot_epitope_presence_counts(stratum: ExactStratum, universe: Universe) -> np.ndarray:
    n_hubs, n_markers = int(stratum.hub_ids.size), len(universe.markers)
    out = np.zeros((n_hubs, n_markers), dtype=np.int32)
    if n_hubs == 0 or stratum.slot_pair_idx.size == 0:
        return out
    pm = universe.pair_marker_idx[stratum.slot_pair_idx.astype(np.int64, copy=False)]
    slot_has = np.zeros((stratum.slot_pair_idx.size, n_markers), dtype=bool)
    rows = np.arange(stratum.slot_pair_idx.size)
    slot_has[rows, pm[:, 0]] = True
    slot_has[rows, pm[:, 1]] = True
    for marker_i in range(n_markers):
        keep = slot_has[:, marker_i]
        if keep.any():
            out[:, marker_i] = np.bincount(stratum.slot_hub_inv[keep], minlength=n_hubs)
    return out


def between_insert_epitope_counts(stratum: ExactStratum, universe: Universe) -> np.ndarray:
    """Hub counts requiring the two compared epitopes to be assignable to different inserts."""
    pair_counts = pair_counts_from_slots(stratum, len(universe.pair_names))
    slot_counts = slot_epitope_presence_counts(stratum, universe).astype(np.int64, copy=False)
    out = np.zeros(len(universe.pair_names), dtype=np.int64)
    for pi, (a, b) in enumerate(universe.pair_marker_idx):
        a, b = int(a), int(b)
        if a == b:
            out[pi] = int(np.count_nonzero(slot_counts[:, a] >= 2))
        else:
            assignments = slot_counts[:, a] * slot_counts[:, b] - pair_counts[:, pi]
            out[pi] = int(np.count_nonzero(assignments > 0))
    return out


def homo_epitope_pair_counts(stratum: ExactStratum, universe: Universe) -> np.ndarray:
    """Hub co-localization counts among homo insert labels such as CD3/CD3 and IgD/IgD."""
    pair_counts = pair_counts_from_slots(stratum, len(universe.pair_names))
    homo_counts = pair_counts[:, universe.homo_pair_idx].astype(np.int64, copy=False)
    n = len(universe.homo_pair_idx)
    out = np.zeros(n * (n + 1) // 2, dtype=np.int64)
    pos = 0
    for i in range(n):
        for j in range(i, n):
            if i == j:
                out[pos] = int(np.count_nonzero(homo_counts[:, i] >= 2))
            else:
                out[pos] = int(np.count_nonzero((homo_counts[:, i] > 0) & (homo_counts[:, j] > 0)))
            pos += 1
    return out


def effective_permutation_count(n_slots: int, requested: int, budget: int, context: str) -> int:
    if requested < 1:
        raise ValueError("--n-permutations must be at least 1")
    if budget <= 0 or n_slots * requested <= budget:
        return int(requested)
    reduced = max(1, int(budget // max(1, n_slots)))
    warnings.warn(f"{context}: reducing permutations {requested} -> {reduced} for insert-slot budget")
    return min(int(requested), reduced)


def shared_slot_shuffle_null(
    stratum: ExactStratum,
    universe: Universe,
    n_permutations: int,
    rng: np.random.Generator,
    insert_budget: int,
) -> dict[str, np.ndarray | int]:
    n_slots = int(stratum.slot_pair_idx.size)
    n_between = len(universe.pair_names)
    n_homo = len(universe.homo_pair_idx) * (len(universe.homo_pair_idx) + 1) // 2
    observed_between = between_insert_epitope_counts(stratum, universe)
    observed_homo = homo_epitope_pair_counts(stratum, universe)
    if stratum.hub_ids.size == 0 or n_slots == 0:
        return {
            "between_observed": observed_between,
            "between_null": np.zeros((0, n_between), dtype=np.int64),
            "homo_observed": observed_homo,
            "homo_null": np.zeros((0, n_homo), dtype=np.int64),
            "n_null_permutations": 0,
        }

    eff = effective_permutation_count(n_slots, n_permutations, insert_budget, "shared exact-k null")
    between_null = np.empty((eff, n_between), dtype=np.int64)
    homo_null = np.empty((eff, n_homo), dtype=np.int64)
    labels = stratum.slot_pair_idx.copy()
    for i in range(eff):
        rng.shuffle(labels)
        perm = ExactStratum(stratum.threshold, stratum.exact_k, stratum.hub_ids, stratum.slot_hub_inv, labels.copy())
        between_null[i] = between_insert_epitope_counts(perm, universe)
        homo_null[i] = homo_epitope_pair_counts(perm, universe)
    return {
        "between_observed": observed_between,
        "between_null": between_null,
        "homo_observed": observed_homo,
        "homo_null": homo_null,
        "n_null_permutations": int(eff),
    }


def empirical_tail_p(null_values: np.ndarray, observed: float, tail: str) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size == 0 or not np.isfinite(observed):
        return math.nan
    if tail == "upper":
        return float((1 + np.count_nonzero(null_values >= observed)) / (null_values.size + 1))
    if tail == "lower":
        return float((1 + np.count_nonzero(null_values <= observed)) / (null_values.size + 1))
    raise ValueError(tail)


def rank_percentile(null_values: np.ndarray, observed: float) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size == 0 or not np.isfinite(observed):
        return math.nan
    return float((1 + np.count_nonzero(null_values < observed)) / (null_values.size + 1))


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


def add_fdr(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for p, q in (("p_enrich", "q_enrich"), ("p_deplete", "q_deplete"), ("p_two_sided", "q_two_sided")):
        out[q] = bh_fdr(out[p].to_numpy(dtype=float))
    return out


def null_summary(null_matrix: np.ndarray, observed: float, cell_i: int) -> dict[str, float]:
    if null_matrix.size == 0:
        return {k: math.nan for k in ("null_median", "null_q025", "null_q975", "lift", "log2_lift", "p_enrich", "p_deplete", "p_two_sided", "rank_percentile")}
    col = null_matrix[:, cell_i].astype(float)
    med = float(np.median(col))
    lift = float(observed / med) if med > 0 else math.nan
    log2_lift = float(np.log2(lift)) if lift > 0 and np.isfinite(lift) else math.nan
    p_enrich = empirical_tail_p(col, observed, "upper")
    p_deplete = empirical_tail_p(col, observed, "lower")
    return {
        "null_median": med,
        "null_q025": float(np.quantile(col, 0.025)),
        "null_q975": float(np.quantile(col, 0.975)),
        "lift": lift,
        "log2_lift": log2_lift,
        "p_enrich": p_enrich,
        "p_deplete": p_deplete,
        "p_two_sided": min(1.0, 2.0 * min(p_enrich, p_deplete)),
        "rank_percentile": rank_percentile(col, observed),
    }


def between_records(stratum: ExactStratum, universe: Universe, observed: np.ndarray, null: np.ndarray, n_perm: int) -> pd.DataFrame:
    records = []
    n_hubs = int(stratum.hub_ids.size)
    for pi, pair_name in enumerate(universe.pair_names):
        a_i, b_i = map(int, universe.pair_marker_idx[pi])
        obs = float(observed[pi])
        s = null_summary(null, obs, pi)
        records.append({
            "threshold": stratum.threshold,
            "exact_hub_insert_count": stratum.exact_k,
            "epitope_pair": pair_name,
            "marker_a": universe.markers[a_i],
            "marker_b": universe.markers[b_i],
            "is_homo_epitope_cell": a_i == b_i,
            "same_insert_excluded": True,
            "n_hubs_in_stratum": n_hubs,
            "n_qualifying_insert_slots_in_stratum": int(stratum.slot_pair_idx.size),
            "hub_cooccurrence_count": int(observed[pi]),
            "hub_cooccurrence_rate": float(observed[pi] / n_hubs) if n_hubs else math.nan,
            "null_median_hub_count": s["null_median"],
            "null_q025_hub_count": s["null_q025"],
            "null_q975_hub_count": s["null_q975"],
            "enrichment_vs_slot_shuffle_null": s["lift"],
            "log2_enrichment_vs_slot_shuffle_null": s["log2_lift"],
            "p_enrich": s["p_enrich"],
            "p_deplete": s["p_deplete"],
            "p_two_sided": s["p_two_sided"],
            "rank_percentile": s["rank_percentile"],
            "n_null_permutations": int(n_perm),
        })
    return add_fdr(pd.DataFrame.from_records(records))


def homo_records(stratum: ExactStratum, universe: Universe, observed: np.ndarray, null: np.ndarray, n_perm: int) -> pd.DataFrame:
    records = []
    n_hubs = int(stratum.hub_ids.size)
    homo_pair_names = [universe.pair_names[int(i)] for i in universe.homo_pair_idx]
    cell_i = 0
    for row_i, row_name in enumerate(homo_pair_names):
        for col_i in range(row_i, len(homo_pair_names)):
            col_name = homo_pair_names[col_i]
            obs = float(observed[cell_i])
            s = null_summary(null, obs, cell_i)
            records.append({
                "threshold": stratum.threshold,
                "exact_hub_insert_count": stratum.exact_k,
                "condition_homo_epitope_pair": row_name,
                "target_homo_epitope_pair": col_name,
                "condition_homo_epitope_pair_plot_label": row_name.replace("/", "-"),
                "target_homo_epitope_pair_plot_label": col_name.replace("/", "-"),
                "comparison": f"{row_name} vs {col_name}",
                "condition_marker": row_name.split("/")[0],
                "target_marker": col_name.split("/")[0],
                "is_diagonal_same_homo_pair": row_i == col_i,
                "n_hubs_in_stratum": n_hubs,
                "n_qualifying_insert_slots_in_stratum": int(stratum.slot_pair_idx.size),
                "hub_cooccurrence_count": int(observed[cell_i]),
                "hub_cooccurrence_rate": float(observed[cell_i] / n_hubs) if n_hubs else math.nan,
                "null_median_hub_count": s["null_median"],
                "null_q025_hub_count": s["null_q025"],
                "null_q975_hub_count": s["null_q975"],
                "enrichment_vs_slot_shuffle_null": s["lift"],
                "log2_enrichment_vs_slot_shuffle_null": s["log2_lift"],
                "p_enrich": s["p_enrich"],
                "p_deplete": s["p_deplete"],
                "p_two_sided": s["p_two_sided"],
                "rank_percentile": s["rank_percentile"],
                "n_null_permutations": int(n_perm),
            })
            cell_i += 1
    return add_fdr(pd.DataFrame.from_records(records))


def safe_token(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value)).strip("_") or "x"


def marker_matrix(universe: Universe, values: Sequence[float]) -> pd.DataFrame:
    mat = np.full((len(universe.markers), len(universe.markers)), np.nan, dtype=float)
    for pi, value in enumerate(values):
        a, b = map(int, universe.pair_marker_idx[pi])
        mat[a, b] = mat[b, a] = value
    return pd.DataFrame(mat, index=universe.markers, columns=universe.markers)


def homo_matrix(df: pd.DataFrame, universe: Universe) -> pd.DataFrame:
    labels = [universe.pair_names[int(i)].replace("/", "-") for i in universe.homo_pair_idx]
    mat = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    for row in df.itertuples(index=False):
        i = str(row.condition_homo_epitope_pair).replace("/", "-")
        j = str(row.target_homo_epitope_pair).replace("/", "-")
        value = float(row.log2_enrichment_vs_slot_shuffle_null)
        mat.loc[i, j] = mat.loc[j, i] = value
    return mat


def significance_matrix_for_pairs(df: pd.DataFrame, universe: Universe, alpha: float) -> np.ndarray:
    sig = np.zeros((len(universe.markers), len(universe.markers)), dtype=np.int8)
    pair_to_i = {p: i for i, p in enumerate(universe.pair_names)}
    for row in df.itertuples(index=False):
        pi = pair_to_i.get(str(row.epitope_pair))
        if pi is None:
            continue
        a, b = map(int, universe.pair_marker_idx[pi])
        v = float(row.log2_enrichment_vs_slot_shuffle_null)
        mark = 0
        if np.isfinite(v) and v > 0 and np.isfinite(row.q_enrich) and row.q_enrich <= alpha:
            mark = 1
        elif np.isfinite(v) and v < 0 and np.isfinite(row.q_deplete) and row.q_deplete <= alpha:
            mark = -1
        sig[a, b] = sig[b, a] = mark
    return sig


def significance_matrix_for_homo(df: pd.DataFrame, universe: Universe, alpha: float) -> np.ndarray:
    labels = [universe.pair_names[int(i)] for i in universe.homo_pair_idx]
    idx = {name: i for i, name in enumerate(labels)}
    sig = np.zeros((len(labels), len(labels)), dtype=np.int8)
    for row in df.itertuples(index=False):
        i, j = idx.get(str(row.condition_homo_epitope_pair)), idx.get(str(row.target_homo_epitope_pair))
        if i is None or j is None:
            continue
        v = float(row.log2_enrichment_vs_slot_shuffle_null)
        mark = 0
        if np.isfinite(v) and v > 0 and np.isfinite(row.q_enrich) and row.q_enrich <= alpha:
            mark = 1
        elif np.isfinite(v) and v < 0 and np.isfinite(row.q_deplete) and row.q_deplete <= alpha:
            mark = -1
        sig[i, j] = sig[j, i] = mark
    return sig


def plot_heatmap(matrix: pd.DataFrame, output_png: Path, title: str, cbar_label: str, *, sig: np.ndarray | None = None) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    data = matrix.to_numpy(dtype=float)
    finite = np.isfinite(data)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * data.shape[1] + 3.5), max(5.5, 0.55 * data.shape[0] + 2.5)))
    if finite.any():
        vmax = np.nanpercentile(np.abs(data[finite]), 95)
        vmax = max(0.5, float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0)
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("#e6e6e6")
        image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=-vmax, vmax=vmax)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isfinite(data[i, j]):
                    continue
                label = f"{data[i, j]:.2f}"
                if sig is not None:
                    label += "\n▲" if sig[i, j] == 1 else "\n▼" if sig[i, j] == -1 else ""
                ax.text(j, i, label, ha="center", va="center", fontsize=10, color="black")
        cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label(cbar_label)
    else:
        ax.text(0.5, 0.5, "No finite enrichment values", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    universe = build_universe()

    print(f"Loading {args.h5ad!r} …", flush=True)
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError("Install anndata to read .h5ad files: pip install anndata") from exc
    ann = ad.read_h5ad(args.h5ad)
    parsed = parse_exact2_inserts(ann, universe, strict_lengths=args.strict_lengths, report_every=args.report_every)
    if parsed.insert_pair_idx.size == 0:
        raise RuntimeError("No exactly-two-match inserts were found; no requested outputs can be computed.")

    thresholds = parse_thresholds(args.thresholds)
    threshold = args.target_threshold if args.target_threshold is not None else min(thresholds)
    if threshold not in thresholds:
        warnings.warn(f"target threshold {threshold} is not listed in --thresholds={args.thresholds!r}; using it anyway")
    stratum = make_exact_stratum(parsed, threshold, args.exact_hub_insert_count)
    if stratum.hub_ids.size == 0:
        raise RuntimeError(f"No hubs have exactly {args.exact_hub_insert_count} qualifying inserts at trust >= {threshold}.")

    print(
        f"Selected trust >= {threshold}, exact hub insert count = {args.exact_hub_insert_count}: "
        f"{stratum.hub_ids.size:,} hubs, {stratum.slot_pair_idx.size:,} insert slots",
        flush=True,
    )
    grids = shared_slot_shuffle_null(stratum, universe, args.n_permutations, rng, args.null_insert_budget)
    n_perm = int(grids["n_null_permutations"])

    between_df = between_records(stratum, universe, grids["between_observed"], grids["between_null"], n_perm)
    homo_df = homo_records(stratum, universe, grids["homo_observed"], grids["homo_null"], n_perm)

    token = f"t{safe_token(threshold)}_k{safe_token(args.exact_hub_insert_count)}"
    between_dir = outdir / "epitope_colocalization_grids_by_exact_hub_insert_count"
    homo_dir = outdir / "conditional_lift_heatmaps_by_exact_hub_insert_count"
    between_csv = between_dir / f"epitope_colocalization_between_insert_log2_enrichment_{token}.csv"
    homo_csv = homo_dir / f"homo_epitope_pair_to_homo_epitope_pair_log2_enrichment_{token}.csv"
    between_df.to_csv(between_csv, index=False)
    homo_df.to_csv(homo_csv, index=False)

    between_mat = marker_matrix(universe, between_df["log2_enrichment_vs_slot_shuffle_null"].to_numpy(dtype=float))
    homo_mat = homo_matrix(homo_df, universe)
    between_mat.to_csv(between_dir / f"epitope_colocalization_between_insert_log2_enrichment_{token}.matrix.csv")
    homo_mat.to_csv(homo_dir / f"homo_epitope_pair_to_homo_epitope_pair_log2_enrichment_{token}.matrix.csv")

    if not args.no_plots:
        plot_heatmap(
            homo_mat,
            homo_dir / f"homo_epitope_pair_to_homo_epitope_pair_log2_enrichment_{token}.png",
            f"Homo-epitope-pair co-localization enrichment, trust ≥ {threshold}, exact hub inserts = {args.exact_hub_insert_count}",
            "log2 enrichment vs exact-k insert-label shuffle",
            sig=significance_matrix_for_homo(homo_df, universe, args.alpha),
        )
        plot_heatmap(
            between_mat,
            between_dir / f"epitope_colocalization_between_insert_log2_enrichment_{token}.png",
            f"Different-insert epitope co-localization enrichment, trust ≥ {threshold}, exact hub inserts = {args.exact_hub_insert_count}",
            "log2 enrichment vs exact-k insert-label shuffle",
            sig=significance_matrix_for_pairs(between_df, universe, args.alpha),
        )

    accepted = parsed.insert_trust >= int(threshold)
    hub_counts_at_threshold = np.bincount(parsed.insert_hub[accepted], minlength=parsed.n_hubs).astype(np.int64)
    parse_summary = {
        "h5ad": str(args.h5ad),
        "n_hubs": int(parsed.n_hubs),
        "n_inserts_total": parsed.n_inserts_total,
        "n_inserts_with_any_recognized_tag": parsed.n_inserts_with_any_tag,
        "n_qualifying_inserts_exactly_2_matches": int(parsed.insert_pair_idx.size),
        "insert_tagcount_distribution": parsed.tagcount_hist,
        "exact2_insert_pair_distribution": {
            universe.pair_names[int(k)]: int(v) for k, v in Counter(parsed.insert_pair_idx.tolist()).most_common()
        },
        "n_rows_with_length_mismatch": parsed.length_mismatch_rows,
        "n_malformed_trust_values": parsed.malformed_trust_values,
        "n_empty_or_missing_rows": parsed.empty_or_missing_rows,
        "max_trust_among_aligned_inserts": parsed.max_trust,
        "top_unrecognized_leading_kmers": [
            {"leading_kmer": k, "count": int(v)} for k, v in parsed.unrecognized_leading_kmers.most_common(20)
        ],
        "selected_analysis": {
            "threshold": int(threshold),
            "exact_hub_insert_count": int(args.exact_hub_insert_count),
            "n_hubs_in_stratum": int(stratum.hub_ids.size),
            "n_qualifying_insert_slots_in_stratum": int(stratum.slot_pair_idx.size),
            "n_null_permutations": n_perm,
            "alpha": float(args.alpha),
        },
        "hub_insert_count_distribution_at_selected_threshold": {
            str(int(k)): int(np.count_nonzero(hub_counts_at_threshold == k))
            for k in np.unique(hub_counts_at_threshold) if int(k) > 0
        },
        "markers": universe.markers,
        "insert_pair_labels": universe.pair_names,
        "target_motifs": TARGET_MOTIFS,
        "outputs": {
            "between_insert_epitope_csv": str(between_csv),
            "homo_epitope_pair_csv": str(homo_csv),
        },
    }
    write_json(outdir / "parse_summary.json", parse_summary)
    write_json(outdir / f"selected_stratum_metadata_{token}.json", parse_summary["selected_analysis"])
    print(f"Done. Focused outputs written to {outdir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Focused exact-k cDNA co-localization outputs only.")
    p.add_argument("--h5ad", default="data_dir/cdna_only.h5ad")
    p.add_argument("--outdir", default="cdna_hub_pair_permutation_outputs")
    p.add_argument("--thresholds", default="1,2", help="Parsed for compatibility; the selected threshold defaults to the minimum value.")
    p.add_argument("--target-threshold", type=int, default=None, help="Trust threshold to plot; default is min(--thresholds), so '1,2' gives t1.")
    p.add_argument("--exact-hub-insert-count", type=int, default=2, help="Exact qualifying insert count, default k=2.")
    p.add_argument("--n-permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--null-insert-budget", type=int, default=DEFAULT_NULL_INSERT_BUDGET)
    p.add_argument("--report-every", type=int, default=100_000)
    p.add_argument("--strict-lengths", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_permutations < 1:
        parser.error("--n-permutations must be at least 1")
    if args.exact_hub_insert_count < 1:
        parser.error("--exact-hub-insert-count must be at least 1")
    if not (0 < args.alpha <= 1):
        parser.error("--alpha must be in (0, 1]")
    try:
        run(args)
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

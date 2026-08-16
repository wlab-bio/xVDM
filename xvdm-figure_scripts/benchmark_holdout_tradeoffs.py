#!/usr/bin/env python3
"""
Minimal train/heldout benchmark for register_zf.

Output contract
---------------
The benchmark root contains only:
  runs/                         per-run feature bundles, metrics, and metadata
  pair_library/                 reusable disjoint pole-pair libraries
  all_pair_metrics.csv          one row per pair × split × stage
  all_runs.csv                  one row per completed run
  run_level_summary.csv         per-run split/stage means
  train_holdout_gap_by_run.csv  heldout-minus-train gaps by run and stage

The saved run bundles are intentionally limited to the arrays needed by the
PC-correlogram plotting script plus lightweight provenance metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1.0e-10
FEATURE_SCALE = "ratio_feature_01_rank_rescaled_0_1"
SPATIAL_SCALE = "post_transport_postrank_over_comparable_slice_nodes"

BACKEND_SIDECARS_TO_PRUNE = {
    "aggregated_to_slice_match_csr.npz",
    "slice_to_aggregated_match_csr.npz",
    "slice_capacity_targets.npz",
    "slice_capacity_spatial_diagnostics.json",
    "matching_context_base.npz",
    "matching_refinement_context.npz",
    "progress.log",
    "progress_status.json",
    "register_zf_benchmark_adapter_metadata.json",
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    slice_path: str
    agg_path: str
    time_value: str
    no_time_filter: bool = False


@dataclass
class DatasetContext:
    spec: DatasetSpec
    source_adata: Any
    source_coords: np.ndarray
    source_node_mass: np.ndarray
    source_smoother: Any
    agg_dataset: dict[str, Any]
    shared_genes: list[str]


def _json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_json_default)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(path: str | Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_register_module(path: str):
    module_path = Path(path).expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find register script: {module_path}")
    spec = importlib.util.spec_from_file_location("register_zf_benchmark_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def call_supported(func, /, *args, **kwargs):
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return func(*args, **kwargs)
    return func(*args, **{k: v for k, v in kwargs.items() if k in sig.parameters})


def parse_int_list(text: str) -> list[int]:
    out = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError(f"Expected a comma-separated integer list, got {text!r}")
    return out


def boolish(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def read_manifest(path: str) -> list[DatasetSpec]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"dataset_id", "slice", "agg", "time_value"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        specs = [
            DatasetSpec(
                dataset_id=str(row["dataset_id"]).strip(),
                slice_path=str(row["slice"]).strip(),
                agg_path=str(row["agg"]).strip(),
                time_value=str(row["time_value"]).strip(),
                no_time_filter=boolish(row.get("no_time_filter", False)),
            )
            for row in reader
        ]
    if not specs:
        raise ValueError("Manifest contained no datasets.")
    return specs


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def shared_genes_digest(genes: Iterable[str]) -> str:
    return stable_hash(sorted(str(g).lower() for g in genes))


def load_context(r, spec: DatasetSpec, *, min_feature_count: int) -> DatasetContext:
    source_adata = r.load_source_adata(
        spec.slice_path,
        time_value=spec.time_value,
        no_time_filter=spec.no_time_filter,
    )
    source_coords = np.asarray(r.get_spatial_coords_from_adata(source_adata), dtype=float)
    source_node_mass = np.asarray(r.get_node_total_mass(source_adata), dtype=float)
    source_smoother, _ = r.build_knn_smoothing_operator(source_coords, r.GENE_POLE_SMOOTH_K)
    agg_dataset = r.load_aggregated_gse_h5ad(spec.agg_path)
    shared_genes = call_supported(r.collect_shared_genes, agg_dataset, source_adata, min_feature_count=min_feature_count)
    return DatasetContext(spec, source_adata, source_coords, source_node_mass, source_smoother, agg_dataset, list(shared_genes))


def pair_library_signature(ctx: DatasetContext, *, library_pair_count: int, max_genes_per_pole: int, abundance_threshold: int) -> dict[str, Any]:
    return {
        "dataset_id": ctx.spec.dataset_id,
        "slice": ctx.spec.slice_path,
        "agg": ctx.spec.agg_path,
        "time_value": ctx.spec.time_value,
        "no_time_filter": bool(ctx.spec.no_time_filter),
        "library_pair_count": int(library_pair_count),
        "max_genes_per_pole": int(max_genes_per_pole),
        "abundance_threshold": int(abundance_threshold),
        "shared_gene_count": int(len(ctx.shared_genes)),
        "shared_genes_digest": shared_genes_digest(ctx.shared_genes),
    }


def build_or_load_pair_library(r, ctx: DatasetContext, *, out_dir: Path, library_pair_count: int, max_genes_per_pole: int, abundance_threshold: int) -> list[dict[str, Any]]:
    ensure_dir(out_dir)
    pairs_path = out_dir / "pair_library.json"
    meta_path = out_dir / "pair_library.meta.json"
    desired = pair_library_signature(
        ctx,
        library_pair_count=library_pair_count,
        max_genes_per_pole=max_genes_per_pole,
        abundance_threshold=abundance_threshold,
    )
    if pairs_path.exists() and meta_path.exists():
        try:
            pairs = read_json(pairs_path)
            if read_json(meta_path) == desired and len(pairs) >= library_pair_count:
                return list(pairs[:library_pair_count])
        except Exception:
            pass

    pairs = r.identify_typeAB_gene_pairs(
        ctx.source_adata,
        ctx.shared_genes,
        str(out_dir),
        num_pairs=int(library_pair_count),
        num_top_genes=int(max_genes_per_pole),
        abundance_threshold=int(abundance_threshold),
        coords=ctx.source_coords,
        smoothing_operator=ctx.source_smoother,
    )
    if len(pairs) < library_pair_count:
        raise ValueError(f"Requested {library_pair_count} disjoint pole pairs, but only found {len(pairs)}")
    write_json(pairs_path, pairs)
    write_json(meta_path, desired)
    return list(pairs[:library_pair_count])


def truncate_pair(pair: dict[str, Any], genes_per_pole: int, suffix: str) -> dict[str, Any]:
    g = int(genes_per_pole)
    out = dict(pair)
    out["typeA_genes"] = [str(x).lower() for x in list(pair["typeA_genes"])[:g]]
    out["typeB_genes"] = [str(x).lower() for x in list(pair["typeB_genes"])[:g]]
    if len(out["typeA_genes"]) < g or len(out["typeB_genes"]) < g:
        raise ValueError(f"Pair {pair.get('pair_id')} does not contain {g} genes per pole")
    out["pair_id"] = f"{pair.get('pair_id', 'pair')}_{suffix}"
    return out


def pair_gene_set(pair: dict[str, Any]) -> set[str]:
    return {str(g).lower() for g in list(pair.get("typeA_genes", [])) + list(pair.get("typeB_genes", []))}


def validate_pair_disjointness(train_pairs: Iterable[dict[str, Any]], heldout_pairs: Iterable[dict[str, Any]]) -> None:
    train_genes: set[str] = set()
    for pair in train_pairs:
        genes = pair_gene_set(pair)
        overlap = train_genes & genes
        if overlap:
            raise ValueError(f"Training gene overlap detected: {sorted(overlap)[:8]}")
        train_genes |= genes
    for pair in heldout_pairs:
        overlap = train_genes & pair_gene_set(pair)
        if overlap:
            raise ValueError(f"Heldout pair {pair.get('pair_id')} overlaps training genes: {sorted(overlap)[:8]}")


def split_indices(n_pairs: int, fold: int, seed: int, n_heldout: int, min_train: int) -> tuple[np.ndarray, np.ndarray]:
    if n_pairs < n_heldout + min_train:
        raise ValueError(f"Need at least {n_heldout + min_train} pairs; library has {n_pairs}")
    rng = np.random.default_rng(int(seed) + int(fold) * 1_000_003)
    order = rng.permutation(n_pairs)
    return order[n_heldout:].astype(int), order[:n_heldout].astype(int)


def rank_rescale_over_mask(values: np.ndarray, mask: np.ndarray, *, neutral: float = 0.5) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    valid = np.asarray(mask, dtype=bool).ravel() & np.isfinite(values)
    out = np.full(values.shape, float(neutral), dtype=float)
    if int(valid.sum()) < 2:
        return out
    ranks = stats.rankdata(values[valid], method="average").astype(float)
    lo, hi = float(np.min(ranks)), float(np.max(ranks))
    out[valid] = (ranks - lo) / (hi - lo) if hi > lo else float(neutral)
    return out


def safe_corr(x: np.ndarray, y: np.ndarray, *, method: str) -> float | None:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return None
    xv, yv = x[valid], y[valid]
    if float(np.std(xv)) <= EPS or float(np.std(yv)) <= EPS:
        return None
    if method == "spearman":
        rho = stats.spearmanr(xv, yv).correlation
        return None if rho is None or not np.isfinite(rho) else float(rho)
    rho = float(np.corrcoef(xv, yv)[0, 1])
    return rho if np.isfinite(rho) else None


def vector_metrics(pred: np.ndarray, target: np.ndarray, *, support: np.ndarray | None, neutral: float) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    valid = np.isfinite(pred) & np.isfinite(target)
    if support is not None:
        valid &= np.asarray(support, dtype=float).ravel() > EPS
    if not np.any(valid):
        return {"n_slice_nodes_compared": 0, "feature_scale": FEATURE_SCALE, "spatial_scale": SPATIAL_SCALE}

    pv, tv = pred[valid], target[valid]
    diff = pv - tv
    neutral_diff = float(neutral) - tv
    pred_rank = rank_rescale_over_mask(pred, valid, neutral=neutral)[valid]
    target_rank = rank_rescale_over_mask(target, valid, neutral=neutral)[valid]
    spatial_diff = pred_rank - target_rank
    neutral_mae = float(np.mean(np.abs(neutral_diff)))
    neutral_rmse = float(np.sqrt(np.mean(neutral_diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return {
        "n_slice_nodes_compared": int(valid.sum()),
        "mae": mae,
        "rmse": rmse,
        "bias": float(np.mean(diff)),
        "neutral_mae": neutral_mae,
        "neutral_rmse": neutral_rmse,
        "mae_improvement_over_neutral": float(neutral_mae - mae),
        "rmse_improvement_over_neutral": float(neutral_rmse - rmse),
        "mae_ratio_to_neutral": float(mae / neutral_mae) if neutral_mae > EPS else None,
        "rmse_ratio_to_neutral": float(rmse / neutral_rmse) if neutral_rmse > EPS else None,
        "spearman_feature01": safe_corr(pv, tv, method="spearman"),
        "pearson_feature01": safe_corr(pv, tv, method="pearson"),
        "spatial_postrank_mae": float(np.mean(np.abs(spatial_diff))),
        "spatial_postrank_rmse": float(np.sqrt(np.mean(spatial_diff ** 2))),
        "spatial_postrank_spearman": safe_corr(pred_rank, target_rank, method="spearman"),
        "spatial_postrank_pearson": safe_corr(pred_rank, target_rank, method="pearson"),
        "pred_mean": float(np.mean(pv)),
        "pred_std": float(np.std(pv)),
        "target_mean": float(np.mean(tv)),
        "target_std": float(np.std(tv)),
        "feature_scale": FEATURE_SCALE,
        "spatial_scale": SPATIAL_SCALE,
    }


def aggregate_assigned_features(r, agg_features: np.ndarray, a_to_slice: np.ndarray, n_slice: int) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(r, "_aggregate_assigned_features_to_slice_nodes"):
        return r._aggregate_assigned_features_to_slice_nodes(np.asarray(agg_features, dtype=float), np.asarray(a_to_slice, dtype=int), int(n_slice))
    agg_features = np.asarray(agg_features, dtype=float)
    if agg_features.ndim == 1:
        agg_features = agg_features[:, None]
    a_to_slice = np.asarray(a_to_slice, dtype=int).ravel()
    counts = np.bincount(a_to_slice, minlength=int(n_slice)).astype(int)[: int(n_slice)]
    sums = np.zeros((int(n_slice), agg_features.shape[1]), dtype=float)
    for j in range(agg_features.shape[1]):
        np.add.at(sums[:, j], a_to_slice, agg_features[:, j])
    means = np.full_like(sums, np.nan, dtype=float)
    means[counts > 0] = sums[counts > 0] / counts[counts > 0, None]
    return means, counts


def score_feature_matrix(r, *, split: str, stage: str, pair_ids: list[str], slice_feature_01: np.ndarray, agg_feature_01: np.ndarray, a_to_slice: np.ndarray, n_slice: int, slice_support: np.ndarray, rank_neutral: float) -> list[dict[str, Any]]:
    feature_mean, assigned_count = aggregate_assigned_features(r, agg_feature_01, a_to_slice, n_slice)
    slice_feature_01 = np.asarray(slice_feature_01, dtype=float)
    slice_support = np.asarray(slice_support, dtype=float)
    if slice_feature_01.ndim == 1:
        slice_feature_01 = slice_feature_01[:, None]
    if slice_support.ndim == 1:
        slice_support = slice_support[:, None]
    rows: list[dict[str, Any]] = []
    for j, pair_id in enumerate(pair_ids):
        m = vector_metrics(
            feature_mean[:, j],
            slice_feature_01[:, j],
            support=slice_support[:, j] if j < slice_support.shape[1] else None,
            neutral=float(rank_neutral),
        )
        m.update({
            "split": split,
            "stage": stage,
            "pair_id": str(pair_id),
            "assigned_slice_nodes_with_count": int(np.sum(np.asarray(assigned_count) > 0)),
        })
        rows.append(m)
    return rows


def write_source_bundle(path: Path, *, source_field: dict[str, Any], ratio_rank: np.ndarray, feature_01: np.ndarray, node_mass: np.ndarray) -> None:
    np.savez_compressed(
        path,
        coords=np.asarray(source_field["coords"], dtype=float),
        ratio=np.asarray(source_field["ratio"], dtype=float),
        ratio_rank=np.asarray(ratio_rank, dtype=float),
        ratio_feature_01=np.asarray(feature_01, dtype=float),
        support=np.asarray(source_field["support"], dtype=float),
        typeA_signal=np.asarray(source_field["typeA_signal"], dtype=float),
        typeB_signal=np.asarray(source_field["typeB_signal"], dtype=float),
        node_total_mass=np.asarray(node_mass, dtype=float),
        pair_ids=np.asarray(source_field["pair_ids"], dtype=object),
    )


def write_feature_maps(r, out_dir: Path, *, slice_coords: np.ndarray, agg_feature_01: np.ndarray, pair_ids: list[str], a_to_slice_base: np.ndarray, a_to_slice_final: np.ndarray, slice_capacity: np.ndarray) -> str:
    base_mean, base_count = aggregate_assigned_features(r, agg_feature_01, a_to_slice_base, int(slice_coords.shape[0]))
    final_mean, final_count = aggregate_assigned_features(r, agg_feature_01, a_to_slice_final, int(slice_coords.shape[0]))
    out_path = out_dir / "slice_assigned_aggregated_feature_maps.npz"
    np.savez_compressed(
        out_path,
        coords=np.asarray(slice_coords, dtype=float),
        feature_mean_base=np.asarray(base_mean, dtype=float),
        feature_mean_final=np.asarray(final_mean, dtype=float),
        feature_delta=np.asarray(final_mean - base_mean, dtype=float),
        count_base=np.asarray(base_count, dtype=int),
        count_final=np.asarray(final_count, dtype=int),
        slice_capacity=np.asarray(slice_capacity, dtype=int),
        pair_ids=np.asarray(pair_ids, dtype=object),
    )
    return str(out_path)


def compute_source_features(r, ctx: DatasetContext, pairs: list[dict[str, Any]], *, slice_smooth_k: int, rank_neutral: float) -> dict[str, Any]:
    source_field = r.compute_source_ratio_fields_multi(
        ctx.source_adata,
        pairs,
        coords=ctx.source_coords,
        smooth_k=int(slice_smooth_k),
        smoothing_operator=ctx.source_smoother if int(slice_smooth_k) == int(r.GENE_POLE_SMOOTH_K) else None,
    )
    ratio_rank = r.rank_transform_ratio_fields(source_field["ratio"], source_field["support"], neutral=float(rank_neutral))
    feature_01 = r.rescale_rank_features_to_unit_interval(ratio_rank, source_field["support"], neutral=float(rank_neutral))
    return {"field": source_field, "ratio_rank": ratio_rank, "feature_01": feature_01, "pair_ids": [str(x) for x in source_field["pair_ids"]]}


def compute_agg_features(r, ctx: DatasetContext, pairs: list[dict[str, Any]], *, agg_smooth_k: int, rank_neutral: float, tree_workers: int | None) -> dict[str, Any]:
    agg_coords, typeA_counts, typeB_counts = r.build_pair_count_matrices_from_aggregated(ctx.agg_dataset, pairs)
    ratio_raw, support, knn_meta = call_supported(
        r.perform_knn_analysis_with_support_multi,
        agg_coords,
        typeA_counts,
        typeB_counts,
        k=int(agg_smooth_k),
        workers=tree_workers,
    )
    ratio_rank = r.rank_transform_ratio_fields(ratio_raw, support, neutral=float(rank_neutral))
    feature_01 = r.rescale_rank_features_to_unit_interval(ratio_rank, support, neutral=float(rank_neutral))
    return {"coords": agg_coords, "ratio_raw": ratio_raw, "support": support, "ratio_rank": ratio_rank, "feature_01": feature_01, "knn_meta": knn_meta}


def prune_backend_sidecars(run_dir: Path) -> None:
    for name in BACKEND_SIDECARS_TO_PRUNE:
        path = run_dir / name
        if path.exists():
            try:
                path.unlink()
            except IsADirectoryError:
                shutil.rmtree(path)


def fit_transport_for_pairs(r, ctx: DatasetContext, train_pairs: list[dict[str, Any]], *, out_dir: Path, args: argparse.Namespace, tree_workers: int | None) -> dict[str, Any]:
    ensure_dir(out_dir)
    source = compute_source_features(r, ctx, train_pairs, slice_smooth_k=args.slice_smooth_k, rank_neutral=args.rank_neutral)
    write_source_bundle(
        out_dir / "slice_smoothed_ratio_fields.npz",
        source_field=source["field"],
        ratio_rank=source["ratio_rank"],
        feature_01=source["feature_01"],
        node_mass=ctx.source_node_mass,
    )
    agg = compute_agg_features(r, ctx, train_pairs, agg_smooth_k=args.agg_smooth_k, rank_neutral=args.rank_neutral, tree_workers=tree_workers)
    match = call_supported(
        r.run_sparse_graph_matching_on_ratio_vectors,
        agg_h5ad_path=ctx.spec.agg_path,
        XA_features_01=np.asarray(agg["feature_01"], dtype=float),
        YB_features_01=np.asarray(source["feature_01"], dtype=float),
        YB_coords=ctx.source_coords,
        source_node_mass=ctx.source_node_mass,
        output_dir=str(out_dir),
        k0=int(args.match_k0),
        k_max=int(args.match_k_max),
        lam_dir=args.match_lam_dir,
        refine_iter=int(args.match_refine_iter),
        tree_workers=tree_workers,
        slice_capacity_mode=args.slice_capacity_mode,
    )
    a_to_base = np.asarray(match["a_to_slice_base"], dtype=int)
    a_to_final = np.asarray(match.get("a_to_slice_final", match["a_to_slice"]), dtype=int)
    maps_path = write_feature_maps(
        r,
        out_dir,
        slice_coords=ctx.source_coords,
        agg_feature_01=np.asarray(agg["feature_01"], dtype=float),
        pair_ids=source["pair_ids"],
        a_to_slice_base=a_to_base,
        a_to_slice_final=a_to_final,
        slice_capacity=np.asarray(match["slice_capacity"], dtype=int),
    )
    train_rows = []
    for stage, assignment in [("base", a_to_base), ("final", a_to_final)]:
        train_rows.extend(score_feature_matrix(
            r,
            split="train",
            stage=stage,
            pair_ids=source["pair_ids"],
            slice_feature_01=np.asarray(source["feature_01"], dtype=float),
            agg_feature_01=np.asarray(agg["feature_01"], dtype=float),
            a_to_slice=assignment,
            n_slice=ctx.source_coords.shape[0],
            slice_support=np.asarray(source["field"]["support"], dtype=float),
            rank_neutral=float(args.rank_neutral),
        ))
    write_json(out_dir / "benchmark_fit_metadata.json", {
        "dataset_id": ctx.spec.dataset_id,
        "slice": ctx.spec.slice_path,
        "agg": ctx.spec.agg_path,
        "time_value": ctx.spec.time_value,
        "train_pair_ids": source["pair_ids"],
        "train_pairs": train_pairs,
        "slice_assigned_aggregated_feature_maps_npz": maps_path,
        "matching": {
            "objective_initial": match.get("objective_initial"),
            "objective_final": match.get("objective_final"),
            "effective_row_k_max": match.get("effective_row_k_max"),
            "effective_edge_count": match.get("effective_edge_count"),
            "fraction_reassigned_by_graph_refinement": match.get("fraction_reassigned_by_graph_refinement"),
            "median_move_distance_normalized": match.get("median_move_distance_normalized"),
            "slice_capacity_mode": match.get("slice_capacity_mode"),
            "graph_regularization_used": match.get("graph_regularization_used"),
        },
        "agg_knn_meta": agg["knn_meta"],
    })
    prune_backend_sidecars(out_dir)
    return {
        "match_result": match,
        "train_metric_rows": train_rows,
        "a_to_slice_base": a_to_base,
        "a_to_slice_final": a_to_final,
        "slice_capacity": np.asarray(match["slice_capacity"], dtype=int),
    }


def evaluate_heldout_pair(r, ctx: DatasetContext, pair: dict[str, Any], *, out_dir: Path, fit: dict[str, Any], args: argparse.Namespace, tree_workers: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_id = str(pair.get("pair_id", "heldout_pair"))
    heldout_dir = ensure_dir(out_dir / f"heldout_{pair_id}")
    source = compute_source_features(r, ctx, [pair], slice_smooth_k=args.slice_smooth_k, rank_neutral=args.rank_neutral)
    slice_path = heldout_dir / "slice_smoothed_ratio_fields.npz"
    write_source_bundle(
        slice_path,
        source_field=source["field"],
        ratio_rank=source["ratio_rank"],
        feature_01=source["feature_01"],
        node_mass=ctx.source_node_mass,
    )
    agg = compute_agg_features(r, ctx, [pair], agg_smooth_k=args.agg_smooth_k, rank_neutral=args.rank_neutral, tree_workers=tree_workers)
    maps_path = write_feature_maps(
        r,
        heldout_dir,
        slice_coords=ctx.source_coords,
        agg_feature_01=np.asarray(agg["feature_01"], dtype=float),
        pair_ids=source["pair_ids"],
        a_to_slice_base=np.asarray(fit["a_to_slice_base"], dtype=int),
        a_to_slice_final=np.asarray(fit["a_to_slice_final"], dtype=int),
        slice_capacity=np.asarray(fit["slice_capacity"], dtype=int),
    )
    rows: list[dict[str, Any]] = []
    for stage, assignment in [("base", fit["a_to_slice_base"]), ("final", fit["a_to_slice_final"])]:
        rows.extend(score_feature_matrix(
            r,
            split="heldout",
            stage=stage,
            pair_ids=source["pair_ids"],
            slice_feature_01=np.asarray(source["feature_01"], dtype=float),
            agg_feature_01=np.asarray(agg["feature_01"], dtype=float),
            a_to_slice=np.asarray(assignment, dtype=int),
            n_slice=ctx.source_coords.shape[0],
            slice_support=np.asarray(source["field"]["support"], dtype=float),
            rank_neutral=float(args.rank_neutral),
        ))
    summary = {
        "pair_id": pair_id,
        "heldout_pair": pair,
        "slice_output_npz": str(slice_path),
        "slice_assigned_feature_maps_npz": str(maps_path),
        "metric_rows": rows,
        "agg_knn_meta": agg["knn_meta"],
    }
    write_json(heldout_dir / "heldout_evaluation.json", summary)
    return summary, rows


def add_run_columns(rows: list[dict[str, Any]], **cols: Any) -> list[dict[str, Any]]:
    return [{**row, **cols} for row in rows]


def make_run_signature(args: argparse.Namespace, spec: DatasetSpec, *, fold: int, run_id: str, budget: int, genes_per_pole: int, num_train_pairs: int, train_pairs: list[dict[str, Any]], heldout_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "dataset_id": spec.dataset_id,
        "slice": spec.slice_path,
        "agg": spec.agg_path,
        "time_value": spec.time_value,
        "fold": int(fold),
        "run_id": run_id,
        "budget": int(budget),
        "genes_per_pole": int(genes_per_pole),
        "num_train_pairs": int(num_train_pairs),
        "train_pairs": train_pairs,
        "heldout_pairs": heldout_pairs,
        "eval_genes_per_pole": int(args.eval_genes_per_pole),
        "slice_smooth_k": int(args.slice_smooth_k),
        "agg_smooth_k": int(args.agg_smooth_k),
        "rank_neutral": float(args.rank_neutral),
        "match_k0": int(args.match_k0),
        "match_k_max": int(args.match_k_max),
        "match_lam_dir": args.match_lam_dir,
        "match_refine_iter": int(args.match_refine_iter),
        "slice_capacity_mode": str(args.slice_capacity_mode),
    }
    payload["signature"] = stable_hash(payload)
    return payload


def required_run_files(heldout_pairs: list[dict[str, Any]]) -> list[str]:
    files = [
        "slice_smoothed_ratio_fields.npz",
        "slice_assigned_aggregated_feature_maps.npz",
        "benchmark_fit_metadata.json",
        "heldout_evaluation_manifest.json",
        "run_pair_metrics.csv",
        "run_row.json",
    ]
    for pair in heldout_pairs:
        pid = str(pair.get("pair_id", "heldout_pair"))
        files.extend([
            f"heldout_{pid}/slice_smoothed_ratio_fields.npz",
            f"heldout_{pid}/slice_assigned_aggregated_feature_maps.npz",
            f"heldout_{pid}/heldout_evaluation.json",
        ])
    return files


def load_completed_run(run_dir: Path, signature: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    marker = run_dir / "run_complete.json"
    if not marker.exists():
        return None
    try:
        meta = read_json(marker)
        if meta.get("signature") != signature.get("signature"):
            return None
        for rel in meta.get("required_files", []):
            p = run_dir / str(rel)
            if not p.exists() or p.stat().st_size == 0:
                return None
        return pd.read_csv(run_dir / "run_pair_metrics.csv").to_dict("records"), read_json(run_dir / "run_row.json")
    except Exception:
        return None


def finalize_run(run_dir: Path, *, signature: dict[str, Any], heldout_pairs: list[dict[str, Any]], metric_rows: list[dict[str, Any]], run_row: dict[str, Any]) -> dict[str, Any]:
    required = required_run_files(heldout_pairs)
    write_csv(run_dir / "run_pair_metrics.csv", metric_rows)
    run_row = dict(run_row)
    run_row["run_signature"] = signature["signature"]
    write_json(run_dir / "run_row.json", run_row)
    missing = [rel for rel in required if not (run_dir / rel).exists()]
    if missing:
        raise RuntimeError(f"Cannot mark run complete; missing files: {missing[:8]}")
    write_json(run_dir / "run_complete.json", {**signature, "status": "complete", "required_files": required, "metric_row_count": len(metric_rows)})
    return run_row


def make_summaries(all_pair_metrics_path: Path, output_root: Path) -> None:
    if not all_pair_metrics_path.exists():
        write_csv(output_root / "run_level_summary.csv", [])
        write_csv(output_root / "train_holdout_gap_by_run.csv", [])
        return
    df = pd.read_csv(all_pair_metrics_path)
    if df.empty:
        write_csv(output_root / "run_level_summary.csv", [])
        write_csv(output_root / "train_holdout_gap_by_run.csv", [])
        return
    group_cols = ["dataset_id", "fold", "run_id", "gene_budget_per_side", "genes_per_pole", "num_train_pairs", "stage", "split"]
    metric_cols = [
        "n_slice_nodes_compared", "mae", "rmse", "bias", "neutral_mae", "neutral_rmse",
        "mae_improvement_over_neutral", "rmse_improvement_over_neutral",
        "mae_ratio_to_neutral", "rmse_ratio_to_neutral", "spearman_feature01", "pearson_feature01",
        "spatial_postrank_mae", "spatial_postrank_rmse", "spatial_postrank_spearman", "spatial_postrank_pearson",
        "pred_mean", "pred_std", "target_mean", "target_std", "assigned_slice_nodes_with_count",
    ]
    present = [c for c in metric_cols if c in df.columns]
    summary = df.groupby(group_cols, dropna=False).agg(
        **{c: (c, "mean") for c in present},
        n_pairs=("pair_id", "nunique"),
    ).reset_index()
    write_csv(output_root / "run_level_summary.csv", summary)

    keys = ["dataset_id", "fold", "run_id", "gene_budget_per_side", "genes_per_pole", "num_train_pairs", "stage"]
    train = summary[summary["split"].astype(str) == "train"]
    hold = summary[summary["split"].astype(str) == "heldout"]
    gaps = pd.merge(train, hold, on=keys, suffixes=("_train", "_heldout"), how="inner")
    for c in present:
        tc, hc = f"{c}_train", f"{c}_heldout"
        if tc in gaps.columns and hc in gaps.columns:
            gaps[f"generalization_gap_{c}"] = gaps[hc] - gaps[tc]
    write_csv(output_root / "train_holdout_gap_by_run.csv", gaps)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal register_zf train/heldout benchmark")
    p.add_argument("--register-script", required=True, help="Adapter or register_zf.py module to import")
    p.add_argument("--manifest", required=True, help="CSV with dataset_id,slice,agg,time_value[,no_time_filter]")
    p.add_argument("--output-root", required=True)
    p.add_argument("--gene-budgets-per-side", default="5,10,20")
    p.add_argument("--genes-per-pole-values", default="1,2,5")
    p.add_argument("--allow-floor-budget", action="store_true")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--heldout-pairs", type=int, default=5)
    p.add_argument("--eval-genes-per-pole", type=int, default=1)
    p.add_argument("--library-pair-count", type=int, default=None)
    p.add_argument("--max-genes-per-pole-library", type=int, default=5)
    p.add_argument("--abundance-threshold", type=int, default=10)
    p.add_argument("--min-feature-count", type=int, default=5)
    p.add_argument("--slice-smooth-k", type=int, default=30)
    p.add_argument("--agg-smooth-k", type=int, default=10)
    p.add_argument("--rank-neutral", type=float, default=0.5)
    p.add_argument("--slice-capacity-mode", default=os.getenv("REGZF_ADAPTER_SLICE_CAPACITY_MODE", "mass_exact"))
    p.add_argument("--match-k0", type=int, default=16)
    p.add_argument("--match-k-max", type=int, default=256)
    p.add_argument("--match-lam-dir", type=float, default=None)
    p.add_argument("--match-refine-iter", type=int, default=2)
    p.add_argument("--tree-workers", type=int, default=0)
    p.add_argument("--force-rerun", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    r = load_register_module(args.register_script)
    out_root = ensure_dir(args.output_root).resolve()
    ensure_dir(out_root / "runs")
    ensure_dir(out_root / "pair_library")

    budgets = parse_int_list(args.gene_budgets_per_side)
    g_values = parse_int_list(args.genes_per_pole_values)
    tree_workers = None if int(args.tree_workers) <= 0 else int(args.tree_workers)
    max_pairs_needed = max(
        [int(b // g) for b in budgets for g in g_values if (b % g == 0 or args.allow_floor_budget) and int(b // g) > 0],
        default=0,
    )
    if max_pairs_needed <= 0:
        raise ValueError("No valid budget/regime combinations")
    library_pair_count = int(args.library_pair_count or (max_pairs_needed + int(args.heldout_pairs)))

    all_metric_rows: list[dict[str, Any]] = []
    all_run_rows: list[dict[str, Any]] = []

    for spec in read_manifest(args.manifest):
        print(f"[benchmark] loading {spec.dataset_id}", flush=True)
        ctx = load_context(r, spec, min_feature_count=int(args.min_feature_count))
        pair_library = build_or_load_pair_library(
            r,
            ctx,
            out_dir=out_root / "pair_library" / spec.dataset_id,
            library_pair_count=library_pair_count,
            max_genes_per_pole=int(args.max_genes_per_pole_library),
            abundance_threshold=int(args.abundance_threshold),
        )
        for fold in range(int(args.folds)):
            train_pool, heldout_idx = split_indices(len(pair_library), fold, int(args.seed), int(args.heldout_pairs), max_pairs_needed)
            heldout_pairs = [truncate_pair(pair_library[int(i)], int(args.eval_genes_per_pole), "heldout") for i in heldout_idx]
            for budget in budgets:
                for g in g_values:
                    if budget % g != 0 and not args.allow_floor_budget:
                        continue
                    num_train_pairs = int(budget // g)
                    if num_train_pairs <= 0 or num_train_pairs > len(train_pool):
                        continue
                    train_pairs = [truncate_pair(pair_library[int(i)], int(g), f"g{g}") for i in train_pool[:num_train_pairs]]
                    validate_pair_disjointness(train_pairs, heldout_pairs)

                    run_id = f"fold{fold:02d}_budget{budget:03d}_g{g:02d}_p{num_train_pairs:03d}"
                    run_dir = out_root / "runs" / spec.dataset_id / run_id
                    signature = make_run_signature(args, spec, fold=fold, run_id=run_id, budget=budget, genes_per_pole=g, num_train_pairs=num_train_pairs, train_pairs=train_pairs, heldout_pairs=heldout_pairs)
                    common = {
                        "dataset_id": spec.dataset_id,
                        "fold": int(fold),
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "gene_budget_per_side": int(num_train_pairs * g),
                        "requested_gene_budget_per_side": int(budget),
                        "genes_per_pole": int(g),
                        "num_train_pairs": int(num_train_pairs),
                        "eval_genes_per_pole": int(args.eval_genes_per_pole),
                        "n_train_genes_total": int(2 * num_train_pairs * g),
                        "n_heldout_pairs": int(len(heldout_pairs)),
                        "run_signature": signature["signature"],
                    }
                    if not bool(args.force_rerun):
                        cached = load_completed_run(run_dir, signature)
                        if cached is not None:
                            rows, run_row = cached
                            print(f"[benchmark] reuse {spec.dataset_id}/{run_id}", flush=True)
                            all_metric_rows.extend(rows)
                            all_run_rows.append(run_row)
                            continue
                    if run_dir.exists():
                        shutil.rmtree(run_dir)
                    ensure_dir(run_dir)
                    print(f"[benchmark] fit {spec.dataset_id}/{run_id}: {num_train_pairs} pairs × {g} genes/pole", flush=True)
                    fit = fit_transport_for_pairs(r, ctx, train_pairs, out_dir=run_dir, args=args, tree_workers=tree_workers)
                    run_metric_rows = add_run_columns(fit["train_metric_rows"], **common)
                    heldout_summaries = []
                    for heldout_pair in heldout_pairs:
                        summary, rows = evaluate_heldout_pair(r, ctx, heldout_pair, out_dir=run_dir, fit=fit, args=args, tree_workers=tree_workers)
                        heldout_summaries.append(summary)
                        run_metric_rows.extend(add_run_columns(rows, **common))
                    write_json(run_dir / "heldout_evaluation_manifest.json", heldout_summaries)
                    mr = fit["match_result"]
                    run_row = dict(common)
                    run_row.update({
                        "effective_row_k_max": mr.get("effective_row_k_max"),
                        "effective_edge_count": mr.get("effective_edge_count"),
                        "objective_initial": mr.get("objective_initial"),
                        "objective_final": mr.get("objective_final"),
                        "fraction_reassigned_by_graph_refinement": mr.get("fraction_reassigned_by_graph_refinement"),
                        "median_move_distance_normalized": mr.get("median_move_distance_normalized"),
                    })
                    run_row = finalize_run(run_dir, signature=signature, heldout_pairs=heldout_pairs, metric_rows=run_metric_rows, run_row=run_row)
                    all_metric_rows.extend(run_metric_rows)
                    all_run_rows.append(run_row)
                    write_csv(out_root / "all_pair_metrics.csv", all_metric_rows)
                    write_csv(out_root / "all_runs.csv", all_run_rows)

    write_csv(out_root / "all_pair_metrics.csv", all_metric_rows)
    write_csv(out_root / "all_runs.csv", all_run_rows)
    make_summaries(out_root / "all_pair_metrics.csv", out_root)
    print(f"[benchmark] done: {out_root / 'all_pair_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()

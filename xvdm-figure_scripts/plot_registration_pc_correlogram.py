#!/usr/bin/env python3
"""
Create the benchmark registration PC correlogram.

This is the only plotting through-line retained from the previous landmark
script: it reads benchmark run bundles, derives target-defined PCs from each
run's train+heldout post-rank target fields, projects target and predicted
fields into that basis, and writes a single PDF plus its records/metadata.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1.0e-10


def apply_publication_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "savefig.dpi": 300,
    })


def read_json(path: Path) -> Any | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None


def to_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, str) and x.strip().lower() in {"", "none", "nan", "null"}:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def infer_sample_id(path: Path) -> str:
    parts = list(path.parts)
    if "runs" in parts:
        i = parts.index("runs")
        if i + 1 < len(parts):
            return parts[i + 1]
    return path.parent.name


def rank_rescale_over_mask(values: np.ndarray, mask: np.ndarray, *, neutral: float) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    valid = np.asarray(mask, dtype=bool).ravel() & np.isfinite(values)
    out = np.full(values.shape, float(neutral), dtype=float)
    if int(valid.sum()) < 2:
        return out
    ranks = stats.rankdata(values[valid], method="average").astype(float)
    lo, hi = float(np.min(ranks)), float(np.max(ranks))
    out[valid] = (ranks - lo) / (hi - lo) if hi > lo else float(neutral)
    return out


@dataclass
class ProjectionBundle:
    run_dir: Path
    sample_id: str
    condition_label: str
    gene_budget_per_side: int | None
    genes_per_pole: int | None
    num_train_pairs: int | None
    train_target: np.ndarray
    train_base: np.ndarray
    train_final: np.ndarray
    heldout_target: np.ndarray
    heldout_base: np.ndarray
    heldout_final: np.ndarray


def rint(row: dict[str, Any] | None, key: str) -> int | None:
    if row is None:
        return None
    val = to_float(row.get(key))
    return int(round(val)) if np.isfinite(val) else None


def condition_label(row: dict[str, Any] | None) -> str:
    if row is None:
        return "condition unknown"
    b = rint(row, "gene_budget_per_side")
    g = rint(row, "genes_per_pole")
    p = rint(row, "num_train_pairs")
    return f"budget={b if b is not None else '?'}; genes/pole={g if g is not None else '?'}; pairs={p if p is not None else '?'}"


def resolve_run_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    p = Path(value).expanduser()
    if p.exists():
        return p.resolve()
    p2 = (root / p).resolve()
    return p2 if p2.exists() else None


def resolve_artifact_path(run_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    p = Path(value).expanduser()
    if p.exists():
        return p.resolve()
    p2 = (run_dir / p).resolve()
    return p2 if p2.exists() else None


def postrank_matrix(slice_feature_01: np.ndarray, slice_support: np.ndarray, feature_mean_base: np.ndarray, feature_mean_final: np.ndarray, *, neutral: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(slice_feature_01, dtype=float)
    support = np.asarray(slice_support, dtype=float)
    base = np.asarray(feature_mean_base, dtype=float)
    final = np.asarray(feature_mean_final, dtype=float)
    if target.ndim == 1:
        target = target[:, None]
    if support.ndim == 1:
        support = support[:, None]
    if base.ndim == 1:
        base = base[:, None]
    if final.ndim == 1:
        final = final[:, None]

    tcols, bcols, fcols = [], [], []
    for j in range(target.shape[1]):
        raw_t = target[:, j]
        raw_b = base[:, j]
        raw_f = final[:, j]
        supp = support[:, j] if j < support.shape[1] else np.ones_like(raw_t)
        finite_slice = np.isfinite(raw_t)
        finite_common = finite_slice & np.isfinite(raw_b) & np.isfinite(raw_f)
        eval_mask = finite_common & (supp > EPS)
        if int(eval_mask.sum()) < 2:
            eval_mask = finite_common
        if int(eval_mask.sum()) < 2:
            eval_mask = finite_slice
        non_eval = finite_slice & (~eval_mask)
        tv = rank_rescale_over_mask(raw_t, eval_mask, neutral=neutral)
        bv = rank_rescale_over_mask(raw_b, eval_mask, neutral=neutral)
        fv = rank_rescale_over_mask(raw_f, eval_mask, neutral=neutral)
        tv[non_eval] = float(neutral)
        bv[non_eval] = float(neutral)
        fv[non_eval] = float(neutral)
        tcols.append(tv)
        bcols.append(bv)
        fcols.append(fv)
    return np.column_stack(tcols), np.column_stack(bcols), np.column_stack(fcols)


def load_run_arrays(run_dir: Path, *, neutral: float) -> tuple[np.ndarray, ...] | None:
    train_slice_path = run_dir / "slice_smoothed_ratio_fields.npz"
    train_maps_path = run_dir / "slice_assigned_aggregated_feature_maps.npz"
    manifest_path = run_dir / "heldout_evaluation_manifest.json"
    if not (train_slice_path.exists() and train_maps_path.exists() and manifest_path.exists()):
        return None
    try:
        train_slice = np.load(train_slice_path, allow_pickle=True)
        train_maps = np.load(train_maps_path, allow_pickle=True)
        train_target, train_base, train_final = postrank_matrix(
            train_slice["ratio_feature_01"],
            train_slice["support"] if "support" in train_slice.files else np.ones_like(train_slice["ratio_feature_01"]),
            train_maps["feature_mean_base"],
            train_maps["feature_mean_final"],
            neutral=neutral,
        )
    except Exception as exc:
        warnings.warn(f"Could not load train arrays for {run_dir}: {exc}")
        return None

    manifest = read_json(manifest_path)
    if not isinstance(manifest, list) or not manifest:
        return None
    held_t, held_b, held_f = [], [], []
    for item in manifest:
        if not isinstance(item, dict):
            continue
        spath = resolve_artifact_path(run_dir, item.get("slice_output_npz"))
        mpath = resolve_artifact_path(run_dir, item.get("slice_assigned_feature_maps_npz"))
        if spath is None or mpath is None:
            continue
        try:
            hs = np.load(spath, allow_pickle=True)
            hm = np.load(mpath, allow_pickle=True)
            t, b, f = postrank_matrix(
                hs["ratio_feature_01"],
                hs["support"] if "support" in hs.files else np.ones_like(hs["ratio_feature_01"]),
                hm["feature_mean_base"],
                hm["feature_mean_final"],
                neutral=neutral,
            )
            held_t.append(t)
            held_b.append(b)
            held_f.append(f)
        except Exception as exc:
            warnings.warn(f"Could not load heldout arrays under {run_dir}: {exc}")
    if not held_t:
        return None
    return train_target, train_base, train_final, np.concatenate(held_t, axis=1), np.concatenate(held_b, axis=1), np.concatenate(held_f, axis=1)


def collect_bundles(roots: list[Path], *, neutral: float) -> list[ProjectionBundle]:
    bundles: list[ProjectionBundle] = []
    seen: set[Path] = set()

    def add_run(root: Path, run_dir: Path, row: dict[str, Any] | None) -> None:
        run_dir = run_dir.resolve()
        if run_dir in seen:
            return
        seen.add(run_dir)
        arrays = load_run_arrays(run_dir, neutral=neutral)
        if arrays is None:
            return
        train_t, train_b, train_f, held_t, held_b, held_f = arrays
        bundles.append(ProjectionBundle(
            run_dir=run_dir,
            sample_id=str(row.get("dataset_id")) if row and row.get("dataset_id") is not None else infer_sample_id(run_dir),
            condition_label=condition_label(row),
            gene_budget_per_side=rint(row, "gene_budget_per_side"),
            genes_per_pole=rint(row, "genes_per_pole"),
            num_train_pairs=rint(row, "num_train_pairs"),
            train_target=train_t,
            train_base=train_b,
            train_final=train_f,
            heldout_target=held_t,
            heldout_base=held_b,
            heldout_final=held_f,
        ))

    for root in roots:
        all_runs = root / "all_runs.csv"
        if all_runs.exists():
            try:
                df = pd.read_csv(all_runs)
                for _, row in df.iterrows():
                    p = resolve_run_path(root, row.to_dict().get("run_dir"))
                    if p is not None:
                        add_run(root, p, row.to_dict())
            except Exception as exc:
                warnings.warn(f"Could not read {all_runs}: {exc}")

    for root in roots:
        for manifest in root.rglob("heldout_evaluation_manifest.json"):
            add_run(root, manifest.parent, None)
    return bundles


def pc_projection_records(bundles: list[ProjectionBundle], *, n_pcs: int, neutral: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for b in bundles:
        target_all = np.concatenate([b.train_target, b.heldout_target], axis=1)
        if target_all.shape[1] < 2:
            continue
        X = np.nan_to_num(target_all - float(neutral), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            U, s, _ = np.linalg.svd(X, full_matrices=False)
        except Exception as exc:
            warnings.warn(f"SVD failed for {b.run_dir}: {exc}")
            continue
        npc = int(min(n_pcs, U.shape[1], target_all.shape[1]))
        if npc <= 0:
            continue
        basis = U[:, :npc]
        target_scores_all = basis.T @ X
        scale = np.std(target_scores_all, axis=1, ddof=1) if target_scores_all.shape[1] > 1 else np.std(target_scores_all, axis=1)
        scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
        explained = (s[:npc] ** 2) / max(float(np.sum(s ** 2)), EPS)
        scenarios = {
            "train_base": (b.train_target, b.train_base, "train", "base"),
            "train_final": (b.train_target, b.train_final, "train", "final"),
            "heldout_base": (b.heldout_target, b.heldout_base, "heldout", "base"),
            "heldout_final": (b.heldout_target, b.heldout_final, "heldout", "final"),
        }
        for _, (target, pred, split, stage) in scenarios.items():
            Ts = basis.T @ (np.asarray(target, dtype=float) - float(neutral))
            Ps = basis.T @ (np.asarray(pred, dtype=float) - float(neutral))
            for pc in range(npc):
                for j in range(Ts.shape[1]):
                    rows.append({
                        "run_dir": str(b.run_dir),
                        "sample_id": b.sample_id,
                        "condition_label": b.condition_label,
                        "gene_budget_per_side": b.gene_budget_per_side,
                        "genes_per_pole": b.genes_per_pole,
                        "num_train_pairs": b.num_train_pairs,
                        "split": split,
                        "stage": stage,
                        "pc": int(pc + 1),
                        "target_pc_score": float(Ts[pc, j] / scale[pc]),
                        "pred_pc_score": float(Ps[pc, j] / scale[pc]),
                        "explained_variance_ratio": float(explained[pc]),
                    })
    return pd.DataFrame.from_records(rows)


def plot_correlogram(df: pd.DataFrame, out_pdf: Path) -> None:
    import matplotlib.pyplot as plt

    scenarios = [
        ("train", "base", "Train: base"),
        ("train", "final", "Train: final"),
        ("heldout", "base", "Heldout: base"),
        ("heldout", "final", "Heldout: final"),
    ]
    pcs = sorted(int(x) for x in df["pc"].dropna().unique())
    if not pcs:
        raise ValueError("No PC records available to plot")
    conditions = sorted(df["condition_label"].dropna().unique())
    samples = sorted(df["sample_id"].dropna().unique())
    color_map = {k: f"C{i % 10}" for i, k in enumerate(conditions)}
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    marker_map = {k: markers[i % len(markers)] for i, k in enumerate(samples)}

    fig, axes = plt.subplots(len(pcs), len(scenarios), figsize=(4.2 * len(scenarios), 3.7 * len(pcs)), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(pcs), len(scenarios))
    all_scores = pd.concat([df["target_pc_score"], df["pred_pc_score"]], ignore_index=True)
    lim = float(np.nanquantile(np.abs(all_scores), 0.98)) if len(all_scores) else 1.0
    lim = lim * 1.05 if np.isfinite(lim) and lim > 0 else 1.0

    for i, pc in enumerate(pcs):
        for j, (split, stage, title) in enumerate(scenarios):
            ax = axes[i, j]
            sub = df[(df["pc"] == pc) & (df["split"] == split) & (df["stage"] == stage)]
            if sub.empty:
                ax.axis("off")
                continue
            for (cond, sample), g in sub.groupby(["condition_label", "sample_id"], dropna=False):
                ax.scatter(g["target_pc_score"], g["pred_pc_score"], s=24, alpha=0.62, c=color_map.get(cond, "C0"), marker=marker_map.get(sample, "o"))
            ax.plot([-lim, lim], [-lim, lim], linestyle="--", linewidth=1.0, color="0.35")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            if i == len(pcs) - 1:
                ax.set_xlabel("Target PC score")
            if j == 0:
                ax.set_ylabel(f"Predicted PC{pc} score")
            if i == 0:
                ax.set_title(title)
            rho = sub[["target_pc_score", "pred_pc_score"]].corr(method="spearman").iloc[0, 1] if len(sub) >= 2 else np.nan
            exp = float(np.nanmedian(sub["explained_variance_ratio"]))
            label = f"n={len(sub)}"
            if np.isfinite(rho):
                label += f"\nρ={rho:.2f}"
            if np.isfinite(exp):
                label += f"\nmed var={exp:.2f}"
            ax.text(0.03, 0.97, label, transform=ax.transAxes, ha="left", va="top", fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "0.7", "linewidth": 0.4, "alpha": 0.9, "pad": 2.5})

    cond_handles = [plt.Line2D([], [], linestyle="", marker="o", color=color_map[c], markersize=7, label=c) for c in conditions]
    sample_handles = [plt.Line2D([], [], linestyle="", marker=marker_map[s], color="0.3", markersize=7, label=s) for s in samples]
    if cond_handles:
        leg = fig.legend(handles=cond_handles, loc="outside right upper", title="Condition")
        fig.add_artist(leg)
    if sample_handles:
        fig.legend(handles=sample_handles, loc="outside right lower", title="Dataset")
    fig.suptitle("Registration PC correlogram\nTarget-defined PCs from train+heldout post-rank fields; predictions are projected into the same basis.")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_roots(values: list[str]) -> list[Path]:
    roots: list[Path] = []
    for value in values:
        for part in str(value).split(","):
            if part.strip():
                roots.append(Path(part).expanduser().resolve())
    return roots


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Write registration_pc_correlogram.pdf from benchmark outputs")
    p.add_argument("--benchmark-roots", nargs="+", required=True, help="Benchmark output root(s) from benchmark_holdout_tradeoffs_clean.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-pcs", type=int, default=3)
    p.add_argument("--rank-neutral", type=float, default=0.5)
    return p


def main() -> None:
    args = build_parser().parse_args()
    apply_publication_style()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = parse_roots(args.benchmark_roots)
    bundles = collect_bundles(roots, neutral=float(args.rank_neutral))
    df = pc_projection_records(bundles, n_pcs=int(args.n_pcs), neutral=float(args.rank_neutral))
    records_path = out_dir / "registration_pc_projection_records.csv"
    df.to_csv(records_path, index=False)
    if df.empty:
        raise SystemExit("No benchmark projection records were found; check --benchmark-roots")
    pdf_path = out_dir / "registration_pc_correlogram.pdf"
    plot_correlogram(df, pdf_path)
    meta = {
        "benchmark_roots": [str(x) for x in roots],
        "n_pcs": int(args.n_pcs),
        "rank_neutral": float(args.rank_neutral),
        "n_runs": int(len({str(x) for x in df["run_dir"].unique()})),
        "n_records": int(len(df)),
        "records_csv": str(records_path),
        "pdf": str(pdf_path),
    }
    with open(out_dir / "registration_pc_correlogram.meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()

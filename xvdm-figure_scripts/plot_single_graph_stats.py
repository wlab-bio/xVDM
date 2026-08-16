#!/usr/bin/env python3
"""Single-graph statistics for completed optimOps.run_GSE() outputs; no 3D rendering."""
from __future__ import annotations
import argparse, json, os, re, warnings
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
from scipy.optimize import minimize
from scipy.sparse import issparse, load_npz
from scipy.special import kl_div

EPS = 1e-10
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
})


def _safe_name(p: str) -> str:
    s = Path(p).resolve().name or "graph"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _gse_sort_key(c):
    try: return (0, int(str(c).split("GSE_", 1)[1]))
    except Exception: return (1, str(c))


def coords_from_h5ad(path: Path, dims: int) -> np.ndarray:
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "obsm" in f and "X_gse" in f["obsm"] and hasattr(f["obsm/X_gse"], "shape"):
                x = f["obsm/X_gse"][:, :dims]
                if x.shape[1] < dims: raise ValueError(f"X_gse has only {x.shape[1]} dims")
                return np.asarray(x, dtype=np.float64)
    except Exception:
        pass

    import anndata as ad
    a = ad.read_h5ad(path, backed="r")
    try:
        if "X_gse" in a.obsm:
            return np.asarray(a.obsm["X_gse"][:, :dims], dtype=np.float64)
        cols = sorted([c for c in a.obs.columns if str(c).startswith("GSE_")], key=_gse_sort_key)[:dims]
        if len(cols) < dims: raise KeyError("final.h5ad lacks obsm['X_gse'] or enough obs GSE_* columns")
        return a.obs[cols].to_numpy(dtype=np.float64)
    finally:
        try: a.file.close()
        except Exception: pass


def load_coords(root: Path, dims: int) -> tuple[np.ndarray, str]:
    for fn in ("final.h5ad", "GSEoutput.txt", "Xumi_GSE.txt"):
        p = root / fn
        if not p.exists(): continue
        if p.suffix == ".h5ad": return coords_from_h5ad(p, dims), str(p)
        x = np.loadtxt(p, delimiter=",", dtype=np.float64)
        if x.ndim == 1: x = x[None, :]
        if x.shape[1] < dims + 1: raise ValueError(f"{p} has {x.shape[1]-1} coord dims; need {dims}")
        return x[:, 1:1 + dims], str(p)
    raise FileNotFoundError(f"{root} lacks final.h5ad, GSEoutput.txt, and Xumi_GSE.txt")


def resolve_graph(p: str) -> tuple[Path, Path]:
    p = Path(p)
    root = p.parent if p.is_file() else p
    for fn in ("link_assoc_reindexed.npz", "link_assoc_reindexed.npy"):
        q = root / fn
        if q.exists(): return root, q
    raise FileNotFoundError(f"{root} lacks link_assoc_reindexed.npz/.npy")


def mb(x, L):
    x = np.maximum(np.asarray(x, dtype=np.float64), EPS)
    s = L / np.sqrt(2.0)
    return x * x * np.exp(-(x * x) / (2.0 * s * s))


def mb2(x, A1, L1, A2, L2): return A1 * mb(x, L1) + A2 * mb(x, L2)


def fit_mb2(x, y):
    def loss(p):
        z = mb2(x, *p)
        return float(np.sum(kl_div(y / (y.sum() + EPS) + EPS, z / (z.sum() + EPS) + EPS)))
    r = minimize(loss, [0.5, 0.5, 0.5, 1.5], method="L-BFGS-B",
                 bounds=((0, 1), (0.01, 100), (0, 1), (0.01, 100)))
    if not r.success: return None, None
    z = mb2(x, *r.x); z = z / (z.sum() + EPS) * y.sum()
    return z, np.asarray(r.x, dtype=np.float64)


def csr_chunks(A, target_edges: int):
    n, r0 = A.shape[0], 0
    while r0 < n:
        p0 = A.indptr[r0]
        r1 = np.searchsorted(A.indptr, min(p0 + target_edges, A.nnz), side="right") - 1
        r1 = min(n, max(r0 + 1, int(r1)))
        yield r0, r1
        r0 = r1


def sparse_stats(A, coords, max_dist=3.0, nbins=100, chunk_edges=2_000_000):
    A = A.tocsr(); A.sum_duplicates(); A.eliminate_zeros()
    n = A.shape[0]
    if A.shape[1] != n or coords.shape[0] != n: raise ValueError(f"graph {A.shape}, coords {coords.shape}")
    wsum = np.asarray(A.sum(axis=1)).ravel() + np.asarray(A.sum(axis=0)).ravel()
    ccnt = np.diff(A.indptr).astype(np.float64) + np.bincount(A.indices, minlength=n).astype(np.float64)
    msd_num = np.zeros(n, dtype=np.float64)
    bins, hist = np.linspace(0, max_dist, nbins), np.zeros(nbins - 1, dtype=np.float64)
    beyond = total = 0.0
    for r0, r1 in csr_chunks(A, chunk_edges):
        p0, p1 = A.indptr[r0], A.indptr[r1]
        if p1 == p0: continue
        rows = np.repeat(np.arange(r0, r1, dtype=np.int64), np.diff(A.indptr[r0:r1 + 1]))
        cols = A.indices[p0:p1].astype(np.int64, copy=False)
        w = A.data[p0:p1].astype(np.float64, copy=False)
        d = coords[rows] - coords[cols]
        d2 = np.einsum("ij,ij->i", d, d, optimize=True)
        dist = np.sqrt(d2)
        hist += np.histogram(dist, bins=bins, weights=w)[0]
        m = dist >= max_dist
        beyond += float(w[m].sum()); total += float(w.sum())
        wd2 = w * d2
        np.add.at(msd_num, rows, wd2); np.add.at(msd_num, cols, wd2)
    rmsd = np.sqrt(msd_num / (wsum + EPS))
    return wsum, ccnt, rmsd, bins, hist, beyond, beyond / (total + EPS)


def dense_edge_stats(E, coords, max_dist=3.0, nbins=100, chunk_edges=2_000_000):
    n = coords.shape[0]
    wsum = np.zeros(n); ccnt = np.zeros(n); msd_num = np.zeros(n)
    bins, hist = np.linspace(0, max_dist, nbins), np.zeros(nbins - 1)
    beyond = total = 0.0
    for s in range(0, E.shape[0], chunk_edges):
        e = min(E.shape[0], s + chunk_edges)
        rows, cols = E[s:e, 0].astype(np.int64), E[s:e, 1].astype(np.int64)
        w = E[s:e, 2].astype(np.float64)
        d = coords[rows] - coords[cols]
        d2 = np.einsum("ij,ij->i", d, d, optimize=True); dist = np.sqrt(d2)
        hist += np.histogram(dist, bins=bins, weights=w)[0]
        m = dist >= max_dist
        beyond += float(w[m].sum()); total += float(w.sum())
        np.add.at(wsum, rows, w); np.add.at(wsum, cols, w)
        np.add.at(ccnt, rows, 1); np.add.at(ccnt, cols, 1)
        wd2 = w * d2; np.add.at(msd_num, rows, wd2); np.add.at(msd_num, cols, wd2)
    return wsum, ccnt, np.sqrt(msd_num / (wsum + EPS)), bins, hist, beyond, beyond / (total + EPS)


def plot_hist(out_npy, bins, hist, beyond, frac, fit=True):
    bc = (bins[:-1] + bins[1:]) / 2
    kernel = np.exp(-bc * bc / 2.0); kernel = kernel / (kernel.sum() + EPS) * hist.sum()
    yfit = pars = None
    if fit:
        try: yfit, pars = fit_mb2(bc, hist)
        except Exception as e: warnings.warn(f"Could not fit two Maxwell-Boltzmann distributions: {e}")
        if pars is not None:
            np.savetxt(out_npy.with_name(out_npy.stem + "_maxwell_boltzmann_fit.csv"),
                       np.c_[['A1','L1','A2','L2'], pars.astype(str)], fmt="%s", delimiter=",",
                       header="Parameter,Estimate", comments="")
    fig = plt.figure(figsize=(8, 4.8))
    gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.05)
    ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1])
    ax1.bar(bc, hist, width=bins[1]-bins[0], alpha=0.7, color="skyblue", label="Observed")
    ax1.plot(bc, kernel, "r-", lw=2, label="Gaussian Kernel")
    if yfit is not None and pars is not None:
        ax1.plot(bc, yfit, color="green", ls="--", lw=2, label=f"2 M-B (L1={pars[1]:.2f}, L2={pars[3]:.2f})")
    ax2.bar(0, beyond, width=1, alpha=0.7, color="salmon", label="≥ 3.0")
    ax1.set_xlabel("Distance (diff units)", fontsize=25); ax1.set_ylabel("UEIs", fontsize=25)
    ax1.set_xlim(0, 3.0); ax1.set_xticks(np.arange(0, 4, 1)); ax1.set_xticklabels(["0", "1.0", "2.0", "3.0"])
    ax1.tick_params(axis="both", which="major", labelsize=25); ax1.legend(fontsize=18, loc="upper right")
    ax1.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True)); ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax1.yaxis.get_offset_text().set_fontsize(20)
    ax2.set_xlim(0, 1); ax2.set_ylim(ax1.get_ylim()); ax2.set_xticks([]); ax2.set_yticks([]); ax2.spines["left"].set_visible(False)
    ax2.text(0.5, beyond, f"≥ 3.0\n{frac:.2%}", ha="center", va="bottom", fontsize=18)
    for ax in (ax1, ax2): ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    d = 0.015; kw = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1-d,1+d),(-d,+d), **kw); ax1.plot((1-d,1+d),(1-d,1+d), **kw)
    kw.update(transform=ax2.transAxes); ax2.plot((-d,+d),(-d,+d), **kw); ax2.plot((-d,+d),(1-d,1+d), **kw)
    fig.set_tight_layout(True)
    fig.savefig(out_npy.with_name(out_npy.stem + "_distance_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def thin(r, a, b, maxpts):
    if maxpts <= 0 or r.size <= maxpts: return r, a, b
    idx = np.unique(np.geomspace(1, r.size, maxpts).astype(np.int64) - 1)
    return r[idx], a[idx], b[idx]


def plot_rank(out_npy, wsum, rmsd, max_rank_points=250_000):
    inv = 1.0 / (EPS + rmsd * rmsd)
    rw, ri = np.sort(wsum)[::-1], np.sort(inv)[::-1]
    ranks = np.arange(1, rw.size + 1)
    rp, rwp, rip = thin(ranks, rw, ri, int(max_rank_points))
    fig, ax1 = plt.subplots(figsize=(8, 4))
    l1 = ax1.plot(rp, rwp, "b-", lw=1.5, label=f"UEI count (median = {np.median(wsum):.1f})")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("UMI rank", fontsize=25); ax1.set_ylabel("UEI count", fontsize=25, color="b")
    ax1.tick_params(axis="y", labelcolor="b", labelsize=25); ax1.tick_params(axis="x", labelsize=25)
    ax2 = ax1.twinx()
    l2 = ax2.plot(rp, rip, "r-", lw=1.5, label=f"1/MSD (median = {np.median(inv):.1e})")
    ax2.set_yscale("log"); ax2.set_ylabel("1/MSD", fontsize=25, color="r"); ax2.tick_params(axis="y", labelcolor="r", labelsize=25)
    ax1.grid(True, which="both", ls="--", lw=0.5)
    lines = l1 + l2; ax1.legend(lines, [x.get_label() for x in lines], loc="lower left", fontsize=16)
    fig.tight_layout(); fig.savefig(out_npy.with_name(out_npy.stem + "_rank_order_plot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_stats(out_npy, coords, wsum, ccnt, rmsd, save=True):
    if not save: return None
    d = coords.shape[1]
    mm = open_memmap(out_npy, mode="w+", dtype=np.float64, shape=(coords.shape[0], d + 3))
    mm[:, :d] = coords; mm[:, d] = wsum; mm[:, d + 1] = ccnt; mm[:, d + 2] = rmsd; mm.flush()
    return [*(f"GSE_{i+1}" for i in range(d)), "uei_weight_sum", "connection_count", "rmsd_spread"]


def process(inp, out, args):
    out.mkdir(parents=True, exist_ok=True)
    root, link = resolve_graph(inp)
    coords, coord_src = load_coords(root, args.gse_dims)
    if link.suffix == ".npz":
        A = load_npz(link)
        stats = sparse_stats(A, coords, args.max_distance, args.bins, args.chunk_edges)
        nnz = int(A.nnz)
    else:
        E = np.load(link, mmap_mode="r")
        stats = dense_edge_stats(E, coords, args.max_distance, args.bins, args.chunk_edges); nnz = int(E.shape[0])
    wsum, ccnt, rmsd, bins, hist, beyond, frac = stats
    out_npy = out / "coords_and_stats.npy"
    cols = write_stats(out_npy, coords, wsum, ccnt, rmsd, save=not args.no_array)
    plot_hist(out_npy, bins, hist, beyond, frac, fit=not args.no_fit)
    plot_rank(out_npy, wsum, rmsd, args.max_rank_points)
    meta = dict(graph_input=str(inp), root=str(root), link_assoc=str(link), coord_source=coord_src,
                n_nodes=int(coords.shape[0]), n_edges=nnz, gse_dims=int(args.gse_dims),
                max_distance=float(args.max_distance), distance_fraction_beyond_max=float(frac),
                output_columns=cols, skipped_3d_rendering=True)
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Reproduce plot_psf.py single-graph statistics figures for completed run_GSE directories, excluding 3D plots.")
    ap.add_argument("graph_inputs", nargs="+", help="run_GSE directory, final.h5ad, or component0 directory")
    ap.add_argument("-o", "--output-dir", required=True)
    ap.add_argument("--gse-dims", type=int, default=3)
    ap.add_argument("--chunk-edges", type=int, default=2_000_000)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--max-distance", type=float, default=3.0)
    ap.add_argument("--max-rank-points", type=int, default=250_000, help="0 plots every sorted rank point")
    ap.add_argument("--no-fit", action="store_true", help="omit the optional two-Maxwell-Boltzmann overlay/CSV")
    ap.add_argument("--no-array", action="store_true", help="do not write coords_and_stats.npy")
    args = ap.parse_args()
    base = Path(args.output_dir)
    multi = len(args.graph_inputs) > 1
    for inp in args.graph_inputs:
        process(inp, base / _safe_name(inp) if multi else base, args)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Global gene-level Fig. 4 rank-correlogram analysis.

This is a focused replacement for the YSL-only Fig. 4d/4e analysis.  It asks,
for a large gene set ranked by external 18 hpf Stereo-seq spatial autocorrelation,
whether two internal xVDM readouts track that ordering:

  1. Global compactness in inferred xVDM 3D coordinates.
  2. Global smoothness on the cell-coarsened raw UEI contact graph.

The analysis is deliberately gene-level and global within the matched 18 hpf xVDM
specimens.  It does not restrict to MC3 or any other metacluster.  This makes the
internal readouts comparable to external Moran's I, which is also a whole-slice
spatial autocorrelation statistic.

Typical usage
-------------
python fig4_gene_rank_correlogram_analysis.py \
  --h5ad-dir /path/to/sample_cell_connectivity_h5ad \
  --zesta18 /path/to/spatial_sixtime_slice_stereoseq.h5ad \
  --samples zf3,zf4 \
  --outdir /path/to/fig4_rankcorr

Outputs
-------
  source_external_moran_all_genes.csv
  source_rankcorr_sample_metrics.csv
  source_rankcorr_gene_metrics.csv
  table_rankcorr_summary.csv
  panel4de_external_moran_rankcorr.pdf/png
  panel4d_external_moran_vs_xvdm_compactness.pdf/png
  panel4e_external_moran_vs_rawuei_smoothness.pdf/png
  panel4f_external_moran_vs_rawuei_to_xvdm_compactness_ratio.pdf/png
  qc_summary.json

Important interpretation
------------------------
The raw-UEI readout is computed on the cell-coarsened raw UEI contact graph.  It
is raw with respect to diffusion-transport smoothing and geometric inference, but
it is still coarsened from molecular hubs to aggregate cells by the same cell
assignment sidecar.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import pdist
from scipy.stats import norm, rankdata, spearmanr
from sklearn.neighbors import kneighbors_graph

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ad = None  # imported lazily so --help works when anndata is not yet loaded


STAGE_BY_SAMPLE = {
    "zf1": "12 hpf",
    "zf2": "12 hpf",
    "zf3": "18 high",
    "zf4": "18 high",
    "zf5": "18 low",
    "zf6": "18 low",
    "zf7": "24 hpf",
    "zf8": "24 hpf",
}

YSL_LIPID_GENES = [
    "apoa1a", "afp4", "tfa", "apoa1b", "apoba", "bhmt",
    "apoeb", "fabp1b.1", "fetub", "fabp2", "apoc1", "apoa2",
]


def require_anndata():
    global ad
    if ad is None:
        try:
            import anndata as _ad  # type: ignore
        except Exception as exc:
            raise SystemExit("Install anndata to read H5AD inputs: pip install anndata h5py") from exc
        ad = _ad
    return ad


def info(msg: str) -> None:
    print(str(msg), flush=True)


def warn(msg: str) -> None:
    warnings.warn(str(msg), stacklevel=2)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [t.strip() for t in re.split(r"[,; ]+", str(value)) if t.strip()]


def clean_gene_name(x: object) -> str:
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="ignore")
    return str(x)


def make_unique(names: Sequence[object]) -> pd.Index:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for raw in names:
        name = clean_gene_name(raw)
        if name not in seen:
            seen[name] = 0
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}__dup{seen[name]}")
    return pd.Index(out)


def as_csr(x) -> sp.csr_matrix:
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(np.asarray(x))


def normalize_log1p_counts(X: sp.csr_matrix, target_sum: float = 1e4) -> sp.csr_matrix:
    X = X.tocsr().astype(np.float32)
    totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    scale = np.zeros_like(totals, dtype=np.float32)
    ok = totals > 0
    scale[ok] = float(target_sum) / totals[ok]
    X = X.multiply(scale[:, None]).tocsr()
    X.data = np.log1p(X.data)
    X.eliminate_zeros()
    return X


def safe_logit_fraction(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return q
    pv = p[finite]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / (np.arange(n) + 1.0)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(n)
    out[order] = adj
    q[finite] = out
    return q


def parse_sample_id(path: Path, uns: Mapping[str, object]) -> str:
    for key in ["sample", "sample_id", "specimen", "specimen_id"]:
        if key in uns:
            val = uns[key]
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="ignore")
            m = re.search(r"zf[1-8]", str(val), flags=re.IGNORECASE)
            if m:
                return m.group(0).lower()
    m = re.search(r"zf[1-8]", path.name, flags=re.IGNORECASE)
    if m:
        return m.group(0).lower()
    return path.stem


def natural_key(x: object) -> Tuple[object, ...]:
    s = str(x)
    parts = re.split(r"(\d+)", s)
    out: List[object] = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        elif p:
            out.append(p)
    return tuple(out)


@dataclass
class SampleData:
    sample: str
    stage: str
    path: Path
    obs: pd.DataFrame
    genes: pd.Index
    X_log: sp.csr_matrix
    coords: np.ndarray
    raw_uei_w: Optional[sp.csr_matrix]

    @property
    def n_obs(self) -> int:
        return int(self.X_log.shape[0])


@dataclass
class ZestaData:
    genes: pd.Index
    X_log: sp.csr_matrix
    coords: np.ndarray
    obs: pd.DataFrame
    time_value: str

    @property
    def n_obs(self) -> int:
        return int(self.X_log.shape[0])


###############################################################################
# H5AD field discovery
###############################################################################


def find_gene_names(adata) -> pd.Index:
    candidates = [
        "cell_gene_counts_var_names",
        "cellgene_countsvar_names",
        "X_cell_gene_counts_var_names",
        "Xcell_genecounts_var_names",
        "gene_names",
        "genes",
    ]
    for key in candidates:
        if key in adata.uns:
            arr = list(adata.uns[key])
            if len(arr) > 0:
                return make_unique(arr)
    for key in adata.uns.keys():
        low = str(key).lower()
        if "gene" in low and ("var" in low or "name" in low):
            try:
                arr = list(adata.uns[key])
            except Exception:
                continue
            if len(arr) > 100:
                return make_unique(arr)
    raise KeyError("Could not find gene names in adata.uns")


def find_expression_matrix(adata, genes: pd.Index) -> sp.csr_matrix:
    candidates = [
        "X_cell_gene_counts", "Xcell_genecounts", "cell_gene_counts",
        "cellgene_counts", "X_cell_gene_count", "gene_counts",
    ]
    for key in candidates:
        if key in adata.obsm:
            mat = adata.obsm[key]
            if mat.shape == (adata.n_obs, len(genes)):
                return as_csr(mat)
    for key in adata.obsm.keys():
        mat = adata.obsm[key]
        if hasattr(mat, "shape") and mat.shape == (adata.n_obs, len(genes)):
            low = str(key).lower()
            if "gene" in low or "count" in low:
                return as_csr(mat)
    raise KeyError("Could not find cell-by-gene count matrix in adata.obsm")


def find_coords(adata) -> np.ndarray:
    candidates = [
        "X_cell_gse_mean", "Xcell_gsemean", "cell_gse_mean", "X_spatial",
        "spatial", "coords", "coordinates", "X_cell_coordinates",
        "X_diffusion_transport_scaled",
    ]
    for key in candidates:
        if key in adata.obsm:
            arr = np.asarray(adata.obsm[key])
            if arr.ndim >= 2 and arr.shape[0] == adata.n_obs and arr.shape[1] >= 2:
                return arr[:, : min(3, arr.shape[1])].astype(float)
    for key in adata.obsm.keys():
        arr = adata.obsm[key]
        if hasattr(arr, "shape") and arr.ndim >= 2 and arr.shape[0] == adata.n_obs and 2 <= arr.shape[1] <= 5:
            low = str(key).lower()
            if any(tok in low for tok in ["coord", "spatial", "gse", "xyz", "cell"]):
                return np.asarray(arr)[:, : min(3, arr.shape[1])].astype(float)
    raise KeyError("Could not find inferred coordinates in adata.obsm")


def gene_pos_index(genes: pd.Index) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, g in enumerate(genes):
        out[str(g)] = i
        out[str(g).lower()] = i
    return out


###############################################################################
# Cell-coarsened raw UEI graph recovery
###############################################################################


def clean_square_graph(G: sp.spmatrix, *, average_sym: bool = True) -> sp.csr_matrix:
    G = as_csr(G).astype(np.float32)
    G.setdiag(0)
    G.eliminate_zeros()
    if average_sym:
        G = ((G + G.T) * 0.5).tocsr()
    else:
        G = (G + G.T).tocsr()
    G.setdiag(0)
    G.eliminate_zeros()
    return G


def _graph_from_matrix_locations(a, logical_name: str) -> Optional[sp.csr_matrix]:
    raw = a.uns.get("connectivity_matrix_locations_json", None)
    if raw is None:
        return None
    try:
        locs = json.loads(str(raw))
    except Exception:
        return None
    loc = locs.get(logical_name)
    if not loc:
        return None
    loc = str(loc)
    if loc == "X":
        return clean_square_graph(a.X, average_sym=True)
    if loc.startswith("obsp/"):
        key = loc.split("/", 1)[1]
        if key in a.obsp:
            return clean_square_graph(a.obsp[key], average_sym=True)
    return None


def _coarsen_link_assoc_to_cells(a, h5ad_path: Path, cache_dir: Optional[Path]) -> Optional[sp.csr_matrix]:
    """Reconstruct a cell-coarsened raw UEI graph from link_assoc and hub labels."""
    conn_source = a.uns.get("connectivity_source", None)
    if conn_source is None:
        return None
    conn_path = Path(str(conn_source))
    if not conn_path.exists():
        alt = h5ad_path.parent / conn_path.name
        if alt.exists():
            conn_path = alt
        else:
            return None
    sidecar = conn_path.with_name("hub_refined_labels.tsv")
    if not sidecar.exists():
        alt = h5ad_path.parent / "hub_refined_labels.tsv"
        if alt.exists():
            sidecar = alt
        else:
            return None
    if cache_dir is not None:
        ensure_dir(cache_dir)
        cache_path = cache_dir / f"{h5ad_path.stem}.raw_uei_cell_graph.npz"
        if cache_path.exists():
            return clean_square_graph(sp.load_npz(cache_path), average_sym=True)
    else:
        cache_path = None

    side = pd.read_csv(sidecar, sep="\t")
    if "hub_original_index" not in side.columns or "refined_cell_label" not in side.columns:
        return None

    if "cell_node_label" in a.obs.columns:
        cell_labels = a.obs["cell_node_label"].astype(str).to_numpy()
    else:
        cell_labels = np.asarray(a.obs_names.astype(str), dtype=object)
    label_to_col = {str(label): i for i, label in enumerate(cell_labels)}

    lab = side["refined_cell_label"].astype(str)
    keep = lab.map(lambda x: x in label_to_col).to_numpy(bool)
    if keep.sum() == 0:
        return None
    hub_orig = side.loc[keep, "hub_original_index"].to_numpy(np.int64)
    cols = lab.loc[keep].map(label_to_col).to_numpy(np.int64)

    C_upper = sp.load_npz(conn_path).tocsr()
    in_bounds = (hub_orig >= 0) & (hub_orig < C_upper.shape[0])
    hub_orig = hub_orig[in_bounds]
    cols = cols[in_bounds]
    if hub_orig.size == 0:
        return None

    C_sub = C_upper[hub_orig, :][:, hub_orig].tocsr()
    C_sub = clean_square_graph(C_sub, average_sym=False)
    A = sp.csr_matrix(
        (np.ones(hub_orig.size, dtype=np.int8), (np.arange(hub_orig.size, dtype=np.int64), cols)),
        shape=(hub_orig.size, len(cell_labels)),
    )
    M = (A.T @ (C_sub @ A)).tocsr().astype(np.float32)
    M.setdiag(0)
    M.eliminate_zeros()
    if cache_path is not None:
        sp.save_npz(cache_path, M)
    return clean_square_graph(M, average_sym=True)


def find_raw_uei_cell_graph(a, h5ad_path: Path, cache_dir: Optional[Path], allow_fallback: bool = True) -> Optional[sp.csr_matrix]:
    """Return the cell-coarsened raw UEI graph, not the diffusion-transport graph."""
    G = _graph_from_matrix_locations(a, "uei_counts")
    if G is not None:
        return G
    for key in ["cell_connectivity_uei_counts", "cell_connectivity_raw_uei", "uei_counts", "raw_uei_counts"]:
        if key in a.obsp:
            return clean_square_graph(a.obsp[key], average_sym=True)
    primary = str(a.uns.get("connectivity_primary_graph", "")).lower()
    if primary in {"uei_counts", "raw", "raw_uei", "uei"}:
        return clean_square_graph(a.X, average_sym=True)
    if allow_fallback:
        return _coarsen_link_assoc_to_cells(a, h5ad_path, cache_dir)
    return None


def collect_h5ad_paths(h5ad_dir: Path, pattern: str) -> List[Path]:
    import glob
    if any(ch in pattern for ch in ["/", "\\"]):
        paths = sorted(Path(p) for p in glob.glob(pattern if os.path.isabs(pattern) else str(h5ad_dir / pattern)))
    else:
        paths = sorted(h5ad_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"No h5ad files found for {h5ad_dir / pattern}")
    return paths


def load_samples(paths: Sequence[Path], sample_keep: Sequence[str], raw_cache: Optional[Path], allow_link_assoc: bool) -> List[SampleData]:
    ad_mod = require_anndata()
    keep_set = {str(s) for s in sample_keep}
    out: List[SampleData] = []
    for path in paths:
        a = ad_mod.read_h5ad(path)
        sample = parse_sample_id(path, a.uns)
        if keep_set and sample not in keep_set:
            continue
        genes = find_gene_names(a)
        X_counts = find_expression_matrix(a, genes)
        X_log = normalize_log1p_counts(X_counts)
        coords = find_coords(a)
        raw_uei = find_raw_uei_cell_graph(a, path, raw_cache, allow_fallback=allow_link_assoc)
        if raw_uei is None:
            warn(f"{sample}: raw UEI cell graph unavailable; smoothness will be missing")
        obs = a.obs.copy()
        obs["sample"] = sample
        obs["stage"] = STAGE_BY_SAMPLE.get(sample, "unknown")
        sd = SampleData(sample, STAGE_BY_SAMPLE.get(sample, "unknown"), path, obs, genes, X_log, coords, raw_uei)
        info(f"  loaded {sample}: cells={sd.n_obs:,}, genes={len(genes):,}, raw_uei={'yes' if raw_uei is not None else 'no'}")
        out.append(sd)
    return sorted(out, key=lambda s: natural_key(s.sample))


###############################################################################
# External Moran's I
###############################################################################


def load_zesta_h5ad(path: Path, time_value: str, layer: str, spatial_x: str, spatial_y: str) -> ZestaData:
    ad_mod = require_anndata()
    z = ad_mod.read_h5ad(path)
    if "time" in z.obs.columns:
        mask = z.obs["time"].astype(str).values == str(time_value)
        if mask.sum() == 0:
            warn(f"No ZESTA spots with time={time_value!r}; using all spots")
            mask = np.ones(z.n_obs, dtype=bool)
        z = z[mask, :].copy()
    genes = make_unique(list(z.var_names))
    X_raw = as_csr(z.layers[layer]) if layer and layer in z.layers else as_csr(z.X)
    X_log = normalize_log1p_counts(X_raw)
    if spatial_x in z.obs.columns and spatial_y in z.obs.columns:
        coords = z.obs[[spatial_x, spatial_y]].to_numpy(float)
    elif "spatial" in z.obsm:
        coords = np.asarray(z.obsm["spatial"])[:, :2].astype(float)
    else:
        coords = find_coords(z)[:, :2]
    info(f"  loaded ZESTA {time_value}: spots={X_log.shape[0]:,}, genes={X_log.shape[1]:,}")
    return ZestaData(genes=genes, X_log=X_log, coords=coords, obs=z.obs.copy(), time_value=time_value)


def moran_i(x: np.ndarray, W: sp.csr_matrix) -> float:
    x = np.asarray(x, dtype=float).ravel()
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return np.nan
    x = x[ok]
    W2 = W[ok, :][:, ok].tocsr()
    z = x - x.mean()
    denom = float(np.sum(z * z))
    if denom <= 0:
        return np.nan
    s0 = float(W2.sum())
    if s0 <= 0:
        return np.nan
    return float((len(x) / s0) * ((z @ (W2 @ z)) / denom))


def sparse_mean_var(X: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(X.mean(axis=0)).ravel()
    mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return mean, var


def compute_external_moran_table(zesta: ZestaData, n_neighbors: int) -> pd.DataFrame:
    W = kneighbors_graph(zesta.coords, n_neighbors=min(int(n_neighbors), max(1, zesta.n_obs - 1)), mode="connectivity", include_self=False)
    W = W.maximum(W.T).tocsr()
    mean, var = sparse_mean_var(zesta.X_log)
    det = np.asarray((zesta.X_log > 0).sum(axis=0)).ravel() / max(1, zesta.n_obs)
    rows = []
    for j, gene in enumerate(zesta.genes):
        x = zesta.X_log[:, j].toarray().ravel()
        I = moran_i(x, W)
        rows.append({
            "gene": str(gene),
            "zesta_moran_I": I,
            "zesta_mean": float(mean[j]),
            "zesta_var": float(var[j]),
            "zesta_detection": float(det[j]),
            "zesta_time": zesta.time_value,
        })
    tbl = pd.DataFrame(rows)
    tbl["zesta_log_mean"] = np.log1p(tbl["zesta_mean"].astype(float))
    tbl["zesta_log_var"] = np.log1p(tbl["zesta_var"].astype(float))
    tbl["zesta_logit_detection"] = safe_logit_fraction(tbl["zesta_detection"].astype(float).to_numpy())
    return tbl.sort_values("zesta_moran_I", ascending=False)


###############################################################################
# Internal xVDM gene metrics
###############################################################################


def mean_pair_distance(coords: np.ndarray, rng: np.random.Generator, exact_n: int, sample_pairs: int) -> float:
    coords = np.asarray(coords, dtype=float)
    coords = coords[np.isfinite(coords).all(axis=1), :]
    n = coords.shape[0]
    if n < 2:
        return np.nan
    if n <= int(exact_n):
        return float(pdist(coords).mean())
    m = max(1, int(sample_pairs))
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n, size=m)
    ok = i != j
    if ok.sum() == 0:
        return np.nan
    return float(np.linalg.norm(coords[i[ok], :] - coords[j[ok], :], axis=1).mean())


def graph_edges_for_dirichlet(W: Optional[sp.csr_matrix], rng: np.random.Generator, max_edges: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    if W is None or W.shape[0] < 2 or W.nnz == 0:
        return None
    U = sp.triu(W, k=1).tocoo()
    if U.nnz == 0:
        return None
    row = U.row.astype(np.int64, copy=False)
    col = U.col.astype(np.int64, copy=False)
    data = U.data.astype(np.float64, copy=False)
    if int(max_edges) > 0 and U.nnz > int(max_edges):
        take = rng.choice(U.nnz, size=int(max_edges), replace=False)
        row, col, data = row[take], col[take], data[take]
    denom = float(np.sum(data))
    if denom <= 0 or not np.isfinite(denom):
        return None
    return row, col, data, denom


def dirichlet_energy(edges: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]], z: np.ndarray) -> float:
    if edges is None:
        return np.nan
    row, col, data, denom = edges
    if len(row) == 0 or denom <= 0:
        return np.nan
    dz = np.asarray(z, dtype=float)[row] - np.asarray(z, dtype=float)[col]
    return float(np.sum(data * dz * dz) / denom)


def compute_sample_gene_metrics(
    sample: SampleData,
    genes: Sequence[str],
    rng: np.random.Generator,
    *,
    high_quantile: float,
    min_detected: float,
    min_high_cells: int,
    exact_distance_n: int,
    compact_pairs_per_gene: int,
    background_pairs: int,
    dirichlet_max_edges: int,
) -> pd.DataFrame:
    pos = gene_pos_index(sample.genes)
    finite_coords = np.isfinite(sample.coords).all(axis=1)
    if finite_coords.sum() < 3:
        warn(f"{sample.sample}: too few finite coordinates")
        return pd.DataFrame()

    bg_dist = mean_pair_distance(sample.coords[finite_coords, :], rng, exact_distance_n, background_pairs)
    edges = graph_edges_for_dirichlet(sample.raw_uei_w, rng=rng, max_edges=dirichlet_max_edges)
    n_edges = 0 if edges is None else int(len(edges[0]))
    edge_weight_sum = np.nan if edges is None else float(edges[3])

    rows: List[Dict[str, object]] = []
    for gene in genes:
        j = pos.get(str(gene), pos.get(str(gene).lower()))
        if j is None:
            continue
        col = sample.X_log[:, j]
        x = col.toarray().ravel() if sp.issparse(col) else np.asarray(col).ravel()
        x = np.asarray(x, dtype=float)
        det = float(np.mean(x > 0))
        mean = float(np.nanmean(x))
        var = float(np.nanvar(x))

        compact_obs = compact_score = np.nan
        n_high = 0
        q_hi = np.nan
        if det >= float(min_detected) and np.isfinite(bg_dist) and bg_dist > 0:
            q_hi = float(np.nanquantile(x, float(high_quantile)))
            if q_hi <= 0:
                # For sparse genes, use detected cells as the high-expression set.
                hi = np.flatnonzero((x > 0) & finite_coords)
            else:
                hi = np.flatnonzero((x >= q_hi) & finite_coords)
            n_high = int(len(hi))
            if n_high >= int(min_high_cells):
                compact_obs = mean_pair_distance(sample.coords[hi, :], rng, exact_distance_n, compact_pairs_per_gene)
                if np.isfinite(compact_obs) and compact_obs > 0:
                    compact_score = float(-np.log2(compact_obs / bg_dist))

        smooth_obs = smooth_score = np.nan
        if edges is not None and det >= float(min_detected):
            sd = float(np.nanstd(x))
            if np.isfinite(sd) and sd > 1e-12:
                z = (x - float(np.nanmean(x))) / sd
                smooth_obs = dirichlet_energy(edges, z)
                # Analytic permutation expectation for standardized expression on a no-self-edge graph.
                null_energy = 2.0 * float(sample.n_obs) / max(1.0, float(sample.n_obs - 1))
                if np.isfinite(smooth_obs) and smooth_obs > 0 and null_energy > 0:
                    smooth_score = float(-np.log2(smooth_obs / null_energy))

        rows.append({
            "sample": sample.sample,
            "stage": sample.stage,
            "gene": str(gene),
            "xvdm_mean": mean,
            "xvdm_var": var,
            "xvdm_detection": det,
            "compactness_obs_dist": compact_obs,
            "compactness_background_dist": bg_dist,
            "xvdm_compactness_score": compact_score,
            "high_quantile": float(high_quantile),
            "high_threshold": q_hi,
            "n_high_cells": n_high,
            "raw_uei_edges_evaluated": n_edges,
            "raw_uei_edge_weight_sum": edge_weight_sum,
            "raw_uei_dirichlet": smooth_obs,
            "raw_uei_smoothness_score": smooth_score,
        })
    return pd.DataFrame(rows)


def aggregate_gene_metrics(sample_tbl: pd.DataFrame) -> pd.DataFrame:
    if sample_tbl.empty:
        return sample_tbl
    rows: List[Dict[str, object]] = []
    for gene, d in sample_tbl.groupby("gene", sort=False):
        row: Dict[str, object] = {"gene": str(gene), "n_xvdm_samples": int(d["sample"].nunique())}
        for col in ["xvdm_mean", "xvdm_var", "xvdm_detection"]:
            vals = d[col].to_numpy(float)
            row[col] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        for col in ["xvdm_compactness_score", "raw_uei_smoothness_score"]:
            vals = d[col].to_numpy(float)
            finite = np.isfinite(vals)
            row[col] = float(np.nanmean(vals[finite])) if finite.any() else np.nan
            row[f"{col}_median"] = float(np.nanmedian(vals[finite])) if finite.any() else np.nan
            row[f"n_samples_{col}"] = int(finite.sum())
        row["mean_n_high_cells"] = float(np.nanmean(d["n_high_cells"].to_numpy(float)))
        row["mean_raw_uei_edges_evaluated"] = float(np.nanmean(d["raw_uei_edges_evaluated"].to_numpy(float)))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["xvdm_log_mean"] = np.log1p(out["xvdm_mean"].astype(float))
    out["xvdm_log_var"] = np.log1p(out["xvdm_var"].astype(float))
    out["xvdm_logit_detection"] = safe_logit_fraction(out["xvdm_detection"].astype(float).to_numpy())
    return out


###############################################################################
# Gene set selection, residualization, rank-normalization, plotting
###############################################################################


def select_gene_set(
    external_tbl: pd.DataFrame,
    xvdm_gene_sets: Sequence[set[str]],
    *,
    require_all_samples: bool,
    max_genes: int,
    zesta_min_detected: float,
    zesta_min_mean: float,
) -> pd.DataFrame:
    tbl = external_tbl.copy()
    if require_all_samples:
        common = set.intersection(*xvdm_gene_sets) if xvdm_gene_sets else set()
    else:
        common = set.union(*xvdm_gene_sets) if xvdm_gene_sets else set()
    tbl = tbl[tbl["gene"].astype(str).isin(common)].copy()
    tbl = tbl[np.isfinite(tbl["zesta_moran_I"].to_numpy(float))]
    tbl = tbl[tbl["zesta_detection"].astype(float) >= float(zesta_min_detected)]
    tbl = tbl[tbl["zesta_mean"].astype(float) >= float(zesta_min_mean)]
    tbl = tbl.sort_values("zesta_moran_I", ascending=False).reset_index(drop=True)
    tbl["external_moran_rank_desc"] = np.arange(1, len(tbl) + 1)
    tbl["selection_group"] = "all_eligible"

    if int(max_genes) <= 0 or len(tbl) <= int(max_genes):
        return tbl

    # Stratified selection across the external Moran ordering: high, low, and evenly spaced middle.
    n = int(max_genes)
    n_high = n // 3
    n_low = n // 3
    n_mid = n - n_high - n_low
    high_idx = np.arange(0, min(n_high, len(tbl)))
    low_idx = np.arange(max(0, len(tbl) - n_low), len(tbl))
    middle_pool = np.setdiff1d(np.arange(len(tbl)), np.r_[high_idx, low_idx], assume_unique=False)
    if len(middle_pool) > 0 and n_mid > 0:
        pick_pos = np.linspace(0, len(middle_pool) - 1, num=min(n_mid, len(middle_pool))).round().astype(int)
        mid_idx = middle_pool[pick_pos]
    else:
        mid_idx = np.array([], dtype=int)
    keep = np.unique(np.r_[high_idx, mid_idx, low_idx]).astype(int)
    out = tbl.iloc[keep].copy()
    out["selection_group"] = "middle_stratified"
    out.loc[out.index.isin(high_idx), "selection_group"] = "external_high"
    out.loc[out.index.isin(low_idx), "selection_group"] = "external_low"
    return out.reset_index(drop=True)


def residualize(y: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 1:
        cov = cov[:, None]
    ok = np.isfinite(y) & np.isfinite(cov).all(axis=1)
    resid = np.full(y.shape, np.nan, dtype=float)
    if ok.sum() < max(5, cov.shape[1] + 2):
        return resid, ok
    X = cov[ok, :].copy()
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    X = (X - mu) / sd
    X = np.c_[np.ones(X.shape[0]), X]
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    resid[ok] = y[ok] - X @ beta
    return resid, ok


def rankit(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x)
    n = int(ok.sum())
    if n == 0:
        return out
    r = rankdata(x[ok], method="average")
    p = (r - 0.5) / float(n)
    out[ok] = norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))
    return out


def add_residualized_rank_columns(tbl: pd.DataFrame) -> pd.DataFrame:
    out = tbl.copy()
    cov_cols = [
        "zesta_log_mean", "zesta_logit_detection", "zesta_log_var",
        "xvdm_log_mean", "xvdm_logit_detection", "xvdm_log_var",
    ]
    cov = out[cov_cols].to_numpy(float)
    for metric in ["zesta_moran_I", "xvdm_compactness_score", "raw_uei_smoothness_score"]:
        resid, ok = residualize(out[metric].to_numpy(float), cov)
        out[f"{metric}_resid"] = resid
        out[f"{metric}_resid_rankit"] = rankit(resid)

    # Ratio panel: compute UEI smoothness relative to inferred-3D compactness
    # on the same residualized scale used by the existing rank-correlograms,
    # then rank-normalize the ratio for plotting.
    numerator = out["raw_uei_smoothness_score_resid"].to_numpy(float)
    denominator = out["xvdm_compactness_score_resid"].to_numpy(float)
    ratio = np.full(numerator.shape, np.nan, dtype=float)
    ok = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    ratio[ok] = numerator[ok] / denominator[ok]
    out["raw_uei_smoothness_to_xvdm_compactness_resid_ratio"] = ratio
    out["raw_uei_smoothness_to_xvdm_compactness_resid_ratio_rankit"] = rankit(ratio)
    return out


def summarize_correlations(tbl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [
        ("compactness", "zesta_moran_I_resid", "xvdm_compactness_score_resid"),
        ("raw_uei_smoothness", "zesta_moran_I_resid", "raw_uei_smoothness_score_resid"),
        (
            "raw_uei_smoothness_to_xvdm_compactness_resid_ratio",
            "zesta_moran_I_resid_rankit",
            "raw_uei_smoothness_to_xvdm_compactness_resid_ratio_rankit",
        ),
        ("compactness_raw", "zesta_moran_I", "xvdm_compactness_score"),
        ("raw_uei_smoothness_raw", "zesta_moran_I", "raw_uei_smoothness_score"),
    ]
    for name, xcol, ycol in pairs:
        x = tbl[xcol].to_numpy(float)
        y = tbl[ycol].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 3:
            rho, p = spearmanr(x[ok], y[ok])
        else:
            rho, p = np.nan, np.nan
        rows.append({"comparison": name, "x": xcol, "y": ycol, "n_genes": int(ok.sum()), "spearman_rho": float(rho), "spearman_p": float(p)})
    return pd.DataFrame(rows)


def configure_matplotlib(font_name: Optional[str], dpi: int) -> str:
    fonts = {f.name for f in fm.fontManager.ttflist}
    candidates = [font_name] if font_name else []
    candidates += ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    chosen = next((f for f in candidates if f and f in fonts), "DejaVu Sans")

    # All font sizes multiplied by 2 to satisfy text size constraint.
    # Line width slightly scaled up to match visual weight.
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [chosen, "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 16.0,
        "axes.titlesize": 17.2,
        "axes.labelsize": 16.4,
        "xtick.labelsize": 14.8,
        "ytick.labelsize": 14.8,
        "legend.fontsize": 14.4,
        "axes.linewidth": 1.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": int(dpi),
    })
    return chosen


def save_figure(fig: plt.Figure, outdir: Path, stem: str, formats: Sequence[str], dpi: int) -> None:
    ensure_dir(outdir)
    for fmt in formats:
        fig.savefig(outdir / f"{stem}.{fmt}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)

def plot_rank_scatter(
    ax: plt.Axes,
    tbl: pd.DataFrame,
    xcol: str,
    ycol: str,
    *,
    title: str,
    y_label: str,
    highlight_genes: Sequence[str],
    label_genes: Sequence[str],
) -> None:
    x = tbl[xcol].to_numpy(float)
    y = tbl[ycol].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    df = tbl.loc[ok].copy()
    ax.scatter(df[xcol], df[ycol], s=9, lw=0, alpha=0.28, color="0.55", rasterized=True, label="other genes")

    highlight = {g.lower() for g in highlight_genes}
    h = df[df["gene"].astype(str).str.lower().isin(highlight)].copy()
    if not h.empty:
        ax.scatter(h[xcol], h[ycol], s=28, facecolors="#f59e0b", edgecolors="black", linewidths=0.45, alpha=0.95, label="YSL/lipid genes")

    label_set = {g.lower() for g in label_genes}

    # Store text objects to pass to adjust_text for inter-avoidance
    texts = []
    for r in h.itertuples(index=False):
        gene = str(getattr(r, "gene"))
        if gene.lower() not in label_set:
            continue
        xv = float(getattr(r, xcol))
        yv = float(getattr(r, ycol))
        if np.isfinite(xv) and np.isfinite(yv):
            # Font size doubled from 6.2 -> 12.4
            t = ax.text(xv, yv, gene, fontsize=12.4, ha="center", va="center")
            texts.append(t)

    # Apply inter-avoidance using adjustText
    if texts:
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
        except ImportError:
            warn("adjustText is not installed. Overlaps will occur. Run 'pip install adjustText'.")

    # Font size doubled from 7.2 -> 14.4
    if len(df) >= 3:
        rho, p = spearmanr(df[xcol].to_numpy(float), df[ycol].to_numpy(float))
        stat = f"Spearman $\\rho$={rho:.2f}\n$p$={p:.1e}\n$n$={len(df):,} genes"
    else:
        stat = f"n={len(df):,} genes"
    ax.text(0.03, 0.97, stat, transform=ax.transAxes, ha="left", va="top", fontsize=14.4)
    ax.axhline(0, color="0.85", lw=0.7, zorder=0)
    ax.axvline(0, color="0.85", lw=0.7, zorder=0)
    ax.set_title(title, pad=4)
    ax.set_xlabel("external Moran's $I$\nresidualized rank")
    ax.set_ylabel(y_label)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.4)

def plot_rankcorr_panel(tbl: pd.DataFrame, outdir: Path, formats: Sequence[str], dpi: int, highlight_genes: Sequence[str], label_genes: Sequence[str]) -> None:
    # Original script generated a 1x2 subplot here.
    # To maintain 4 height and 2:1 aspect ratio per plot, total width must be 12.6.
    fig, axes = plt.subplots(1, 2, figsize=(15, 4), constrained_layout=True)
    plot_rank_scatter(
        axes[0], tbl,
        "zesta_moran_I_resid_rankit", "xvdm_compactness_score_resid_rankit",
        title="inferred 3D compactness",
        y_label="xVDM compactness\nresidualized rank",
        highlight_genes=highlight_genes,
        label_genes=label_genes,
    )
    plot_rank_scatter(
        axes[1], tbl,
        "zesta_moran_I_resid_rankit", "raw_uei_smoothness_score_resid_rankit",
        title="raw contact-graph smoothness",
        y_label="raw-UEI smoothness\nresidualized rank",
        highlight_genes=highlight_genes,
        label_genes=label_genes,
    )
    save_figure(fig, outdir, "panel4de_external_moran_rankcorr", formats, dpi)

    # Single panels: 4 height, 5 width
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
    plot_rank_scatter(
        ax, tbl,
        "zesta_moran_I_resid_rankit", "xvdm_compactness_score_resid_rankit",
        title="inferred 3D compactness",
        y_label="xVDM compactness\nresidualized rank",
        highlight_genes=highlight_genes,
        label_genes=label_genes,
    )
    save_figure(fig, outdir, "panel4d_external_moran_vs_xvdm_compactness", formats, dpi)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
    plot_rank_scatter(
        ax, tbl,
        "zesta_moran_I_resid_rankit", "raw_uei_smoothness_score_resid_rankit",
        title="raw contact-graph smoothness",
        y_label="raw-UEI smoothness\nresidualized rank",
        highlight_genes=highlight_genes,
        label_genes=label_genes,
    )
    save_figure(fig, outdir, "panel4e_external_moran_vs_rawuei_smoothness", formats, dpi)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
    plot_rank_scatter(
        ax, tbl,
        "zesta_moran_I_resid_rankit", "raw_uei_smoothness_to_xvdm_compactness_resid_ratio_rankit",
        title="raw-UEI / 3D compactness ratio",
        y_label="raw-UEI smoothness / xVDM compactness\nrank-transformed ratio",
        highlight_genes=highlight_genes,
        label_genes=label_genes,
    )
    save_figure(fig, outdir, "panel4f_external_moran_vs_rawuei_to_xvdm_compactness_ratio", formats, dpi)


###############################################################################
# CLI
###############################################################################


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Global gene-level rank-correlogram: external Moran's I vs xVDM compactness/raw-UEI smoothness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--h5ad-dir", type=Path, required=True, help="Directory containing per-sample connectivity H5AD files.")
    p.add_argument("--h5ad-glob", type=str, default="*ann12_leiden_cell_connectivity*.h5ad", help="Glob for per-sample connectivity H5AD files.")
    p.add_argument("--samples", type=str, default="zf3,zf4", help="Comma-separated xVDM samples to compare against the 18 hpf external slice.")
    p.add_argument("--zesta18", type=Path, required=True, help="Stereo-seq/ZESTA whole-dataset H5AD; filtered by --zesta-time.")
    p.add_argument("--zesta-time", type=str, default="18hpf", help="Value in ZESTA obs['time'] for the external reference.")
    p.add_argument("--zesta-layer", type=str, default="counts", help="Counts layer in ZESTA H5AD; falls back to .X if missing.")
    p.add_argument("--zesta-spatial-x", type=str, default="spatial_x")
    p.add_argument("--zesta-spatial-y", type=str, default="spatial_y")
    p.add_argument("--zesta-neighbors", type=int, default=6, help="kNN graph size for external Moran's I.")
    p.add_argument("--outdir", type=Path, required=True)

    p.add_argument("--require-all-samples", action="store_true", default=True, help="Candidate genes must be present in all selected xVDM samples.")
    p.add_argument("--allow-partial-sample-genes", dest="require_all_samples", action="store_false", help="Allow genes present in only some selected samples.")
    p.add_argument("--max-genes", type=int, default=3000, help="0 uses all eligible genes. Positive values select high, low, and stratified middle genes by external Moran rank for tractability.")
    p.add_argument("--zesta-min-detected", type=float, default=0.02, help="External detection fraction filter.")
    p.add_argument("--zesta-min-mean", type=float, default=0.0, help="External mean-expression filter after normalization/log1p.")
    p.add_argument("--xvdm-min-detected", type=float, default=0.02, help="xVDM detection fraction needed for compactness/smoothness calculation.")
    p.add_argument("--high-quantile", type=float, default=0.80, help="Quantile defining gene-high cells for compactness.")
    p.add_argument("--min-high-cells", type=int, default=25, help="Minimum gene-high cells per sample for compactness.")
    p.add_argument("--exact-distance-n", type=int, default=1500, help="Use exact pairwise distances below this group size.")
    p.add_argument("--background-pairs", type=int, default=50000, help="Sampled pairs for sample-wide compactness background.")
    p.add_argument("--compact-pairs-per-gene", type=int, default=2500, help="Sampled high-cell pairs per gene for compactness screen.")
    p.add_argument("--dirichlet-max-edges", type=int, default=75000, help="Maximum raw-UEI edges sampled per sample for gene-level Dirichlet energies. 0 uses all edges.")
    p.add_argument("--raw-uei-from-link-assoc", action="store_true", default=True, help="Reconstruct raw UEI cell graph from link_assoc_reindexed.npz and hub_refined_labels.tsv when absent from H5AD.")
    p.add_argument("--no-raw-uei-from-link-assoc", dest="raw_uei_from_link_assoc", action="store_false")
    p.add_argument("--raw-uei-cache-dir", type=Path, default=None, help="Optional cache dir for reconstructed raw-UEI cell graphs.")

    p.add_argument("--highlight-genes", type=str, default=",".join(YSL_LIPID_GENES), help="Comma-separated genes to highlight in the correlograms.")
    p.add_argument("--label-genes", type=str, default="apoa1a,afp4,tfa,apoa1b,apoba,bhmt,apoc1", help="Highlighted genes to text-label.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--font", type=str, default=None)
    p.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    p.add_argument("--dpi", type=int, default=400)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    ensure_dir(args.outdir)
    font = configure_matplotlib(args.font, args.dpi)
    info(f"Using font: {font}")

    sample_ids = parse_csv_list(args.samples)
    raw_cache = args.raw_uei_cache_dir or (args.outdir / "_raw_uei_cell_graph_cache")
    paths = collect_h5ad_paths(args.h5ad_dir, args.h5ad_glob)
    samples = load_samples(paths, sample_ids, raw_cache, bool(args.raw_uei_from_link_assoc))
    if not samples:
        raise SystemExit(f"No selected samples loaded from {args.h5ad_dir}")

    zesta = load_zesta_h5ad(args.zesta18, args.zesta_time, args.zesta_layer, args.zesta_spatial_x, args.zesta_spatial_y)
    external = compute_external_moran_table(zesta, args.zesta_neighbors)
    external.to_csv(args.outdir / "source_external_moran_all_genes.csv", index=False)

    xvdm_gene_sets = [set(map(str, s.genes)) for s in samples]
    selected = select_gene_set(
        external,
        xvdm_gene_sets,
        require_all_samples=bool(args.require_all_samples),
        max_genes=int(args.max_genes),
        zesta_min_detected=float(args.zesta_min_detected),
        zesta_min_mean=float(args.zesta_min_mean),
    )
    genes = selected["gene"].astype(str).tolist()
    if not genes:
        raise SystemExit("No candidate genes after external/xVDM gene filters")
    info(f"Selected {len(genes):,} genes for internal xVDM metrics")

    sample_tables: List[pd.DataFrame] = []
    for s in samples:
        info(f"Computing xVDM gene metrics for {s.sample}")
        st = compute_sample_gene_metrics(
            s,
            genes,
            rng,
            high_quantile=float(args.high_quantile),
            min_detected=float(args.xvdm_min_detected),
            min_high_cells=int(args.min_high_cells),
            exact_distance_n=int(args.exact_distance_n),
            compact_pairs_per_gene=int(args.compact_pairs_per_gene),
            background_pairs=int(args.background_pairs),
            dirichlet_max_edges=int(args.dirichlet_max_edges),
        )
        sample_tables.append(st)
    sample_tbl = pd.concat(sample_tables, ignore_index=True) if sample_tables else pd.DataFrame()
    sample_tbl.to_csv(args.outdir / "source_rankcorr_sample_metrics.csv", index=False)

    gene_tbl = aggregate_gene_metrics(sample_tbl)
    merged = selected.merge(gene_tbl, on="gene", how="left")
    merged = add_residualized_rank_columns(merged)
    merged.to_csv(args.outdir / "source_rankcorr_gene_metrics.csv", index=False)

    summary = summarize_correlations(merged)
    summary.to_csv(args.outdir / "table_rankcorr_summary.csv", index=False)
    info("Correlation summary:\n" + summary.to_string(index=False))

    highlight = parse_csv_list(args.highlight_genes)
    label_genes = parse_csv_list(args.label_genes)
    plot_rankcorr_panel(merged, args.outdir, args.formats, args.dpi, highlight, label_genes)

    qc = {
        "font": font,
        "samples": [s.sample for s in samples],
        "zesta_time": args.zesta_time,
        "n_external_genes_total": int(len(external)),
        "n_selected_genes": int(len(merged)),
        "max_genes": int(args.max_genes),
        "selection_groups": merged["selection_group"].value_counts(dropna=False).to_dict() if "selection_group" in merged else {},
        "raw_uei_graph_samples": {s.sample: bool(s.raw_uei_w is not None) for s in samples},
        "filters": {
            "zesta_min_detected": float(args.zesta_min_detected),
            "zesta_min_mean": float(args.zesta_min_mean),
            "xvdm_min_detected": float(args.xvdm_min_detected),
            "high_quantile": float(args.high_quantile),
            "min_high_cells": int(args.min_high_cells),
        },
        "notes": [
            "Compactness is global within each selected xVDM sample, not metacluster-restricted.",
            "Raw-UEI smoothness is computed on the cell-coarsened raw UEI contact graph.",
            "The ratio panel uses raw-UEI smoothness residuals divided by xVDM compactness residuals, then rank-normalizes the ratio.",
            "Rank-correlograms use residualized rank-normal scores after regressing out external and xVDM mean-expression, detection, and variance covariates.",
        ],
    }
    (args.outdir / "qc_summary.json").write_text(json.dumps(qc, indent=2, default=float))
    info(f"Done. Outputs written to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

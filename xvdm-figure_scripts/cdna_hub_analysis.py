#!/usr/bin/env python3
"""
Cleaned non-spatial DNAMIC cDNA pipeline.

This script keeps only the raw-data through line needed to produce:
  1. 01_compartment_split_volcano.* plus compartment-split tables/metadata
  2. 01_compartment_split_stage_profiles.* plus per-sample/stage tables
  3. 02_module_coherence_by_stage.* plus module-coherence tables/metadata
  4. 06_07_anchor_neighborhood_pair.* plus hbae3/ctslb anchor tables/metadata

The unit of replication is biological sample. Rows are matched within
(umi_type, clipped row complexity) strata. voxel_index/voxelindex is detected
only for provenance and is never used for inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import anndata as ad
except ImportError:  # keeps --help/import usable outside analysis envs
    ad = None
import numpy as np
import pandas as pd
from scipy import sparse, stats


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cdna_four_output_sets")

DEFAULT_SEED = 42
COMPLEXITY_MAX_BIN = 6
SIGNFLIP_EXACT_MAX_SAMPLES = 20
SIGNFLIP_MC_PATTERNS = 20_000
DISPLAY_FLOOR_Q = 1e-30
DISPLAY_FLOOR_NEGLOG10 = 30.0
FIGURE_DIRNAME = "figures"

DEFAULT_STAGE_SAMPLES: Dict[str, List[str]] = {
    "12hpf": ["zf1", "zf2"],
    "18hpf_high_perm": ["zf3", "zf4"],
    "18hpf_low_perm": ["zf5", "zf6"],
    "24hpf": ["zf7", "zf8"],
}

STAGE_DISPLAY = {
    "12hpf": "12 hpf",
    "18hpf_high_perm": "18 hpf high-perm",
    "18hpf_low_perm": "18 hpf low-perm",
    "24hpf": "24 hpf",
}

PROTEIN_CODING_ALIASES = {
    "proteincoding",
    "protein_coding",
    "protein-coding",
    "protein coding",
}

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "gray": "#9A9A9A",
    "dark_gray": "#4D4D4D",
    "light_gray": "#D9D9D9",
    "black": "#000000",
}


# -----------------------------------------------------------------------------
# I/O and statistics
# -----------------------------------------------------------------------------


def save_json(obj: object, path: Path) -> None:
    """Write JSON with numpy-safe scalar/array serialization."""

    class Encoder(json.JSONEncoder):
        def default(self, o):  # type: ignore[override]
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.bool_):
                return bool(o)
            return super().default(o)

    with path.open("w") as f:
        json.dump(obj, f, indent=2, cls=Encoder)
    log.info("Saved %s", path)


def tsv_write(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)
    log.info("Saved %s", path)


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    prev = 1.0
    for i in range(len(ranked) - 1, -1, -1):
        prev = min(prev, ranked[i] * len(ranked) / (i + 1))
        q[i] = prev
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def log_or_and_var(case_x: np.ndarray, case_n: int, ctrl_x: np.ndarray, ctrl_n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized log odds-ratio and variance with Haldane-Anscombe correction."""
    a = case_x.astype(float) + 0.5
    b = case_n - case_x.astype(float) + 0.5
    c = ctrl_x.astype(float) + 0.5
    d = ctrl_n - ctrl_x.astype(float) + 0.5
    return np.log((a * d) / (b * c)), 1 / a + 1 / b + 1 / c + 1 / d


def fixed_effect_meta(lors: Sequence[np.ndarray], vars_: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-effect meta-analysis across sample-level log odds-ratios."""
    if not lors:
        raise ValueError("No sample-level effects supplied.")
    lor = np.vstack(lors)
    var = np.vstack(vars_)
    w = 1.0 / var
    meta_lor = (w * lor).sum(axis=0) / w.sum(axis=0)
    meta_var = 1.0 / w.sum(axis=0)
    z = meta_lor / np.sqrt(meta_var)
    return meta_lor, meta_var, z, 2.0 * stats.norm.sf(np.abs(z))


def sample_signflip_meta_p(lors: Sequence[np.ndarray], vars_: Sequence[np.ndarray], seed: int = DEFAULT_SEED) -> Tuple[np.ndarray, str, int]:
    """Two-sided sign-flip sensitivity over biological-sample effect directions."""
    if not lors:
        return np.array([], float), "none", 0
    lor = np.vstack([np.asarray(x, float) for x in lors])
    var = np.vstack([np.asarray(x, float) for x in vars_])
    w = np.where(np.isfinite(var) & (var > 0), 1.0 / var, 0.0)
    denom = np.sqrt(w.sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        obs = np.where(denom > 0, (w * lor).sum(axis=0) / denom, np.nan)
    abs_obs = np.abs(obs)
    n_samples, n_genes = lor.shape
    if n_samples <= SIGNFLIP_EXACT_MAX_SAMPLES:
        n_patterns = 1 << n_samples
        exceed = np.zeros(n_genes, np.int64)
        for mask in range(n_patterns):
            signs = np.ones(n_samples, float)
            for i in range(n_samples):
                if (mask >> i) & 1:
                    signs[i] = -1.0
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(denom > 0, (signs[:, None] * w * lor).sum(axis=0) / denom, np.nan)
            exceed += (np.abs(z) >= abs_obs - 1e-15) & np.isfinite(abs_obs)
        p = exceed.astype(float) / n_patterns
        p[~np.isfinite(abs_obs)] = np.nan
        return np.clip(p, 0, 1), "exact", n_patterns

    rng = np.random.default_rng(seed + 9176)
    exceed = np.zeros(n_genes, np.int64)
    for _ in range(SIGNFLIP_MC_PATTERNS):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_samples)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(denom > 0, (signs[:, None] * w * lor).sum(axis=0) / denom, np.nan)
        exceed += (np.abs(z) >= abs_obs - 1e-15) & np.isfinite(abs_obs)
    p = (exceed.astype(float) + 1.0) / (SIGNFLIP_MC_PATTERNS + 1)
    p[~np.isfinite(abs_obs)] = np.nan
    return np.clip(p, 0, 1), "monte_carlo", SIGNFLIP_MC_PATTERNS


# -----------------------------------------------------------------------------
# Backed AnnData access
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleInfo:
    name: str
    code: int
    spans: Tuple[Tuple[int, int], ...]
    n_rows: int


class BackedDNAMIC:
    """Small backed AnnData wrapper for sample-aware sparse row access."""

    def __init__(self, h5ad_path: Path):
        if ad is None:
            raise ImportError("This pipeline requires anndata: pip install anndata")
        self.path = Path(h5ad_path)
        self.adata = ad.read_h5ad(self.path, backed="r")
        self.n_obs, self.n_vars = self.adata.shape

        obs_cols = set(self.adata.obs.columns)
        var_cols = set(self.adata.var.columns)
        self.sample_col = self._pick(obs_cols, ["sample"])
        self.umi_col = self._pick(obs_cols, ["umi_type", "umitype"])
        self.voxel_index_col = self._pick_optional(obs_cols, ["voxel_index", "voxelindex"])
        self.feature_type_col = self._pick(var_cols, ["feature_type", "featuretype"])

        self.var_names = np.asarray(self.adata.var_names).astype(str)
        self.feature_type = self._normalize_feature_types(self.adata.var[self.feature_type_col])
        self.pc_idx, self.rrna_idx, self.mtrna_idx = self._infer_feature_groups()

        sample_series = self.adata.obs[self.sample_col]
        if isinstance(sample_series.dtype, pd.CategoricalDtype):
            self.sample_codes = sample_series.cat.codes.to_numpy(np.int16, copy=True)
            self.sample_levels = [str(x) for x in sample_series.cat.categories]
        else:
            codes, levels = pd.factorize(sample_series.astype(str), sort=False)
            self.sample_codes = codes.astype(np.int16, copy=False)
            self.sample_levels = [str(x) for x in levels]
        self.umi = np.asarray(self.adata.obs[self.umi_col], dtype=np.int8)
        self.sample_info = self._build_sample_info()
        self.sample_names = [s.name for s in self.sample_info]
        self.sample_name_to_info = {s.name: s for s in self.sample_info}
        self.sample_alias_to_name = self._build_sample_alias_map()

        log.info("Opened %s in backed mode", self.path)
        log.info("Shape: %s obs x %s vars", f"{self.n_obs:,}", f"{self.n_vars:,}")
        log.info(
            "Detected columns: sample=%s, umi=%s, voxel_index=%s, feature_type=%s",
            self.sample_col,
            self.umi_col,
            self.voxel_index_col,
            self.feature_type_col,
        )
        log.info("Samples: %s", self.sample_names)
        log.info("Protein-coding=%d, rRNA=%d, MtrRNA=%d", len(self.pc_idx), len(self.rrna_idx), len(self.mtrna_idx))
        if min(len(self.pc_idx), len(self.rrna_idx), len(self.mtrna_idx)) == 0:
            log.warning("One or more inferred feature groups are empty; check feature_type labels.")

    @staticmethod
    def _pick(cols: set[str], candidates: Sequence[str]) -> str:
        for c in candidates:
            if c in cols:
                return c
        raise KeyError(f"Missing required column among: {candidates}")

    @staticmethod
    def _pick_optional(cols: set[str], candidates: Sequence[str]) -> Optional[str]:
        return next((c for c in candidates if c in cols), None)

    @staticmethod
    def _normalize_feature_types(series: pd.Series) -> np.ndarray:
        vals = pd.Series(series, copy=False).astype("string").fillna("")
        vals = vals.str.strip().str.lower().replace({"<na>": "", "nan": "", "none": ""})
        vals = vals.str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
        return np.array(["" if pd.isna(x) else str(x) for x in vals.to_numpy(object)], dtype=object)

    def _infer_feature_groups(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pc, rrna, mtrna = [], [], []
        for i, raw in enumerate(self.feature_type):
            x = "" if pd.isna(raw) else str(raw)
            if x in PROTEIN_CODING_ALIASES or ("protein" in x and "coding" in x):
                pc.append(i)
            if "rrna" in x:
                (mtrna if ("mt" in x or "mito" in x or "mtrrna" in x) else rrna).append(i)
        if not mtrna:
            mtrna = [i for i, g in enumerate(self.var_names) if str(g).lower().startswith(("mt-rnr", "mtrnr"))]
        if not rrna:
            prefixes = ("rna18s", "rna28s", "5s_", "5_8s", "rn45s", "rrna")
            rrna = [i for i, g in enumerate(self.var_names) if str(g).lower().startswith(prefixes)]
        return tuple(np.array(sorted(set(v)), np.int32) for v in (pc, rrna, mtrna))  # type: ignore[return-value]

    def _build_sample_info(self) -> List[SampleInfo]:
        codes = self.sample_codes
        starts = np.r_[0, np.flatnonzero(np.diff(codes) != 0) + 1]
        ends = np.r_[starts[1:], len(codes)]
        spans_by_code: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for s, e in zip(starts, ends):
            spans_by_code[int(codes[s])].append((int(s), int(e)))
        out = []
        for code, name in enumerate(self.sample_levels):
            spans = tuple(spans_by_code.get(code, []))
            n_rows = int(sum(e - s for s, e in spans))
            if n_rows:
                out.append(SampleInfo(str(name), code, spans, n_rows))
        return out

    @staticmethod
    def _sample_alias(name: str) -> Optional[str]:
        m = re.search(r"(zf\d+)", str(name).lower())
        return m.group(1) if m else None

    def _build_sample_alias_map(self) -> Dict[str, str]:
        groups: Dict[str, List[str]] = defaultdict(list)
        for info in self.sample_info:
            alias = self._sample_alias(info.name)
            if alias:
                groups[alias].append(info.name)
        out = {}
        for alias, names in groups.items():
            if len(names) == 1:
                out[alias] = names[0]
            else:
                log.warning("Sample alias %s is ambiguous across %s; not using alias.", alias, names)
        return out

    def resolve_sample_name(self, sample_name: str) -> str:
        if sample_name in self.sample_name_to_info:
            return sample_name
        alias = self._sample_alias(sample_name)
        if alias and alias in self.sample_alias_to_name:
            return self.sample_alias_to_name[alias]
        raise KeyError(sample_name)

    def resolve_sample_names(self, samples: Sequence[str]) -> List[str]:
        return [self.resolve_sample_name(s) for s in samples]

    def samples_for_stage(self, stage: str) -> List[str]:
        return self.resolve_sample_names(DEFAULT_STAGE_SAMPLES[stage])

    def get_sample_info(self, sample: str) -> SampleInfo:
        return self.sample_name_to_info[self.resolve_sample_name(sample)]

    def random_rows(self, sample: str, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample global row indices without materializing large per-sample aranges."""
        info = self.get_sample_info(sample)
        n = min(int(n), info.n_rows)
        if n <= 0:
            return np.array([], np.int64)
        if len(info.spans) == 1:
            start, _ = info.spans[0]
            return np.sort(start + rng.choice(info.n_rows, n, replace=False).astype(np.int64))
        offsets = np.sort(rng.choice(info.n_rows, n, replace=False).astype(np.int64))
        sizes = np.array([e - s for s, e in info.spans], np.int64)
        cum = np.cumsum(sizes)
        prev = np.r_[0, cum[:-1]]
        span_ids = np.searchsorted(cum, offsets, side="right")
        rows = np.empty(offsets.size, np.int64)
        for sid in np.unique(span_ids):
            mask = span_ids == sid
            start, _ = info.spans[int(sid)]
            rows[mask] = start + offsets[mask] - prev[int(sid)]
        return rows

    def fetch_rows(self, rows: np.ndarray, cols: Optional[np.ndarray] = None) -> sparse.csr_matrix:
        rows = np.sort(np.asarray(rows, np.int64))
        if rows.size == 0:
            return sparse.csr_matrix((0, self.n_vars if cols is None else len(cols)), dtype=np.uint8)
        x = self.adata[rows, :].X if cols is None else self.adata[rows, np.asarray(cols, np.int64)].X
        return to_csr_binary(x)


# -----------------------------------------------------------------------------
# Sparse helpers and matching
# -----------------------------------------------------------------------------


def to_csr_binary(x) -> sparse.csr_matrix:
    """Return a binary CSR sparse matrix."""
    csr = x.tocsr(copy=True) if sparse.issparse(x) else sparse.csr_matrix(x)
    if csr.nnz:
        csr.data[:] = 1
    return csr


def row_nnz_csr(x: sparse.csr_matrix) -> np.ndarray:
    return np.diff(x.indptr).astype(np.int16, copy=False)


def clip_complexity(row_nnz: np.ndarray, max_bin: int = COMPLEXITY_MAX_BIN) -> np.ndarray:
    out = np.asarray(row_nnz, np.int16).copy()
    out[out > max_bin] = max_bin
    return out


def rows_with_any(x: sparse.csr_matrix, cols: np.ndarray) -> np.ndarray:
    return np.asarray(x[:, cols].sum(axis=1)).ravel() > 0 if cols.size else np.zeros(x.shape[0], bool)


def _bucket_indices(mask: np.ndarray, umi: np.ndarray, complexity: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i in np.flatnonzero(mask):
        buckets[(int(umi[i]), int(complexity[i]))].append(int(i))
    return {k: np.asarray(v, np.int32) for k, v in buckets.items()}


def matched_case_control_indices(
    case_mask: np.ndarray,
    control_mask: np.ndarray,
    umi: np.ndarray,
    row_nnz: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match case/control local row indices within (umi_type, complexity_bin)."""
    complexity = clip_complexity(row_nnz)
    case_b = _bucket_indices(case_mask, umi, complexity)
    ctrl_b = _bucket_indices(control_mask, umi, complexity)
    case_keep, ctrl_keep = [], []
    for key, case_idx in case_b.items():
        ctrl_idx = ctrl_b.get(key)
        if ctrl_idx is None or case_idx.size == 0 or ctrl_idx.size == 0:
            continue
        n = min(case_idx.size, ctrl_idx.size)
        if n <= 0:
            continue
        case_keep.append(np.sort(rng.choice(case_idx, n, replace=False) if case_idx.size > n else case_idx))
        ctrl_keep.append(np.sort(rng.choice(ctrl_idx, n, replace=False) if ctrl_idx.size > n else ctrl_idx[:n]))
    if not case_keep:
        return np.array([], np.int32), np.array([], np.int32)
    return np.sort(np.concatenate(case_keep)), np.sort(np.concatenate(ctrl_keep))


def pairwise_jaccard_values(x: sparse.csr_matrix, genes: Sequence[str], gene_to_col: Mapping[str, int]) -> np.ndarray:
    idx = [gene_to_col[g] for g in genes if g in gene_to_col]
    if len(idx) < 2:
        return np.array([], float)
    sub = x[:, idx].astype(np.uint8)
    inter = (sub.T @ sub).toarray().astype(float)
    per_gene = np.asarray(sub.sum(axis=0)).ravel().astype(float)
    union = per_gene[:, None] + per_gene[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        j = np.where(union > 0, inter / union, 0.0)
    return j[np.triu_indices_from(j, k=1)]


def cross_jaccard_values(x: sparse.csr_matrix, genes_a: Sequence[str], genes_b: Sequence[str], gene_to_col: Mapping[str, int]) -> np.ndarray:
    a_idx = [gene_to_col[g] for g in genes_a if g in gene_to_col]
    b_idx = [gene_to_col[g] for g in genes_b if g in gene_to_col]
    if not a_idx or not b_idx:
        return np.array([], float)
    a = x[:, a_idx].astype(np.uint8)
    b = x[:, b_idx].astype(np.uint8)
    inter = (a.T @ b).toarray().astype(float)
    na = np.asarray(a.sum(axis=0)).ravel().astype(float)
    nb = np.asarray(b.sum(axis=0)).ravel().astype(float)
    union = na[:, None] + nb[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0).ravel()


def is_ribosomal_gene(name: str) -> bool:
    g = name.lower()
    return g.startswith("rpl") or g.startswith("rps")


def is_hemoglobin_gene(name: str) -> bool:
    return name.lower().startswith("hb")


def is_housekeeping_partner(name: str) -> bool:
    g = name.lower()
    return g.startswith(("rpl", "rps", "eef", "eif", "mt-", "act"))


def select_module_genes(var_names: Sequence[str]) -> Dict[str, List[str]]:
    """Fixed module definitions used by the all-stage coherence diagnostic."""
    genes = [str(g) for g in var_names]
    gene_set = set(genes)
    lipid_seed = [
        "apoa1a", "apoa2", "apoa4a", "apobb.1", "apoc1", "apoc2",
        "fabp1b.1", "fabp2", "fabp6", "fabp10a", "acsl4a", "acsl5", "cd36", "lpl", "scd",
    ]
    ribosomal = [g for g in genes if is_ribosomal_gene(g)][:40]
    return {
        "lipid": [g for g in lipid_seed if g in gene_set],
        "protein_synthesis": ribosomal,
        "mt_encoded": [g for g in genes if g.lower().startswith("mt-")],
        "nuclear_translation": [g for g in ribosomal if not g.lower().startswith("mt-")][:20],
    }


# -----------------------------------------------------------------------------
# Sample/stage labels
# -----------------------------------------------------------------------------


def _sample_alias(sample_name: str) -> str:
    m = re.search(r"(zf\d+)", str(sample_name).lower())
    return m.group(1) if m else str(sample_name)


def _sample_order_key(sample_name: str) -> Tuple[int, str]:
    m = re.search(r"zf(\d+)", _sample_alias(sample_name))
    return (int(m.group(1)) if m else 9999, str(sample_name))


def _sample_stage(sample_name: str) -> str:
    alias = _sample_alias(sample_name)
    for stage, aliases in DEFAULT_STAGE_SAMPLES.items():
        if alias in aliases:
            return stage
    return "unknown"


# -----------------------------------------------------------------------------
# 01: compartment split volcano and stage profiles
# -----------------------------------------------------------------------------


def _fixed_effect_1d(lor: np.ndarray, var: np.ndarray) -> Tuple[float, float, int]:
    ok = np.isfinite(lor) & np.isfinite(var) & (var > 0)
    if not ok.any():
        return np.nan, np.nan, 0
    w = 1.0 / var[ok]
    return float((w * lor[ok]).sum() / w.sum()), float(np.sqrt(1.0 / w.sum())), int(ok.sum())


def _write_compartment_stage_tables(per_sample: pd.DataFrame, diff_df: pd.DataFrame, outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if per_sample.empty:
        empty = pd.DataFrame()
        tsv_write(empty, outdir / "01_compartment_split_by_stage_long.tsv")
        tsv_write(empty, outdir / "01_compartment_split_by_stage.tsv")
        return empty, empty

    records = []
    for (gene, stage), d in per_sample.groupby(["gene", "stage_group"], sort=False):
        est, se, n = _fixed_effect_1d(
            d["sample_log_or_mtrrna_vs_rrna"].to_numpy(float),
            d["sample_var"].to_numpy(float),
        )
        records.append({
            "gene": str(gene),
            "stage_group": str(stage),
            "stage_label": STAGE_DISPLAY.get(str(stage), str(stage)),
            "stage_log_or_mtrrna_vs_rrna": est,
            "stage_se": se,
            "stage_n_samples": n,
            "stage_matched_rrna_rows": int(d["matched_rrna_n"].sum()),
            "stage_matched_mtrrna_rows": int(d["matched_mtrrna_n"].sum()),
        })
    long = pd.DataFrame(records).sort_values(["gene", "stage_group"])
    tsv_write(long, outdir / "01_compartment_split_by_stage_long.tsv")

    wide = diff_df[["gene", "meta_log_or_mtrrna_vs_rrna", "meta_q", "direction"]].copy()
    for stage in DEFAULT_STAGE_SAMPLES:
        sub = long.loc[long["stage_group"] == stage, ["gene", "stage_log_or_mtrrna_vs_rrna", "stage_se", "stage_n_samples"]]
        wide = wide.merge(
            sub.rename(columns={
                "stage_log_or_mtrrna_vs_rrna": f"log_or_{stage}",
                "stage_se": f"se_{stage}",
                "stage_n_samples": f"n_samples_{stage}",
            }),
            on="gene",
            how="left",
        )
    wide["delta_log_or_24hpf_minus_12hpf"] = wide.get("log_or_24hpf", np.nan) - wide.get("log_or_12hpf", np.nan)

    sign_map = wide.set_index("gene")["meta_log_or_mtrrna_vs_rrna"].to_dict()
    tmp = per_sample[["gene", "sample_log_or_mtrrna_vs_rrna"]].copy()
    tmp["pooled_sign"] = tmp["gene"].map(sign_map).astype(float)
    tmp["same_sign_as_pooled"] = (
        np.sign(tmp["sample_log_or_mtrrna_vs_rrna"].to_numpy(float)) == np.sign(tmp["pooled_sign"].to_numpy(float))
    ) & (np.sign(tmp["pooled_sign"].to_numpy(float)) != 0)
    consistency = tmp.groupby("gene").agg(
        sign_consistent_n=("same_sign_as_pooled", "sum"),
        n_samples_effect=("same_sign_as_pooled", "size"),
    ).reset_index()
    wide = wide.merge(consistency, on="gene", how="left")
    wide["sign_consistency"] = wide["sign_consistent_n"].astype("Int64").astype(str) + "/" + wide["n_samples_effect"].astype("Int64").astype(str)
    wide = wide.sort_values(["meta_q", "meta_log_or_mtrrna_vs_rrna"], ascending=[True, False])
    tsv_write(wide, outdir / "01_compartment_split_by_stage.tsv")
    return long, wide


def compartment_split_meta(ds: BackedDNAMIC, outdir: Path, rng: np.random.Generator, per_sample_n: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Matched rRNA-only versus MtrRNA-only rows, meta-analyzed by sample."""
    log.info("Running compartment split meta-analysis ...")
    pc_names = ds.var_names[ds.pc_idx]
    lor_list, var_list, sample_frames = [], [], []
    total_rrna = np.zeros(ds.pc_idx.size, np.int64)
    total_mtrna = np.zeros(ds.pc_idx.size, np.int64)
    n_rrna_total = n_mtrna_total = 0
    sample_qc: Dict[str, dict] = {}

    for sample in ds.sample_names:
        rows = ds.random_rows(sample, per_sample_n, rng)
        if rows.size == 0:
            continue
        x = ds.fetch_rows(rows)
        row_nnz = row_nnz_csr(x)
        umi = ds.umi[rows]
        has_pc = rows_with_any(x, ds.pc_idx)
        has_rrna = rows_with_any(x, ds.rrna_idx)
        has_mt = rows_with_any(x, ds.mtrna_idx)
        rrna_group = has_pc & has_rrna & ~has_mt
        mtrna_group = has_pc & has_mt & ~has_rrna
        rrna_idx, mtrna_idx = matched_case_control_indices(rrna_group, mtrna_group, umi, row_nnz, rng)
        if rrna_idx.size == 0 or mtrna_idx.size == 0:
            sample_qc[sample] = {
                "sampled_rows": int(rows.size),
                "raw_rrna_group": int(rrna_group.sum()),
                "raw_mtrrna_group": int(mtrna_group.sum()),
                "matched_rrna_group": 0,
                "matched_mtrrna_group": 0,
                "status": "insufficient_overlap_after_matching",
            }
            continue

        x_r = x[rrna_idx][:, ds.pc_idx]
        x_m = x[mtrna_idx][:, ds.pc_idx]
        count_r = np.asarray(x_r.sum(axis=0)).ravel().astype(np.int64)
        count_m = np.asarray(x_m.sum(axis=0)).ravel().astype(np.int64)
        lor, var = log_or_and_var(count_m, x_m.shape[0], count_r, x_r.shape[0])
        lor_list.append(lor)
        var_list.append(var)
        total_rrna += count_r
        total_mtrna += count_m
        n_rrna_total += x_r.shape[0]
        n_mtrna_total += x_m.shape[0]

        sample_frames.append(pd.DataFrame({
            "sample": sample,
            "sample_alias": _sample_alias(sample),
            "stage_group": _sample_stage(sample),
            "stage_label": STAGE_DISPLAY.get(_sample_stage(sample), _sample_stage(sample)),
            "gene": pc_names,
            "sample_log_or_mtrrna_vs_rrna": lor,
            "sample_se": np.sqrt(var),
            "sample_var": var,
            "matched_rrna_n": int(x_r.shape[0]),
            "matched_mtrrna_n": int(x_m.shape[0]),
            "rrna_count": count_r,
            "mtrrna_count": count_m,
            "matched_freq_rrna_group": count_r / max(1, x_r.shape[0]),
            "matched_freq_mtrrna_group": count_m / max(1, x_m.shape[0]),
        }))
        sample_qc[sample] = {
            "sampled_rows": int(rows.size),
            "raw_rrna_group": int(rrna_group.sum()),
            "raw_mtrrna_group": int(mtrna_group.sum()),
            "matched_rrna_group": int(x_r.shape[0]),
            "matched_mtrrna_group": int(x_m.shape[0]),
            "pct_rrna_retained": float(x_r.shape[0] / max(1, rrna_group.sum())),
            "pct_mtrrna_retained": float(x_m.shape[0] / max(1, mtrna_group.sum())),
            "status": "ok",
        }

    if not lor_list:
        raise RuntimeError("No samples produced matched rRNA/MtrRNA groups.")

    meta_lor, meta_var, meta_z, meta_p = fixed_effect_meta(lor_list, var_list)
    meta_q = bh_fdr(meta_p)
    sf_p, sf_mode, sf_patterns = sample_signflip_meta_p(lor_list, var_list)
    sf_q = bh_fdr(np.where(np.isfinite(sf_p), sf_p, 1.0))
    freq_r = total_rrna / max(1, n_rrna_total)
    freq_m = total_mtrna / max(1, n_mtrna_total)

    df = pd.DataFrame({
        "gene": pc_names,
        "meta_log_or_mtrrna_vs_rrna": meta_lor,
        "meta_se": np.sqrt(meta_var),
        "meta_z": meta_z,
        "meta_p": meta_p,
        "meta_q": meta_q,
        "meta_p_sample_signflip": sf_p,
        "meta_q_sample_signflip": sf_q,
        "matched_freq_rrna_group": freq_r,
        "matched_freq_mtrrna_group": freq_m,
        "log2_prevalence_ratio": np.log2((freq_m + 1 / max(1, n_mtrna_total)) / (freq_r + 1 / max(1, n_rrna_total))),
        "direction": np.where(meta_lor > 0, "MtrRNA_pref", np.where(meta_lor < 0, "rRNA_pref", "neutral")),
    }).sort_values(["meta_q", "meta_p", "meta_log_or_mtrrna_vs_rrna"], ascending=[True, True, False])
    tsv_write(df, outdir / "01_compartment_split_meta.tsv")

    per_sample = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    if not per_sample.empty:
        order = {g: i for i, g in enumerate(df["gene"].astype(str))}
        per_sample["gene_order"] = per_sample["gene"].astype(str).map(order).fillna(10**9).astype(int)
        per_sample = per_sample.sort_values(["gene_order", "sample_alias"]).drop(columns="gene_order")
    tsv_write(per_sample, outdir / "01_compartment_split_per_sample.tsv")
    _write_compartment_stage_tables(per_sample, df, outdir)

    summary = {
        "samples_used": list(sample_qc),
        "sample_qc": sample_qc,
        "n_matched_rrna_group": int(n_rrna_total),
        "n_matched_mtrrna_group": int(n_mtrna_total),
        "sample_signflip_mode": sf_mode,
        "sample_signflip_n_patterns": int(sf_patterns),
        "n_mtrrna_pref_q05": int(((df["meta_q"] < 0.05) & (df["meta_log_or_mtrrna_vs_rrna"] > 0)).sum()),
        "n_rrna_pref_q05": int(((df["meta_q"] < 0.05) & (df["meta_log_or_mtrrna_vs_rrna"] < 0)).sum()),
        "n_mtrrna_pref_q05_sample_signflip": int(((df["meta_q_sample_signflip"] < 0.05) & (df["meta_log_or_mtrrna_vs_rrna"] > 0)).sum()),
        "n_rrna_pref_q05_sample_signflip": int(((df["meta_q_sample_signflip"] < 0.05) & (df["meta_log_or_mtrrna_vs_rrna"] < 0)).sum()),
        "top_mtrrna_pref": df[df["meta_log_or_mtrrna_vs_rrna"] > 0].head(20)[["gene", "meta_log_or_mtrrna_vs_rrna", "matched_freq_mtrrna_group", "matched_freq_rrna_group", "meta_q", "meta_q_sample_signflip"]].to_dict("records"),
        "top_rrna_pref": df[df["meta_log_or_mtrrna_vs_rrna"] < 0].head(20)[["gene", "meta_log_or_mtrrna_vs_rrna", "matched_freq_mtrrna_group", "matched_freq_rrna_group", "meta_q", "meta_q_sample_signflip"]].to_dict("records"),
        "tables": {
            "meta": "01_compartment_split_meta.tsv",
            "per_sample": "01_compartment_split_per_sample.tsv",
            "by_stage_long": "01_compartment_split_by_stage_long.tsv",
            "by_stage_wide": "01_compartment_split_by_stage.tsv",
        },
    }
    save_json(summary, outdir / "01_compartment_split_meta.json")
    return df, summary


# -----------------------------------------------------------------------------
# 02: all-stage module coherence
# -----------------------------------------------------------------------------


def _jaccard_metric_record(stage: str, sample: str, context: str, metric: str, vals: np.ndarray, matched_rows: int) -> dict:
    finite = np.asarray(vals, float)
    finite = finite[np.isfinite(finite)]
    return {
        "stage_group": stage,
        "stage_label": STAGE_DISPLAY.get(stage, stage),
        "sample": sample,
        "sample_alias": _sample_alias(sample),
        "context": context,
        "metric": metric,
        "mean_jaccard": float(finite.mean()) if finite.size else np.nan,
        "n_values": int(finite.size),
        "matched_rows": int(matched_rows),
    }


def module_coherence_by_stage(ds: BackedDNAMIC, diff_df: pd.DataFrame, outdir: Path, rng: np.random.Generator, per_sample_n: int) -> Dict[str, object]:
    """Per-sample module-coherence effect sizes across developmental groups."""
    log.info("Running all-stage module coherence analyses ...")
    modules = select_module_genes(ds.var_names)
    gene_to_col = {g: i for i, g in enumerate(ds.var_names)}
    rrna_pref = set(diff_df.loc[(diff_df["meta_q"] < 0.05) & (diff_df["meta_log_or_mtrrna_vs_rrna"] < 0), "gene"].astype(str))
    mtrna_pref = set(diff_df.loc[(diff_df["meta_q"] < 0.05) & (diff_df["meta_log_or_mtrrna_vs_rrna"] > 0), "gene"].astype(str))
    per_sample: Dict[str, object] = {}
    records: List[dict] = []

    for stage, aliases in DEFAULT_STAGE_SAMPLES.items():
        for sample in ds.resolve_sample_names(aliases):
            rows = ds.random_rows(sample, per_sample_n, rng)
            x = ds.fetch_rows(rows)
            row_nnz = row_nnz_csr(x)
            umi = ds.umi[rows]
            has_pc = rows_with_any(x, ds.pc_idx)
            has_rrna = rows_with_any(x, ds.rrna_idx)
            has_mt = rows_with_any(x, ds.mtrna_idx)
            rrna_group = has_pc & has_rrna & ~has_mt
            mtrna_group = has_pc & has_mt & ~has_rrna
            rrna_idx, mtrna_idx = matched_case_control_indices(rrna_group, mtrna_group, umi, row_nnz, rng)
            if rrna_idx.size == 0 or mtrna_idx.size == 0:
                per_sample[sample] = {"stage_group": stage, "status": "insufficient_matches"}
                continue

            x_rrna = x[rrna_idx]
            x_mtrna = x[mtrna_idx]
            rrna_lipid = pairwise_jaccard_values(x_rrna, modules["lipid"], gene_to_col)
            rrna_ps = pairwise_jaccard_values(x_rrna, modules["protein_synthesis"], gene_to_col)
            rrna_cross = cross_jaccard_values(x_rrna, modules["lipid"], modules["protein_synthesis"], gene_to_col)
            mt_mt = pairwise_jaccard_values(x_mtrna, modules["mt_encoded"], gene_to_col)
            mt_nuc = pairwise_jaccard_values(x_mtrna, modules["nuclear_translation"], gene_to_col)
            mt_cross = cross_jaccard_values(x_mtrna, modules["mt_encoded"], modules["nuclear_translation"], gene_to_col)

            records.extend([
                _jaccard_metric_record(stage, sample, "rRNA", "lipid_within", rrna_lipid, x_rrna.shape[0]),
                _jaccard_metric_record(stage, sample, "rRNA", "protein_synthesis_within", rrna_ps, x_rrna.shape[0]),
                _jaccard_metric_record(stage, sample, "rRNA", "lipid_vs_protein_cross", rrna_cross, x_rrna.shape[0]),
                _jaccard_metric_record(stage, sample, "MtrRNA", "mt_encoded_within", mt_mt, x_mtrna.shape[0]),
                _jaccard_metric_record(stage, sample, "MtrRNA", "nuclear_translation_within", mt_nuc, x_mtrna.shape[0]),
                _jaccard_metric_record(stage, sample, "MtrRNA", "mt_vs_nuclear_cross", mt_cross, x_mtrna.shape[0]),
            ])
            per_sample[sample] = {
                "stage_group": stage,
                "matched_rrna_rows": int(x_rrna.shape[0]),
                "matched_mtrna_rows": int(x_mtrna.shape[0]),
                "rrna_context": {
                    "lipid_mean_jaccard": float(np.mean(rrna_lipid)) if rrna_lipid.size else None,
                    "protein_synthesis_mean_jaccard": float(np.mean(rrna_ps)) if rrna_ps.size else None,
                    "cross_mean_jaccard": float(np.mean(rrna_cross)) if rrna_cross.size else None,
                },
                "mtrna_context": {
                    "mt_encoded_mean_jaccard": float(np.mean(mt_mt)) if mt_mt.size else None,
                    "nuclear_translation_mean_jaccard": float(np.mean(mt_nuc)) if mt_nuc.size else None,
                    "cross_mean_jaccard": float(np.mean(mt_cross)) if mt_cross.size else None,
                },
                "status": "ok",
            }

    per_df = pd.DataFrame(records)
    tsv_write(per_df, outdir / "02_module_coherence_by_stage_per_sample.tsv")
    if per_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = per_df.groupby(["stage_group", "stage_label", "context", "metric"], as_index=False).agg(
            mean_jaccard_mean=("mean_jaccard", "mean"),
            mean_jaccard_min=("mean_jaccard", "min"),
            mean_jaccard_max=("mean_jaccard", "max"),
            n_samples=("mean_jaccard", lambda x: int(np.isfinite(np.asarray(x, float)).sum())),
            n_values_total=("n_values", "sum"),
        )
    tsv_write(summary_df, outdir / "02_module_coherence_by_stage.tsv")

    results = {
        "modules": modules,
        "volcano_overlap_not_used_for_module_definition": {
            "lipid_overlap_rrna_pref": sorted(set(modules["lipid"]) & rrna_pref),
            "mt_encoded_overlap_mtrrna_pref": sorted(set(modules["mt_encoded"]) & mtrna_pref),
        },
        "per_sample": per_sample,
        "stage_summary": summary_df.to_dict("records"),
        "tables": {
            "per_sample": "02_module_coherence_by_stage_per_sample.tsv",
            "stage_summary": "02_module_coherence_by_stage.tsv",
        },
    }
    save_json(results, outdir / "02_module_coherence_by_stage.json")
    return results


# -----------------------------------------------------------------------------
# 06/07: paired anchor neighborhoods
# -----------------------------------------------------------------------------


def matched_neighborhood_meta(
    ds: BackedDNAMIC,
    anchor_gene: str,
    samples: Sequence[str],
    per_sample_n: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Matched anchor-positive versus anchor-negative rows, meta-analyzed by sample."""
    gene_to_var = {g: i for i, g in enumerate(ds.var_names)}
    if anchor_gene not in gene_to_var:
        raise KeyError(anchor_gene)
    anchor_col = gene_to_var[anchor_gene]
    pc_names = ds.var_names[ds.pc_idx]
    lor_list, var_list = [], []
    total_case = np.zeros(ds.pc_idx.size, np.int64)
    total_ctrl = np.zeros(ds.pc_idx.size, np.int64)
    n_case_total = n_ctrl_total = 0
    sample_qc: Dict[str, dict] = {}

    for sample in samples:
        rows = ds.random_rows(sample, per_sample_n, rng)
        x = ds.fetch_rows(rows)
        row_nnz = row_nnz_csr(x)
        umi = ds.umi[rows]
        anchor_pos = np.asarray(x[:, anchor_col].sum(axis=1)).ravel() > 0
        case_idx, ctrl_idx = matched_case_control_indices(anchor_pos, ~anchor_pos, umi, row_nnz, rng)
        if case_idx.size == 0 or ctrl_idx.size == 0:
            sample_qc[sample] = {
                "sampled_rows": int(rows.size),
                "anchor_positive": int(anchor_pos.sum()),
                "matched_cases": 0,
                "matched_controls": 0,
                "status": "insufficient_matches",
            }
            continue

        x_case = x[case_idx][:, ds.pc_idx]
        x_ctrl = x[ctrl_idx][:, ds.pc_idx]
        count_case = np.asarray(x_case.sum(axis=0)).ravel().astype(np.int64)
        count_ctrl = np.asarray(x_ctrl.sum(axis=0)).ravel().astype(np.int64)
        lor, var = log_or_and_var(count_case, x_case.shape[0], count_ctrl, x_ctrl.shape[0])
        lor_list.append(lor)
        var_list.append(var)
        total_case += count_case
        total_ctrl += count_ctrl
        n_case_total += x_case.shape[0]
        n_ctrl_total += x_ctrl.shape[0]
        sample_qc[sample] = {
            "sampled_rows": int(rows.size),
            "anchor_positive": int(anchor_pos.sum()),
            "matched_cases": int(x_case.shape[0]),
            "matched_controls": int(x_ctrl.shape[0]),
            "status": "ok",
        }

    if not lor_list:
        raise RuntimeError(f"No matched case/control rows for anchor {anchor_gene}")
    meta_lor, meta_var, meta_z, meta_p = fixed_effect_meta(lor_list, var_list)
    meta_q = bh_fdr(meta_p)
    sf_p, sf_mode, sf_patterns = sample_signflip_meta_p(lor_list, var_list)
    sf_q = bh_fdr(np.where(np.isfinite(sf_p), sf_p, 1.0))

    df = pd.DataFrame({
        "gene": pc_names,
        "meta_log_or_anchor_pos_vs_neg": meta_lor,
        "meta_se": np.sqrt(meta_var),
        "meta_z": meta_z,
        "meta_p": meta_p,
        "meta_q": meta_q,
        "meta_p_sample_signflip": sf_p,
        "meta_q_sample_signflip": sf_q,
        "matched_freq_anchor_pos": total_case / max(1, n_case_total),
        "matched_freq_anchor_neg": total_ctrl / max(1, n_ctrl_total),
    }).sort_values(["meta_q", "meta_p", "meta_log_or_anchor_pos_vs_neg"], ascending=[True, True, False])

    summary = {
        "anchor_gene": anchor_gene,
        "sample_qc": sample_qc,
        "n_matched_anchor_pos": int(n_case_total),
        "n_matched_anchor_neg": int(n_ctrl_total),
        "sample_signflip_mode": sf_mode,
        "sample_signflip_n_patterns": int(sf_patterns),
    }
    return df, summary


def hbae3_neighborhood(ds: BackedDNAMIC, diff_df: pd.DataFrame, outdir: Path, rng: np.random.Generator, per_sample_n: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """hbae3-positive matched neighborhood table used in the paired anchor figure."""
    log.info("Running hbae3 matched-neighborhood analysis ...")
    df, summary = matched_neighborhood_meta(ds, "hbae3", ds.samples_for_stage("24hpf"), per_sample_n, rng)
    tsv_write(df, outdir / "06_hbae3_neighborhood.tsv")
    non_hb = df[(df["gene"] != "hbae3") & (~df["gene"].map(is_hemoglobin_gene))].copy()
    top_non_hb = non_hb.head(50)
    rrna_pref = set(diff_df.loc[(diff_df["meta_q"] < 0.05) & (diff_df["meta_log_or_mtrrna_vs_rrna"] < 0), "gene"].astype(str))
    summary.update({
        "top_non_hemoglobin_partners": top_non_hb.head(20)[["gene", "meta_log_or_anchor_pos_vs_neg", "meta_q", "meta_q_sample_signflip"]].to_dict("records"),
        "pct_ribosomal_in_top50_nonhb": float(100.0 * top_non_hb["gene"].map(is_ribosomal_gene).mean()) if not top_non_hb.empty else 0.0,
        "rrna_pref_overlap_top50_nonhb": sorted(rrna_pref & set(top_non_hb["gene"].astype(str))),
        "tables": {"neighborhood": "06_hbae3_neighborhood.tsv"},
    })
    save_json(summary, outdir / "06_hbae3_neighborhood.json")
    return df, summary


def ctslb_neighborhood(ds: BackedDNAMIC, outdir: Path, rng: np.random.Generator, per_sample_n: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """ctslb-positive matched neighborhood table used in the paired anchor figure."""
    log.info("Running ctslb matched-neighborhood analysis ...")
    df, summary = matched_neighborhood_meta(ds, "ctslb", ds.samples_for_stage("24hpf"), per_sample_n, rng)
    tsv_write(df, outdir / "07_ctslb_neighborhood.tsv")
    filtered = df[(df["gene"] != "ctslb") & (~df["gene"].map(is_housekeeping_partner))].copy()
    top_filtered = filtered.head(100)
    canonical = ["si:dkey-269i1.4", "zgc:174153", "zgc:174855", "zgc:158463"]
    summary.update({
        "top_filtered_partners": top_filtered.head(20)[["gene", "meta_log_or_anchor_pos_vs_neg", "meta_q", "meta_q_sample_signflip"]].to_dict("records"),
        "canonical_partner_hits": top_filtered[top_filtered["gene"].isin(canonical)][["gene", "meta_log_or_anchor_pos_vs_neg", "meta_q", "meta_q_sample_signflip"]].to_dict("records"),
        "tables": {"neighborhood": "07_ctslb_neighborhood.tsv"},
    })
    save_json(summary, outdir / "07_ctslb_neighborhood.json")
    return df, summary


# -----------------------------------------------------------------------------
# Figure helpers and requested plots
# -----------------------------------------------------------------------------


_PLOT_RC_APPLIED = False


def _plot_setup():
    """Lazy matplotlib setup with editable vector text and sans-serif defaults."""
    global _PLOT_RC_APPLIED
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import colors, font_manager as fm
    from matplotlib.lines import Line2D
    if not _PLOT_RC_APPLIED:
        if not any("arial" in f.name.lower() for f in fm.fontManager.ttflist):
            log.warning("Arial is unavailable; falling back to Liberation Sans / Helvetica / DejaVu Sans.")
        matplotlib.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.default": "regular",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        })
        _PLOT_RC_APPLIED = True
    return plt, colors, Line2D


def _figure_dir(outdir: Path) -> Path:
    d = outdir / FIGURE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_pub_figure(fig, outdir: Path, stem: str, title: str, caption: str, section: str) -> dict:
    figdir = _figure_dir(outdir)
    pdf_path = figdir / f"{stem}.pdf"
    png_path = figdir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass
    return {
        "section": section,
        "title": title,
        "caption": caption,
        "pdf": str(pdf_path.relative_to(outdir)),
        "png": str(png_path.relative_to(outdir)),
    }


def _safe_neglog10(vals: Sequence[float], minval: float = 1e-300) -> np.ndarray:
    return -np.log10(np.clip(np.asarray(vals, float), minval, 1.0))


def _spread_positions(values: Sequence[float], min_gap: float, lower: Optional[float] = None, upper: Optional[float] = None) -> np.ndarray:
    """Spread label positions on one axis while preserving order."""
    arr = np.asarray(values, float)
    if arr.size == 0:
        return arr
    order = np.argsort(arr)
    y = arr[order].copy()
    lower = float(np.min(y)) if lower is None else lower
    upper = float(np.max(y)) if upper is None else upper
    y[0] = max(y[0], lower)
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + min_gap)
    if y[-1] > upper:
        y[-1] = upper
        for i in range(len(y) - 2, -1, -1):
            y[i] = min(y[i], y[i + 1] - min_gap)
        if y[0] < lower:
            y += lower - y[0]
            if y[-1] > upper:
                y -= y[-1] - upper
    out = np.empty_like(y)
    out[order] = y
    return out


def _ordered_unique(items: Sequence[str]) -> List[str]:
    out, seen = [], set()
    for item in items:
        s = str(item)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _truncate(text: str, max_chars: int = 48) -> str:
    text = str(text)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "..."


def plot_compartment_split_figure(diff_df: pd.DataFrame, outdir: Path) -> Optional[dict]:
    """Write 01_compartment_split_volcano.{pdf,png}."""
    if diff_df.empty:
        return None
    plt, _, Line2D = _plot_setup()
    df = diff_df.copy()
    raw_q = df["meta_q"].to_numpy(float)
    df["neglog10q_display"] = -np.log10(np.clip(raw_q, DISPLAY_FLOOR_Q, 1.0))
    df["is_clipped"] = raw_q < DISPLAY_FLOOR_Q
    sig_m = (df["meta_q"] < 0.05) & (df["meta_log_or_mtrrna_vs_rrna"] > 0)
    sig_r = (df["meta_q"] < 0.05) & (df["meta_log_or_mtrrna_vs_rrna"] < 0)
    neutral = ~(sig_m | sig_r)

    fig = plt.figure(figsize=(8.8, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    axh = fig.add_subplot(gs[1, 0], sharex=ax)
    ax.tick_params(axis="x", labelbottom=False)

    for mask, color, size, alpha in [
        (neutral, OKABE_ITO["light_gray"], 14, 0.75),
        (sig_r, OKABE_ITO["blue"], 18, 0.92),
        (sig_m, OKABE_ITO["vermilion"], 18, 0.92),
    ]:
        ax.scatter(
            df.loc[mask, "meta_log_or_mtrrna_vs_rrna"],
            df.loc[mask, "neglog10q_display"],
            s=size,
            color=color,
            alpha=alpha,
            edgecolors="white" if mask is not neutral else "none",
            linewidths=0.2,
            rasterized=True,
        )
    ax.axhline(-math.log10(0.05), color=OKABE_ITO["dark_gray"], linestyle="--", lw=0.9)
    ax.axvline(0, color=OKABE_ITO["gray"], lw=0.8)
    ax.set_ylabel("-log10(display q; clipped at 1e-30)")
    ax.set_title("Matched compartment preference by protein-coding gene")
    xmin = min(-1.55, float(df["meta_log_or_mtrrna_vs_rrna"].min()) - 0.05)
    xmax = max(1.05, float(df["meta_log_or_mtrrna_vs_rrna"].max()) + 0.08)
    ymax = max(5.0, float(df["neglog10q_display"].max()) * 1.05)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)

    n_clip = int(df["is_clipped"].sum())
    if n_clip:
        clipped = df[df["is_clipped"]]
        ax.scatter(clipped["meta_log_or_mtrrna_vs_rrna"], np.full(n_clip, DISPLAY_FLOOR_NEGLOG10), marker="^", s=28, facecolor="none", edgecolor=OKABE_ITO["black"], lw=0.7, zorder=5, clip_on=False)
        ax.text(0.99, 0.98, f"^ {n_clip} genes at display floor", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=OKABE_ITO["dark_gray"], bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=OKABE_ITO["light_gray"]))

    left = df.loc[sig_r].sort_values(["meta_q", "meta_log_or_mtrrna_vs_rrna"], ascending=[True, True]).head(6)
    if not left.empty:
        ys = _spread_positions(left["neglog10q_display"], max(0.8, ymax * 0.06), lower=max(2.0, left["neglog10q_display"].min() - 2.0), upper=ymax * 0.92)
        for row, y_text in zip(left.itertuples(index=False), ys):
            ax.annotate(row.gene, xy=(row.meta_log_or_mtrrna_vs_rrna, row.neglog10q_display), xytext=(xmin + 0.16, y_text), ha="right", va="center", fontsize=8, color=OKABE_ITO["blue"], arrowprops=dict(arrowstyle="-", lw=0.5, color=OKABE_ITO["blue"]))

    right = df.loc[sig_m].sort_values(["meta_q", "meta_log_or_mtrrna_vs_rrna"], ascending=[True, False]).head(12)
    if not right.empty:
        right = right.assign(is_mt=right["gene"].str.lower().str.startswith("mt-"))
        label_df = pd.concat([right[right["is_mt"]], right[~right["is_mt"]].head(2)])
        ys = _spread_positions(label_df["neglog10q_display"], max(0.55, ymax * 0.035), lower=max(1.5, label_df["neglog10q_display"].min() - 1.0), upper=ymax * (0.78 if n_clip else 0.92))
        for row, y_text in zip(label_df.itertuples(index=False), ys):
            ax.annotate(row.gene, xy=(row.meta_log_or_mtrrna_vs_rrna, row.neglog10q_display), xytext=(xmax - 0.05, y_text), ha="right", va="center", fontsize=8, color=OKABE_ITO["vermilion"], arrowprops=dict(arrowstyle="-", lw=0.5, color=OKABE_ITO["vermilion"]))

    bins = np.linspace(df["meta_log_or_mtrrna_vs_rrna"].min() - 0.05, df["meta_log_or_mtrrna_vs_rrna"].max() + 0.05, 60)
    axh.hist(df["meta_log_or_mtrrna_vs_rrna"], bins=bins, color=OKABE_ITO["light_gray"], edgecolor="white")
    axh.hist(df.loc[sig_r, "meta_log_or_mtrrna_vs_rrna"], bins=bins, histtype="step", lw=1.1, color=OKABE_ITO["blue"])
    axh.hist(df.loc[sig_m, "meta_log_or_mtrrna_vs_rrna"], bins=bins, histtype="step", lw=1.1, color=OKABE_ITO["vermilion"])
    axh.axvline(0, color=OKABE_ITO["gray"], lw=0.8)
    axh.set_xlabel("Preference for MtrRNA-conditioned rows (meta log-odds)")
    axh.set_ylabel("Genes")

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=OKABE_ITO["blue"], markeredgecolor="none", label="rRNA-preferring q < 0.05"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=OKABE_ITO["vermilion"], markeredgecolor="none", label="MtrRNA-preferring q < 0.05"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=OKABE_ITO["light_gray"], markeredgecolor="none", label="Not significant"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3)
    return _save_pub_figure(
        fig,
        outdir,
        "01_compartment_split_volcano",
        "Compartment preference volcano",
        "Protein-coding genes are plotted by fixed-effect matched log-odds and meta q. q-values below 1e-30 are clipped for display only and marked with carets.",
        "01",
    )


def plot_compartment_stage_profiles(diff_df: pd.DataFrame, outdir: Path) -> Optional[dict]:
    """Write 01_compartment_split_stage_profiles.{pdf,png}."""
    per_path = outdir / "01_compartment_split_per_sample.tsv"
    stage_path = outdir / "01_compartment_split_by_stage.tsv"
    if not per_path.exists() or not stage_path.exists():
        return None
    per = pd.read_csv(per_path, sep="\t")
    stage = pd.read_csv(stage_path, sep="\t")
    if per.empty or stage.empty:
        return None
    plt, _, _ = _plot_setup()

    manual = ["mt-co1", "mt-co3", "mt-co2", "mt-atp6", "mt-nd6", "mt-cyb", "apoa1a", "afp4", "apoa1b", "tfa", "bhmt", "fabp1b.1", "zgc:158463"]
    top_m = diff_df.loc[diff_df["meta_log_or_mtrrna_vs_rrna"] > 0].sort_values("meta_q").head(10)["gene"].astype(str).tolist()
    top_r = diff_df.loc[diff_df["meta_log_or_mtrrna_vs_rrna"] < 0].sort_values("meta_q").head(12)["gene"].astype(str).tolist()
    selected = [g for g in _ordered_unique(top_m + manual + top_r) if g in set(per["gene"].astype(str))][:28]
    if not selected:
        return None

    samples = sorted(per["sample"].drop_duplicates().astype(str), key=_sample_order_key)
    mat = per[per["gene"].astype(str).isin(selected)].pivot_table(index="gene", columns="sample", values="sample_log_or_mtrrna_vs_rrna", aggfunc="mean")
    row_order = [g for g in selected if g in mat.index]
    vals = mat.reindex(row_order)[samples].to_numpy(float)
    vlim = max(float(np.nanpercentile(np.abs(vals), 95)) if np.isfinite(vals).any() else 1.0, 0.5)

    fig = plt.figure(figsize=(11.2, max(5.2, 0.23 * len(row_order) + 2.6)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(np.ma.masked_invalid(vals), aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax.set_xticks(np.arange(len(samples)))
    ax.set_xticklabels([_sample_alias(s) for s in samples])
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Selected genes")
    ax.set_title("Per-sample compartment effect")
    fig.colorbar(im, ax=ax, fraction=0.048, pad=0.02).set_label("sample log OR (MtrRNA vs rRNA)")

    boundaries, runs, last, start = [], [], None, 0
    for i, sample in enumerate(samples):
        st = _sample_stage(sample)
        if last is None:
            last, start = st, i
        elif st != last:
            boundaries.append(i - 0.5)
            runs.append((last, start, i - 1))
            last, start = st, i
    if last is not None:
        runs.append((last, start, len(samples) - 1))
    for b in boundaries:
        ax.axvline(b, color="white", lw=1.2)
    for st, a, b in runs:
        ax.text((a + b) / 2, -1.15, STAGE_DISPLAY.get(st, st), ha="center", va="bottom", fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    x = stage["meta_log_or_mtrrna_vs_rrna"].to_numpy(float)
    y = stage["delta_log_or_24hpf_minus_12hpf"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    ax2.scatter(x[ok], y[ok], s=6, color=OKABE_ITO["light_gray"], alpha=0.35, linewidth=0)
    sub = stage[stage["gene"].astype(str).isin(row_order)].copy()
    sx = sub["meta_log_or_mtrrna_vs_rrna"].to_numpy(float)
    sy = sub["delta_log_or_24hpf_minus_12hpf"].to_numpy(float)
    ax2.scatter(sx, sy, s=26, c=np.where(sx >= 0, OKABE_ITO["vermilion"], OKABE_ITO["blue"]), edgecolor="white", lw=0.5, zorder=3)
    label_sub = sub.reindex(sub["delta_log_or_24hpf_minus_12hpf"].abs().sort_values(ascending=False).index).head(10)
    for row in label_sub.itertuples(index=False):
        xv, yv = float(row.meta_log_or_mtrrna_vs_rrna), float(row.delta_log_or_24hpf_minus_12hpf)
        if np.isfinite(xv) and np.isfinite(yv):
            ax2.text(xv + 0.02, yv, str(row.gene), fontsize=8, va="center", ha="left")
    ax2.axhline(0, color=OKABE_ITO["gray"], lw=0.8, linestyle="--")
    ax2.axvline(0, color=OKABE_ITO["gray"], lw=0.8)
    ax2.set_xlabel("pooled log OR (MtrRNA vs rRNA)")
    ax2.set_ylabel("delta log OR, 24 hpf - 12 hpf")
    ax2.set_title("Developmental effect-size change")
    fig.suptitle("Compartment preference decomposed into sample and developmental effects", y=0.995)
    return _save_pub_figure(
        fig,
        outdir,
        "01_compartment_split_stage_profiles",
        "Stage-resolved compartment preference profiles",
        "Per-sample log-odds are shown for selected compartment-preferring genes, with a companion 24 hpf minus 12 hpf effect-size diagnostic.",
        "01b",
    )


def plot_module_coherence_by_stage_figure(results: Mapping[str, object], outdir: Path) -> Optional[dict]:
    """Write 02_module_coherence_by_stage.{pdf,png}."""
    rec_path = outdir / "02_module_coherence_by_stage_per_sample.tsv"
    if not rec_path.exists():
        return None
    df = pd.read_csv(rec_path, sep="\t")
    if df.empty:
        return None
    plt, _, Line2D = _plot_setup()
    order = list(DEFAULT_STAGE_SAMPLES)
    xmap = {st: i for i, st in enumerate(order)}
    scale = 1e3
    specs = {
        "rRNA": [
            ("lipid_within", "lipid within", OKABE_ITO["orange"], "o"),
            ("protein_synthesis_within", "protein synthesis within", OKABE_ITO["blue"], "s"),
            ("lipid_vs_protein_cross", "cross-module", OKABE_ITO["gray"], "^"),
        ],
        "MtrRNA": [
            ("mt_encoded_within", "mt-encoded within", OKABE_ITO["vermilion"], "o"),
            ("nuclear_translation_within", "nuclear translation within", OKABE_ITO["purple"], "s"),
            ("mt_vs_nuclear_cross", "cross-module", OKABE_ITO["gray"], "^"),
        ],
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    fig.subplots_adjust(top=0.82, bottom=0.20, wspace=0.32)
    for ax, context in zip(axes, ["rRNA", "MtrRNA"]):
        context_df = df[df["context"] == context]
        for metric, label, color, marker in specs[context]:
            sub = context_df[context_df["metric"] == metric]
            xs, means = [], []
            for st in order:
                vals = sub.loc[sub["stage_group"] == st, "mean_jaccard"].to_numpy(float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    xs.append(xmap[st])
                    means.append(float(vals.mean()) * scale)
                    jitter = np.linspace(-0.055, 0.055, vals.size) if vals.size > 1 else np.array([0.0])
                    ax.scatter(xmap[st] + jitter, vals * scale, s=34, color=color, marker=marker, edgecolor="white", lw=0.5, zorder=3, alpha=0.95)
            if xs:
                ax.plot(xs, means, color=color, marker=marker, lw=1.4, label=label)
        ax.set_xticks([xmap[st] for st in order])
        ax.set_xticklabels([STAGE_DISPLAY.get(st, st).replace(" ", "\n", 1) for st in order])
        ax.set_xlabel("Developmental group")
        ax.set_title(f"{context}-conditioned rows")
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.26), frameon=False)
    axes[0].set_ylabel("Mean pairwise Jaccard (x1e-3)")
    fig.suptitle("Module coherence per sample across developmental groups", y=0.995)
    return _save_pub_figure(
        fig,
        outdir,
        "02_module_coherence_by_stage",
        "All-stage module coherence in RNA-conditioned rows",
        "Dots show biological samples and lines show group means. Module definitions are fixed from gene names rather than selected from the volcano.",
        "02",
    )


def _anchor_display_subset(df: pd.DataFrame, partner_filter, n_each_side: int = 8) -> pd.DataFrame:
    plot_df = df[partner_filter(df)].copy()
    if plot_df.empty:
        return plot_df
    pos = plot_df[plot_df["meta_log_or_anchor_pos_vs_neg"] >= 0].sort_values(["meta_log_or_anchor_pos_vs_neg", "meta_q"], ascending=[False, True]).head(n_each_side)
    neg = plot_df[plot_df["meta_log_or_anchor_pos_vs_neg"] < 0].sort_values(["meta_log_or_anchor_pos_vs_neg", "meta_q"], ascending=[True, True]).head(n_each_side)
    return pd.concat([neg, pos]).drop_duplicates("gene").sort_values("meta_log_or_anchor_pos_vs_neg")


def plot_anchor_neighborhood_pair_figure(
    hb_df: pd.DataFrame,
    hb_summary: Mapping[str, object],
    ct_df: pd.DataFrame,
    ct_summary: Mapping[str, object],
    outdir: Path,
) -> Optional[dict]:
    """Write 06_07_anchor_neighborhood_pair.{pdf,png}."""
    if hb_df.empty or ct_df.empty:
        return None
    plt, colors, _ = _plot_setup()
    panels = [
        {
            "df": _anchor_display_subset(hb_df, lambda d: (d["gene"] != "hbae3") & (~d["gene"].map(is_hemoglobin_gene))),
            "title": "hbae3 anchor: non-Hb partners",
            "xlabel": "log OR with hbae3-positive rows",
            "bar": "#9CC9EB",
            "stars": set(hb_summary.get("rrna_pref_overlap_top50_nonhb", [])),
        },
        {
            "df": _anchor_display_subset(ct_df, lambda d: (d["gene"] != "ctslb") & (~d["gene"].map(is_housekeeping_partner))),
            "title": "ctslb anchor: protease partners",
            "xlabel": "log OR with ctslb-positive rows",
            "bar": "#E7C46A",
            "stars": {x["gene"] for x in ct_summary.get("canonical_partner_hits", []) if isinstance(x, Mapping) and "gene" in x},
        },
    ]
    if any(p["df"].empty for p in panels):
        return None

    all_neglog = np.concatenate([_safe_neglog10(p["df"]["meta_q"].to_numpy(float), DISPLAY_FLOOR_Q) for p in panels])
    vmax = float(max(4.0, np.nanmax(all_neglog)))
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    cmap = colors.LinearSegmentedColormap.from_list("anchor_pair_sig", ["#E8EEF5", "#88A6C1", "#2F5877", "#17384F"])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharex=False)
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.43, right=0.90)
    last_sc = None
    for ax, panel in zip(axes, panels):
        d = panel["df"]
        ypos = np.arange(d.shape[0])
        xvals = d["meta_log_or_anchor_pos_vs_neg"].to_numpy(float)
        neglogq = _safe_neglog10(d["meta_q"].to_numpy(float), DISPLAY_FLOOR_Q)
        sizes = 32.0 + 16.0 * (neglogq / vmax)
        ax.barh(ypos, xvals, color=panel["bar"], alpha=0.74, edgecolor=OKABE_ITO["black"], lw=0.4, height=0.70, zorder=1)
        last_sc = ax.scatter(xvals, ypos, c=neglogq, cmap=cmap, norm=norm, s=sizes, edgecolor="white", lw=0.6, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(d["gene"].astype(str))
        ax.axvline(0.0, color=OKABE_ITO["dark_gray"], lw=0.8)
        ax.set_title(panel["title"])
        ax.set_xlabel(panel["xlabel"])
        if panel["stars"]:
            xpad = max(0.08, max(np.nanmax(np.abs(xvals)), 0.5) * 0.08)
            for yi, gene, xv in zip(ypos, d["gene"].astype(str), xvals):
                if gene in panel["stars"]:
                    ax.scatter(xv + (xpad if xv >= 0 else -xpad), yi, marker="*", s=95, facecolor="white", edgecolor=OKABE_ITO["black"], lw=0.9, clip_on=False, zorder=4)
        ax.set_xlim(float(min(0.0, np.nanmin(xvals))) - 0.30, float(max(0.0, np.nanmax(xvals))) + 0.40)

    if last_sc is not None:
        cax = fig.add_axes([0.925, 0.22, 0.018, 0.55])
        fig.colorbar(last_sc, cax=cax).set_label("-log10(display q)")
    fig.text(0.5, 0.93, "Matched 24 hpf anchor-neighborhood controls", ha="center", va="top", fontsize=12)
    fig.text(0.5, 0.885, "hbae3 gives a weak non-hemoglobin neighborhood; ctslb recovers a protease-associated cDNA-hub neighborhood", ha="center", va="top", fontsize=9)
    return _save_pub_figure(
        fig,
        outdir,
        "06_07_anchor_neighborhood_pair",
        "Paired hbae3 and ctslb anchor-neighborhood supplement",
        "The paired display compares matched 24 hpf anchor-positive neighborhoods. Statistics are fixed-effect matched meta-analysis values; sample-level sign-flip sensitivity is written in the source tables.",
        "06_07",
    )


# -----------------------------------------------------------------------------
# Main: only requested output sets
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-output-set non-spatial DNAMIC cDNA pipeline")
    parser.add_argument("--h5ad", required=True, help="Path to consolidated cDNA h5ad")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--sample-n", type=int, default=250_000, help="Per-sample rows for matched analyses")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _figure_dir(outdir)
    rng = np.random.default_rng(args.seed)
    ds = BackedDNAMIC(Path(args.h5ad))
    figure_index: List[dict] = []

    input_metadata = {
        "h5ad": str(Path(args.h5ad)),
        "n_obs": int(ds.n_obs),
        "n_vars": int(ds.n_vars),
        "sample_counts": {info.name: int(info.n_rows) for info in ds.sample_info},
        "protein_coding_features": int(ds.pc_idx.size),
        "rrna_features": int(ds.rrna_idx.size),
        "mtrrna_features": int(ds.mtrna_idx.size),
        "voxel_index_column_detected": ds.voxel_index_col,
        "uses_voxel_index_for_inference": False,
    }

    diff_df, diff_summary = compartment_split_meta(ds, outdir, rng, args.sample_n)
    figs01 = [f for f in [plot_compartment_split_figure(diff_df, outdir), plot_compartment_stage_profiles(diff_df, outdir)] if f]
    diff_summary["figures"] = figs01
    figure_index.extend(figs01)
    save_json(diff_summary, outdir / "01_compartment_split_meta.json")

    coh = module_coherence_by_stage(ds, diff_df, outdir, rng, args.sample_n)
    fig02 = plot_module_coherence_by_stage_figure(coh, outdir)
    if fig02:
        coh["figures"] = [fig02]
        figure_index.append(fig02)
    save_json(coh, outdir / "02_module_coherence_by_stage.json")

    hb_df, hb_summary = hbae3_neighborhood(ds, diff_df, outdir, rng, args.sample_n)
    ct_df, ct_summary = ctslb_neighborhood(ds, outdir, rng, args.sample_n)
    fig067 = plot_anchor_neighborhood_pair_figure(hb_df, hb_summary, ct_df, ct_summary, outdir)
    if fig067:
        hb_summary["figures"] = [fig067]
        ct_summary["figures"] = [fig067]
        figure_index.append(fig067)
    save_json(hb_summary, outdir / "06_hbae3_neighborhood.json")
    save_json(ct_summary, outdir / "07_ctslb_neighborhood.json")

    final_note = {
        "completed": True,
        "spatial_proxy_used": False,
        "input_metadata": input_metadata,
        "requested_output_sets_only": True,
        "figure_index": figure_index,
        "output_sets": {
            "01_compartment_split_volcano": ["figures/01_compartment_split_volcano.pdf", "figures/01_compartment_split_volcano.png", "01_compartment_split_meta.tsv", "01_compartment_split_meta.json"],
            "01_compartment_split_stage_profiles": ["figures/01_compartment_split_stage_profiles.pdf", "figures/01_compartment_split_stage_profiles.png", "01_compartment_split_per_sample.tsv", "01_compartment_split_by_stage_long.tsv", "01_compartment_split_by_stage.tsv"],
            "02_module_coherence_by_stage": ["figures/02_module_coherence_by_stage.pdf", "figures/02_module_coherence_by_stage.png", "02_module_coherence_by_stage_per_sample.tsv", "02_module_coherence_by_stage.tsv", "02_module_coherence_by_stage.json"],
            "06_07_anchor_neighborhood_pair": ["figures/06_07_anchor_neighborhood_pair.pdf", "figures/06_07_anchor_neighborhood_pair.png", "06_hbae3_neighborhood.tsv", "06_hbae3_neighborhood.json", "07_ctslb_neighborhood.tsv", "07_ctslb_neighborhood.json"],
        },
        "removed_tangents": [
            "dataset overview figure",
            "exact triplet/quadruplet mining and diagnostics",
            "developmental pairwise rewiring",
            "targeted exact-k module checks",
            "individual 06 and 07 anchor figures",
            "module x compartment overlap tests",
            "optional GMT/ORA outputs",
        ],
    }
    save_json(final_note, outdir / "99_run_summary.json")
    log.info("Done. Results written to %s", outdir.resolve())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
plot.py

Benchmark a reconstructed UEI/UMI embedding against ground-truth coordinates encoded in
raw cDNA-insert sequences (base-4 encoded in A/C/G/T).

This script is designed to keep the original simulation/benchmarking workflow working even
after pipeline refactors that:
  * move/reindex final_labels/final_coords, and/or
  * store cDNA-insert sequences inside an AnnData (.h5ad) one-hot encoding.

It will try, in order:
  1) Read sequences from an .h5ad inside --group_dir (preferred for patched pipelines).
  2) Fall back to decoding sequences from final_labels.txt (attr_8) and coordinates from
     final_coords.txt inside --group_dir.

If neither yields enough valid decoded points, it raises a RuntimeError with diagnostics.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import procrustes


# ----------------------------
# Utilities
# ----------------------------

_BASE4_RE = re.compile(r"^[ACGTNacgtn]+$")


def _strip_prefix_casefold(s: str, prefixes: Sequence[str]) -> str:
    """Remove the first matching prefix (case-insensitive)."""
    s_stripped = s.strip()
    s_cf = s_stripped.casefold()
    for p in prefixes:
        p_cf = p.casefold()
        if s_cf.startswith(p_cf):
            return s_stripped[len(p) :].strip()
    return s_stripped


def _split_semicolon_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s or s.upper() in {"NA", "NAN", "NONE"}:
        return []
    return [tok.strip() for tok in s.split(";") if tok.strip()]


def decode_indices_vectorized(
    tokens: np.ndarray,
    *,
    digits: str = "ACGT",
    strict: bool = True,
) -> np.ndarray:
    """
    Decode base-4 strings (A/C/G/T) into integer indices.

    If strict=True: any token containing characters outside digits is invalid => -1.

    Notes:
      * This is intentionally strict for benchmarking: it will NOT attempt to "extract"
        an ACGT substring from an arbitrary string (to avoid decoding gene names, FQNs, etc.).
      * Tokens must be comprised only of A/C/G/T (optionally after prefix stripping upstream).
    """
    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1D array of strings")

    toks = tokens.astype(str)
    n = toks.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return out

    # Normalize
    toks = np.char.upper(np.char.strip(toks))

    # Reject empty / NA-like
    na_mask = (toks == "") | (toks == "NA") | (toks == "NAN") | (toks == "NONE")
    if np.all(na_mask):
        return out

    # Length check (require all non-NA tokens same length in strict mode)
    lengths = np.fromiter((len(t) for t in toks.tolist()), dtype=np.int64, count=n)
    max_len = int(lengths.max())
    if max_len <= 0:
        return out

    if strict:
        # If any non-NA token is shorter than max_len, mark invalid
        len_ok = (lengths == max_len) | na_mask
    else:
        len_ok = ~na_mask

    # Build a (n, max_len) char matrix efficiently via frombuffer
    padded = [t.ljust(max_len) for t in toks.tolist()]
    joined = "".join(padded).encode("ascii", errors="ignore")
    chars = np.frombuffer(joined, dtype="S1").reshape(n, max_len)

    # Map chars to 0..3
    mat = np.full((n, max_len), -1, dtype=np.int16)
    mapping = {digits[i].encode(): i for i in range(len(digits))}
    for bch, val in mapping.items():
        mat[chars == bch] = val

    if strict:
        valid = (~na_mask) & len_ok & np.all(mat >= 0, axis=1)
    else:
        valid = (~na_mask) & len_ok

    # Weighted sum base-4
    powers = (4 ** np.arange(max_len - 1, -1, -1)).astype(np.int64)
    mat64 = mat.astype(np.int64)
    mat64[mat64 < 0] = 0
    out[valid] = (mat64[valid] * powers).sum(axis=1)

    return out


def _similarity_transform(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    allow_reflection: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute a similarity transform mapping src -> dst in dst units.

    Returns (scale, R, t) such that:
        dst ≈ scale * src @ R + t

    Unlike scipy.spatial.procrustes, this preserves the physical scale of dst.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape:
        raise ValueError(f"src and dst must have same shape; got {src.shape} vs {dst.shape}")
    if src.ndim != 2 or src.shape[0] < 2:
        raise ValueError("src/dst must be (n_points, n_dims) with n_points>=2")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src0 = src - mu_src
    dst0 = dst - mu_dst

    var_src = float(np.sum(src0 ** 2))
    if not np.isfinite(var_src) or var_src <= 0.0:
        raise RuntimeError("Degenerate transform: source points have zero (or non-finite) variance")

    H = src0.T @ dst0
    U, S, Vt = np.linalg.svd(H, full_matrices=True)
    R = U @ Vt
    if (not allow_reflection) and (np.linalg.det(R) < 0):
        Vt[-1, :] *= -1.0
        R = U @ Vt

    scale = float(np.sum(S) / var_src)
    t = mu_dst - (scale * (mu_src @ R))
    return scale, R, t
 

def _load_gse_coords(gse_path: str, n_obs: int) -> np.ndarray:
    """Load (possibly sparse-indexed) GSEoutput coords into an (n_obs, d) array."""
    raw = np.loadtxt(gse_path, delimiter=",")
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] < 3:
        raise ValueError(f"{gse_path} must have at least 3 columns: idx, x, y")

    idx = raw[:, 0].astype(np.int64)
    coords = raw[:, 1:]

    d = coords.shape[1]
    out = np.full((n_obs, d), np.nan, dtype=float)
    valid = (idx >= 0) & (idx < n_obs)
    out[idx[valid]] = coords[valid]
    return out


def _pick_first_file(patterns: Sequence[str]) -> Optional[str]:
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


# ----------------------------
# Reading sequences
# ----------------------------

def _infer_seq_mask_from_var_names(
    var_names: np.ndarray,
    *,
    seq_prefixes: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infer which var_names represent sequence features.

    Returns:
      mask: boolean mask over var_names
      stripped: var_names stripped of recognized prefixes
    """
    var_str = var_names.astype(str)
    stripped = np.array([_strip_prefix_casefold(v, seq_prefixes) for v in var_str], dtype=str)

    # Candidate sequences are those that look like A/C/G/T/N only after stripping prefixes
    looks_like_seq = np.array([bool(_BASE4_RE.match(s)) for s in stripped], dtype=bool)

    # Also allow explicit prefix match even if it contains other chars after prefix removal
    pref_match = np.array([v.casefold().startswith(tuple(p.casefold() for p in seq_prefixes)) for v in var_str], dtype=bool)

    mask = looks_like_seq | pref_match
    return mask, stripped


def _read_sequences_from_h5ad(
    h5ad_path: str,
    *,
    seq_layer: str = "seq",
    seq_prefixes: Sequence[str] = ("SEQ:", "seq:", "SEQUENCE:", "sequence:", "SEQ_", "seq_"),
) -> List[List[str]]:
    """
    Extract per-row sequence tokens from an .h5ad.

    Supported storage layouts:
      - adata.obs['seq_str'] (semicolon-separated sequences)
      - a sparse CSR matrix in layers[seq_layer], with sequence features identifiable from var_names
      - if layers[seq_layer] missing, falls back to X (same logic)

    The function is tolerant to whether sequence features are named:
      - "SEQ:ACGT..."
      - "seq:ACGT..."
      - "ACGT..." (no prefix)
    """
    # Try anndata first if available
    try:
        import anndata as ad  # type: ignore

        adata = ad.read_h5ad(h5ad_path)

        # Case 1: seq_str in obs (preferred if present)
        if "seq_str" in adata.obs.columns:
            return [_split_semicolon_list(str(s)) for s in adata.obs["seq_str"].astype(str).tolist()]

        # Select matrix
        if seq_layer in adata.layers:
            mat = adata.layers[seq_layer]
        else:
            mat = adata.X

        # var_names and mask
        var_names = np.asarray(adata.var_names)
        mask, stripped = _infer_seq_mask_from_var_names(var_names, seq_prefixes=seq_prefixes)

        # If anndata provides feature_type and it marks sequences, trust it
        if "feature_type" in adata.var.columns:
            ft = adata.var["feature_type"].astype(str).str.casefold().to_numpy()
            ft_mask = ft == "sequence"
            if ft_mask.any():
                mask = ft_mask
                stripped = np.array(
                    [_strip_prefix_casefold(v, seq_prefixes) for v in var_names.astype(str)],
                    dtype=str,
                )

        # Build per-row token lists
        try:
            # Ensure CSR for fast row slicing
            from scipy.sparse import csr_matrix  # type: ignore

            mat = csr_matrix(mat)
            indptr, indices = mat.indptr, mat.indices
        except Exception:
            raise RuntimeError("Expected a sparse matrix (CSR-compatible) for sequence one-hot layer/X")

        out: List[List[str]] = []
        for i in range(mat.shape[0]):
            cols = indices[indptr[i] : indptr[i + 1]]
            cols = cols[mask[cols]]
            toks = [stripped[j] for j in cols]
            out.append([t for t in toks if t])
        return out

    except ModuleNotFoundError:
        # Fall back to minimal H5AD reading with h5py
        import h5py  # type: ignore

        with h5py.File(h5ad_path, "r") as f:
            # obs seq_str
            if "obs" in f and "seq_str" in f["obs"]:
                seq_str = f["obs/seq_str"][()]
                if isinstance(seq_str, np.ndarray):
                    seq_str = [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in seq_str.tolist()]
                else:
                    seq_str = [str(seq_str)]
                return [_split_semicolon_list(s) for s in seq_str]

            # var names
            if "var" not in f or "_index" not in f["var"]:
                raise RuntimeError(f"{h5ad_path}: missing var/_index; cannot infer sequence features.")
            var_raw = f["var/_index"][()]
            var_names = np.array(
                [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in var_raw.tolist()],
                dtype=str,
            )
            mask, stripped = _infer_seq_mask_from_var_names(var_names, seq_prefixes=seq_prefixes)

            # Choose matrix group: layers/seq_layer or X
            mat_grp_path = f"layers/{seq_layer}"
            if mat_grp_path in f:
                grp = f[mat_grp_path]
            elif "X" in f:
                grp = f["X"]
            else:
                raise RuntimeError(f"{h5ad_path}: neither layers/{seq_layer} nor X found; cannot read sequences.")

            # Expect CSR layout
            required = {"data", "indices", "indptr", "shape"}
            if not required.issubset(set(grp.keys())):
                raise RuntimeError(
                    f"{h5ad_path}: {mat_grp_path if mat_grp_path in f else 'X'} is missing CSR datasets {required}."
                )
            indices = grp["indices"][()].astype(np.int64)
            indptr = grp["indptr"][()].astype(np.int64)
            shape = tuple(grp["shape"][()].astype(np.int64).tolist())
            n_obs = int(shape[0])

            out: List[List[str]] = []
            for i in range(n_obs):
                cols = indices[indptr[i] : indptr[i + 1]]
                cols = cols[mask[cols]]
                toks = [stripped[j] for j in cols]
                out.append([t for t in toks if t])
            return out


def _read_sequences_from_final_labels(final_labels_path: str) -> List[List[str]]:
    """
    Extract per-row sequence tokens from final_labels.txt.

    The patched optimOps schema is:
      col0 point_type
      col1 true_raw
      col2..col10 attr_1..attr_9

    For the simulation benchmarking, the base4 cDNA-insert strings are typically stored in attr_8
    (column index 9, 0-based) as a semicolon-separated list.

    We still auto-detect a "sequence-like" column if that assumption doesn't hold.
    """
    import csv

    rows: List[List[str]] = []
    with open(final_labels_path, "r", newline="") as fh:
        reader = csv.reader(fh)
        for r in reader:
            rows.append([c.strip() for c in r])

    if not rows:
        return []

    n_cols = max(len(r) for r in rows)

    # Helper: count how many rows in a column look like base4 tokens
    def score_col(j: int) -> int:
        score = 0
        for r in rows:
            if j >= len(r):
                continue
            toks = _split_semicolon_list(r[j])
            if any(_BASE4_RE.match(t) and set(t.upper()) <= set("ACGTN") for t in toks):
                score += 1
        return score

    # Prefer attr_8 (col 9) if present and has signal
    preferred = 9 if n_cols > 9 else None
    best_j = preferred if preferred is not None else 0
    best_score = score_col(best_j)

    if best_score == 0:
        # Auto-detect among all columns
        for j in range(n_cols):
            s = score_col(j)
            if s > best_score:
                best_score, best_j = s, j

    # Parse sequences from chosen column
    out: List[List[str]] = []
    for r in rows:
        val = r[best_j] if best_j < len(r) else ""
        out.append(_split_semicolon_list(val))
    return out


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Plot and benchmark inferred embedding vs ground truth (base4 cDNA inserts).")
    ap.add_argument("--group_dir", default=os.path.join("sim_fastq", "uei", "uei_grp0"), help="UEI group directory.")
    ap.add_argument("--pos", default="pos.csv", help="Ground-truth positions CSV.")
    ap.add_argument("--out", default="reconstruction.png", help="Output plot PNG.")
    ap.add_argument("--h5ad", default=None, help="Explicit .h5ad path (overrides auto-detect in group_dir).")
    ap.add_argument(
        "--seq_layer",
        default="seq",
        help="Layer name in .h5ad containing sequence one-hot. Falls back to X if missing.",
    )
    ap.add_argument(
        "--seq_prefixes",
        default="SEQ:,seq:,SEQUENCE:,sequence:,SEQ_,seq_",
        help="Comma-separated list of prefixes to strip/detect for sequence features.",
    )
    args = ap.parse_args()

    group_dir = args.group_dir.rstrip("/")

    # Load pos.csv (expects numeric columns; x,y in cols 2 and 3)
    pos_data = np.loadtxt(args.pos, delimiter=",")
    if pos_data.ndim == 1:
        pos_data = pos_data[None, :]
    if pos_data.shape[1] < 4:
        raise RuntimeError(f"{args.pos} must have >= 4 numeric columns; expected x,y in columns 3-4 (0-based 2-3).")

    seq_prefixes = tuple(p for p in (s.strip() for s in args.seq_prefixes.split(",")) if p)

    # Try H5AD path
    h5ad_path = args.h5ad
    if h5ad_path is None:
        h5ad_path = _pick_first_file([os.path.join(group_dir, "*.h5ad")])

    # Determine coords + sequences
    seqs_by_row: List[List[str]] = []
    coords_by_row: Optional[np.ndarray] = None
    used_source = None

    if h5ad_path is not None and os.path.exists(h5ad_path):
        seqs_by_row = _read_sequences_from_h5ad(
            h5ad_path,
            seq_layer=args.seq_layer,
            seq_prefixes=seq_prefixes,
        )
        n_obs = len(seqs_by_row)

        gse_path = os.path.join(group_dir, "GSEoutput.txt")
        if os.path.exists(gse_path):
            coords_by_row = _load_gse_coords(gse_path, n_obs=n_obs)
        else:
            # If we don't have GSEoutput, attempt to fall back to final_coords (but row order may differ).
            final_coords_path = os.path.join(group_dir, "final_coords.txt")
            if os.path.exists(final_coords_path):
                coords_by_row = np.loadtxt(final_coords_path, delimiter=",")[:, :2]
            else:
                coords_by_row = None

        used_source = f"h5ad:{os.path.basename(h5ad_path)}"

    # Expand row coordinates to one point per sequence token
    def expand(coords: np.ndarray, seqs: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        expanded_coords: List[np.ndarray] = []
        expanded_seqs: List[str] = []
        for i, toks in enumerate(seqs):
            if not toks:
                continue
            for t in toks:
                expanded_coords.append(coords[i])
                expanded_seqs.append(_strip_prefix_casefold(t, seq_prefixes))
        if not expanded_coords:
            return np.empty((0, 2), float), np.array([], dtype=str)
        return np.vstack(expanded_coords), np.array(expanded_seqs, dtype=str)

    if coords_by_row is not None and len(seqs_by_row) == coords_by_row.shape[0]:
        expanded_coords, expanded_seqs = expand(coords_by_row, seqs_by_row)
    else:
        expanded_coords, expanded_seqs = np.empty((0, 2), float), np.array([], dtype=str)

    decoded = decode_indices_vectorized(expanded_seqs, digits="ACGT", strict=True) if expanded_seqs.size else np.array([], dtype=np.int64)

    # Filter valid decoded points
    valid_mask = (
        (decoded >= 0)
        & (decoded < pos_data.shape[0])
        & np.isfinite(expanded_coords).all(axis=1)
    )

    # If insufficient valid points, fall back to final_labels/final_coords
    if valid_mask.sum() < 2:
        final_labels_path = os.path.join(group_dir, "final_labels.txt")
        final_coords_path = os.path.join(group_dir, "final_coords.txt")
        if os.path.exists(final_labels_path) and os.path.exists(final_coords_path):
            seqs_by_row = _read_sequences_from_final_labels(final_labels_path)
            coords_by_row = np.loadtxt(final_coords_path, delimiter=",")[:, :2]
            used_source = "final_labels/final_coords"

            expanded_coords, expanded_seqs = expand(coords_by_row, seqs_by_row)
            decoded = decode_indices_vectorized(expanded_seqs, digits="ACGT", strict=True) if expanded_seqs.size else np.array([], dtype=np.int64)
            valid_mask = (
                (decoded >= 0)
                & (decoded < pos_data.shape[0])
                & np.isfinite(expanded_coords).all(axis=1)
            )

    n_valid = int(valid_mask.sum())
    if n_valid < 2:
        # Diagnostics to help pinpoint why decoding failed
        diag = [
            f"Sequence source tried: {used_source!r}",
            f"Found h5ad: {h5ad_path!r}" if h5ad_path else "No h5ad found",
            f"Total expanded tokens: {expanded_seqs.size}",
            f"Non-empty tokens: {int(np.sum(expanded_seqs != '')) if expanded_seqs.size else 0}",
            f"Valid decoded indices: {n_valid}",
        ]
        # Show a few example tokens for debugging
        examples = [t for t in expanded_seqs[:20].tolist()] if expanded_seqs.size else []
        if examples:
            diag.append("Example tokens (first 20): " + ", ".join(examples))
        raise RuntimeError(
            "Not enough valid decoded points to run Procrustes: "
            f"{n_valid} valid points.\n\n"
            + "\n".join(diag)
            + "\n\n"
            "Likely causes:\n"
            "  * sequence features in the one-hot .h5ad are not named like ACGT strings (or are missing), OR\n"
            "  * plot.py didn't detect the sequence columns due to naming/prefix differences.\n"
            "Try inspecting the .h5ad var_names, or ensure final_labels.txt contains base4 insert sequences."
        )

    valid_coords = expanded_coords[valid_mask]
    valid_indices = decoded[valid_mask].astype(np.int64)
    pos_coords = pos_data[valid_indices, 2:4]

    # Procrustes disparity is scale/translation invariant. Note: procrustes() returns
    # unit-norm coordinates (mtx1/mtx2), which are NOT in pos/reconstruction units.
    _, _, disparity = procrustes(pos_coords, valid_coords)

    # Compute a similarity transform that maps inferred coords -> ground truth coords
    # in pos.csv units (so axes match the simulation coordinates).
    sim_scale, sim_R, sim_t = _similarity_transform(valid_coords, pos_coords, allow_reflection=True)
    aligned = (valid_coords @ sim_R) * sim_scale + sim_t

    rmse = float(np.sqrt(np.mean(np.sum((aligned - pos_coords) ** 2, axis=1))))
 
    # Plot
    import matplotlib.pyplot as plt

    # Ground-truth X values (pos.csv x coordinate) for each decoded point
    gt_x = pos_coords[:, 0]

    # Use global ground-truth x-range for consistent color scaling
    # (pos_data columns 2/3 are x/y in this benchmark.)
    vmin = float(np.nanmin(pos_data[:, 2]))
    vmax = float(np.nanmax(pos_data[:, 2]))

    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(13.5, 6), sharex=True, sharey=True)


    sc_gt = ax_gt.scatter(
        pos_coords[:, 0],
        pos_coords[:, 1],
        c=gt_x,
        cmap="viridis",
        s=8,
        alpha=0.85,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    ax_gt.set_title("Ground truth (decoded)\ncolor = ground truth X")
    ax_gt.set_xlabel("X (pos units)")
    ax_gt.set_ylabel("Y (pos units)")

    sc_pred = ax_pred.scatter(
        aligned[:, 0],
        aligned[:, 1],
        c=gt_x,
        cmap="viridis",
        s=8,
        alpha=0.85,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    ax_pred.set_title("Inferred (aligned)\ncolor = ground truth X")
    ax_pred.set_xlabel("X (pos units)")
    ax_pred.set_ylabel("")
    ax_pred.tick_params(labelleft=False)

    # Match axis limits between panels for a true scale comparison
    x_min = float(np.nanmin([np.nanmin(pos_coords[:, 0]), np.nanmin(aligned[:, 0])]))
    x_max = float(np.nanmax([np.nanmax(pos_coords[:, 0]), np.nanmax(aligned[:, 0])]))
    y_min = float(np.nanmin([np.nanmin(pos_coords[:, 1]), np.nanmin(aligned[:, 1])]))
    y_max = float(np.nanmax([np.nanmax(pos_coords[:, 1]), np.nanmax(aligned[:, 1])]))
    pad = 0.02 * max(x_max - x_min, y_max - y_min)

    for ax in (ax_gt, ax_pred):
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25, linestyle="--")

    # Reserve room on the right for the colorbar
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.10, top=0.84, wspace=0.08)

    # Put the colorbar in its own axes so it never overlaps the plots
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.68])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(sc_pred, cax=cbar_ax)
    cbar.set_label("Ground truth X (pos.csv col 3)")

    fig.suptitle(
        f"Procrustes disparity={disparity:.4g} | RMSE={rmse:.4g} (pos units)\n"
        f"n_valid={n_valid}, source={used_source}",
        y=0.96,
    )
    fig.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")
    print(f"Procrustes disparity: {disparity:.6g} using {n_valid} decoded points (source={used_source}).")
    detR = float(np.linalg.det(sim_R))
    print(f"Similarity transform: scale={sim_scale:.6g}, det(R)={detR:.6g}, RMSE(pos units)={rmse:.6g}")

if __name__ == "__main__":
    main()

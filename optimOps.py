from __future__ import annotations
from threads_bootstrap import NTHREADS
import numpy as np
import scipy
from annoy import AnnoyIndex
import sysOps
import os
import faiss
import pymetis
from numpy import linalg as LA
from scipy.sparse.linalg import ArpackNoConvergence, ArpackError  # eigsh unused (krylov solver below)
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from numba import jit, njit, types, prange, float64, int64
import json
from joblib import Parallel, delayed
import pandas as pd
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import math
from scipy.sparse.csgraph import connected_components

# Final post-embedding clustering layout/version.
_FINAL_CLUSTER_LABELS_LAYOUT_VERSION = 2
_FINAL_CLUSTER_MIN_CLUSTER_SIZE = 100
_FINAL_HDBSCAN_MIN_SAMPLES = 50

_FILTER_EDGE_ZSCORE = 6.0
_FILTER_DISTANCE_CHUNK = 2_000_000


def min_contig_edges(index_link_array, dataset_index_array, link_data, Nassoc):
    # Function is used for single-linkage clustering of pts (to identify which sets are contiguous and which are not)
    # Inputs:
    #    1. index_link_array: indices for individual pts
    #    2. dataset_index_array: belonging to the same set is a requirement for two pts to be examined for linkage -- subsets of the data that have different values in dataset_index_array will not be merged

    min_index_links_changed = 1  # initiate flag to enter while-loop

    while min_index_links_changed > 0:
        min_index_links_changed = 0

        # Extract link pairs and their dataset indices
        link0 = link_data[:, 0].astype(int)
        link1 = link_data[:, 1].astype(int)
        dataset0 = dataset_index_array[link0]
        dataset1 = dataset_index_array[link1]

        # Determine valid links where datasets match
        valid_links = (dataset0 == dataset1)

        # Update index_link_array where needed
        changes_0_to_1 = (index_link_array[link0] > index_link_array[link1]) & valid_links
        changes_1_to_0 = (index_link_array[link1] > index_link_array[link0]) & valid_links

        if np.any(changes_0_to_1):
            index_link_array[link0[changes_0_to_1]] = index_link_array[link1[changes_0_to_1]]
            min_index_links_changed += np.sum(changes_0_to_1)

        if np.any(changes_1_to_0):
            index_link_array[link1[changes_1_to_0]] = index_link_array[link0[changes_1_to_0]]
            min_index_links_changed += np.sum(changes_1_to_0)

    return


@njit(parallel=True)
def parallel_dot(u, v):
    """
    Returns dot(u, v) in parallel via prange reduction.
    Numba automatically accumulates partial sums from each thread.
    """
    local_sum = 0.0
    for i in prange(u.size):
        local_sum += u[i] * v[i]
    return local_sum

@njit(parallel=True)
def parallel_norm(u):
    """
    Returns the Euclidean norm of u in parallel.
    """
    local_sum = 0.0
    for i in prange(u.size):
        val = u[i]
        local_sum += val * val
    return math.sqrt(local_sum)

@njit(parallel=True)
def parallel_scale(u, alpha):
    """
    In-place scaling: u[:] *= alpha, done in parallel.
    """
    for i in prange(u.size):
        u[i] *= alpha

@njit(parallel=True)
def parallel_axpy(y, alpha, x):
    """
    In-place AXPY: y[:] -= alpha * x[:], done in parallel.
    """
    for i in prange(y.size):
        y[i] -= alpha * x[i]

def orth_preserve_order(M):
    M = np.asfortranarray(M)
    mgs_inplace_parallel(M)
    return M

@njit
def mgs_inplace_parallel(M):
    """
    Orthonormalizes the columns of M in-place via one pass of 
    (modified) Gram–Schmidt in parallel over rows.

    Args:
        M: A 2D numpy array of shape (n, k).
           Usually n >> k, e.g. n ~ 1e7 and k ~ 100.

    Procedure:
      for i in 0..(k-1):
        1) norm_i = || col i ||
           if norm_i < 1e-15 -> set col i to a degenerate vector
        2) col i /= norm_i
        3) for j in (i+1)..(k-1):
             dot_ij = dot(col i, col j)
             col j -= dot_ij * col i
    
    NOTE: If you need re-orthonormalization (e.g. near-linear-dependency),
          you can call `mgs_inplace_parallel(M)` again in a loop.
    """
    n, k = M.shape
    for i in range(k):
        # 1) Compute norm of column i
        col_i = M[:, i]
        norm_i = parallel_norm(col_i)
        if norm_i < 1e-15:
            # Degenerate column => set to [1,0,0...]
            for r in range(n):
                col_i[r] = 0.0
            if n > 0:
                col_i[0] = 1.0
            norm_i = 1.0

        # 2) Normalize column i
        inv_norm_i = 1.0 / norm_i
        parallel_scale(col_i, inv_norm_i)

        # 3) Orthogonalize subsequent columns w.r.t. col_i
        for j in range(i+1, k):
            col_j = M[:, j]
            dot_ij = parallel_dot(col_i, col_j)
            # col_j -= dot_ij * col_i
            parallel_axpy(col_j, dot_ij, col_i)


def _coarsen_param_is_enabled(value):
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return True
        return any(v not in (None, "", False) for v in value)
    if isinstance(value, str):
        return value != ""
    return True


def _reject_removed_coarsen_params(params):
    removed = sorted(
        str(k)
        for k, v in params.items()
        if str(k).startswith('-coarsen') and _coarsen_param_is_enabled(v)
    )
    if removed:
        raise ValueError(
            "The coarsen/Infomap bootstrap pipeline has been removed. Unsupported parameters: "
            + ", ".join(removed)
        )


def run_GSE(output_name, params, coarsen=True):
    # `coarsen` is retained only for API compatibility; the coarsen/Infomap
    # bootstrap route has been removed.
    if type(params['-inference_eignum']) == list:
        fill_params(params)
    _reject_removed_coarsen_params(params)

    # When run_GSE() is called programmatically (already-parsed params), some legacy
    # keys may be absent. Set minimal defaults here without touching caller-provided
    # values. (fill_params() handles the CLI/list case.)
    params.setdefault('-final_eignum', params.get('-final_eignum', 100))
    params.setdefault('-calc_final', params.get('-calc_final', None))
    params.setdefault('-filter', params.get('-filter', None))
    inference_eignum = int(params['-inference_eignum'])
    inference_dim = int(params['-inference_dim'])
    # '-scales' is a legacy multi-outer-iter knob (default: 1)
    params.setdefault('-scales', 1)
    worker_processes = NTHREADS

    # Freeze label_root derived from -calc_final before any downstream path rewrites.
    # -calc_final is treated as a label directory path (often relative to the *parent*
    # of the run directory).
    if params.get('-calc_final') is not None and ('_h5ad_label_root' not in params):
        _cf = str(params.get('-calc_final') or "").strip()
        if _cf == "":
            params['_h5ad_label_root'] = None
        else:
            _base = os.path.dirname(str(params['-path']).rstrip(os.sep))
            params['_h5ad_label_root'] = _cf if os.path.isabs(_cf) else os.path.abspath(os.path.join(_base, _cf))
    # Expose also via sysOps so downstream helpers (h5ad builder) can find it.
    sysOps.h5ad_label_root = params.get('_h5ad_label_root', None)

    sysOps.globaldatapath = str(params['-path'])
    # Default: auto. If '-h5ad_include_sequences' is present, always include.
    # Otherwise, include sequences only when label_pt appears STAR-less and contains sequences.
    sysOps.h5ad_include_nonunique_genes = ('-h5ad_include_nonunique_genes' in params)
    sysOps.h5ad_include_sequences = True if ('-h5ad_include_sequences' in params) else None

    sysOps.num_workers = worker_processes
    sysOps.throw_status("params = " + str(params))

    this_GSEobj = GSEobj(inference_dim, inference_eignum)

    if this_GSEobj.seq_evecs is not None:
        del this_GSEobj.seq_evecs
    this_GSEobj.seq_evecs = None

    if not sysOps.check_file_exists("orig_evecs.npy"):
        sysOps.throw_status("this_GSEobj.link_data.data.shape = " + str(this_GSEobj.link_data.data.shape))
        this_GSEobj.inference_eignum = int(1 * inference_eignum)
        this_GSEobj.eigen_decomp(orth=False, pmax=0)
        this_GSEobj.inference_eignum = int(this_GSEobj.inference_eignum / 1)
        os.rename(sysOps.globaldatapath + "evecs.npy", sysOps.globaldatapath + "orig_evecs.npy")
        os.rename(sysOps.globaldatapath + "evals.npy", sysOps.globaldatapath + "orig_evals.npy")
    else:
        this_GSEobj.seq_evecs = np.load(sysOps.globaldatapath + "orig_evecs.npy").T

    if not sysOps.check_file_exists("orig_evecs_gapnorm.npy"):
        Y = generalized_eigen_embedding(
            this_GSEobj.seq_evecs.T,
            this_GSEobj.link_data + this_GSEobj.link_data.T,
            r=this_GSEobj.inference_eignum,
        )
        np.save(sysOps.globaldatapath + "orig_evecs_gapnorm.npy", Y)

    del this_GSEobj

    full_GSE(output_name, params)

    if params['-calc_final'] is not None and not sysOps.check_file_exists("cluster_labels.npy"):
        # ------------------------------------------------------------------
        # Final post-embedding clustering
        #
        # This stage is intentionally isolated from the main GSE solve.
        # We keep the original HDBSCAN parameterization on FINAL coords, and
        # replace the separate hybrid_cluster dependency with several runs of
        # the updated execute_clusters_infomap() on the FINAL-embedding
        # transport graph.
        #
        # The output is saved as cluster_labels.npy with shape (N, 5):
        #   col0 = hdbscan labels (final coords; min_cluster_size=10, min_samples=10)
        #   col1 = infomap_final_default
        #   col2 = infomap_final_min10
        #   col3 = infomap_final_min10_frac90_k50
        #   col4 = infomap_final_min10_mt2
        #
        # NOTE: col2 is intentionally retained as an explicit fixed-min10
        # benchmark column, even though it currently matches the default.
        #
        # NOTE: keeping >4 columns is intentional so _build_augmented_h5ad()
        # falls back to generic cluster_0.. names instead of stale hybrid-
        # specific labels.
        # ------------------------------------------------------------------
        coords_df = pd.read_csv(sysOps.globaldatapath + str(output_name), header=None)
        Xpts_final = coords_df.iloc[:, 1:].values.astype(np.float64, copy=False)

        # Load the (possibly directed/bipartite) link graph used for HDBSCAN curation
        link_path = os.path.join(sysOps.globaldatapath, "link_assoc_reindexed.npz")
        link_csr = load_npz(link_path).tocsr()

        cl_path = os.path.join(sysOps.globaldatapath, "cluster_labels.npy")
        cl_meta_path = os.path.join(sysOps.globaldatapath, "cluster_labels_meta.json")

        ex_mat = None
        ex_meta = None
        if os.path.exists(cl_path):
            try:
                existing = np.asarray(np.load(cl_path))
                if existing.ndim == 1:
                    ex_mat = existing.reshape(-1, 1)
                elif existing.ndim >= 2:
                    ex_mat = existing.reshape(existing.shape[0], -1)
                if ex_mat is not None and ex_mat.shape[0] != Xpts_final.shape[0]:
                    ex_mat = None
            except Exception:
                ex_mat = None
        if os.path.exists(cl_meta_path):
            try:
                with open(cl_meta_path, "r") as fh:
                    ex_meta = json.load(fh)
            except Exception:
                ex_meta = None
        # Only reuse cached final labels when the sidecar metadata says they
        # match the current layout/version.

        final_infomap_variants = [
            ("default", {
                "min_cluster_size": _FINAL_CLUSTER_MIN_CLUSTER_SIZE,
            }),
        ]
        desired_meta = {
            "layout_version": int(_FINAL_CLUSTER_LABELS_LAYOUT_VERSION),
            "hdbscan_min_cluster_size": int(_FINAL_CLUSTER_MIN_CLUSTER_SIZE),
            "hdbscan_min_samples": int(_FINAL_HDBSCAN_MIN_SAMPLES),
            "infomap_variants": [
                {"name": str(name), **variant_kwargs}
                for name, variant_kwargs in final_infomap_variants
            ],
        }
        expected_label_cols = 1 + len(final_infomap_variants)
        reuse_final_clusters = (
            (ex_mat is not None)
            and (ex_mat.shape[1] == expected_label_cols)
            and (ex_meta is not None)
            and (int(ex_meta.get("layout_version", -1)) == int(_FINAL_CLUSTER_LABELS_LAYOUT_VERSION))
        )
        need_hdbscan = not reuse_final_clusters
        need_any_final_infomap = not reuse_final_clusters

        from clusterplot import execute_clusters_hdbscan, execute_clusters_infomap

        if need_hdbscan:
            sysOps.throw_status(
                f"Running final HDBSCAN clustering "
                f"(min_cluster_size={_FINAL_CLUSTER_MIN_CLUSTER_SIZE}, "
                f"min_samples={_FINAL_HDBSCAN_MIN_SAMPLES})."
            )
            labels_hdbscan = execute_clusters_hdbscan(
                Xpts_final,
                link_csr,
                min_cluster_size=_FINAL_CLUSTER_MIN_CLUSTER_SIZE,
                min_samples=_FINAL_HDBSCAN_MIN_SAMPLES,
            ).astype(np.int32, copy=False)
        else:
            labels_hdbscan = ex_mat[:, 0].astype(np.int32, copy=False)

        # Build transformed_matrix from FINAL embedding once; reuse it across all
        # final Infomap parameterizations.
        base = os.path.splitext(os.path.basename(str(output_name)))[0]
        tm_final_name = f"transformed_matrix_final_{base}.npz"
        tm_final_path = os.path.join(sysOps.globaldatapath, tm_final_name)

        if need_any_final_infomap:
            if not os.path.exists(tm_final_path):
                sysOps.throw_status(
                    "Building transformed matrix from FINAL embedding: " + tm_final_name
                )
                sym_link = (link_csr + link_csr.T).tocsr()
                kneighbors = max(
                    2 * int(params.get('-inference_eignum', 30)),
                    10 * int(params.get('-inference_dim', Xpts_final.shape[1])),
                )
                _build_transformed_matrix_from_coords(
                    sym_link_csr=sym_link,
                    coords=Xpts_final,
                    kneighbors=int(kneighbors),
                    out_npz_path=tm_final_path,
                    workers=int(getattr(sysOps, "num_workers", NTHREADS)),
                )

        num_threads = int(getattr(sysOps, "num_workers", NTHREADS))
        final_infomap_cols = []
        for col_idx, (variant_name, variant_kwargs) in enumerate(final_infomap_variants, start=1):
            if reuse_final_clusters:
                labels_variant = ex_mat[:, col_idx].astype(np.int32, copy=False)
            else:
                sysOps.throw_status(
                    "Running final Infomap variant '" + str(variant_name) +
                    "' on transformed matrix derived from FINAL embedding."
                )
                infomap_kwargs = dict(
                    transformed_matrix_path=tm_final_path,
                    out_dir=os.path.join(sysOps.globaldatapath, "tmp", f"infomap_tm_final_{variant_name}"),
                    out_name=f"tm_modules_final_{variant_name}_{base}",
                    seed=1,
                    num_trials=10,
                    silent=True,
                    num_threads=num_threads,
                )
                infomap_kwargs.update(variant_kwargs)
                labels_variant = execute_clusters_infomap(**infomap_kwargs).astype(np.int32, copy=False)
            final_infomap_cols.append(labels_variant)

        # Combine: hdbscan + multiple FINAL-transport Infomap parameterizations.
        cluster_labels = np.column_stack([labels_hdbscan] + final_infomap_cols).astype(np.int32, copy=False)
        np.save(cl_path, cluster_labels)
        try:
            with open(cl_meta_path, "w") as fh:
                json.dump(desired_meta, fh, indent=2, sort_keys=True)
        except Exception as e:
            sysOps.throw_status("Warning: could not write cluster_labels_meta.json: " + str(e))

    if params['-calc_final'] is not None:
        sysOps.throw_status("Calculating final : " + str(sysOps.h5ad_label_root))
        # Defer building the AnnData/h5ad until the very end so it can include
        # downstream results (GSEoutput + optional cluster labels) and to avoid
        # paying the IO cost during every GSEobj instantiation.
        if os.path.isdir(sysOps.h5ad_label_root):
            _build_augmented_h5ad(
                group_path=sysOps.globaldatapath,
                gse_output_name=str(output_name),
            )

    if params.get('-filter'):
        if output_name is None:
            raise ValueError("-filter requires an output_name/GSEoutput file.")
        sysOps.throw_status("-filter enabled: exporting filtered subgraph and launching child rerun.")
        filtered_path = _export_filtered_subgraph_from_gse(
            base_path=sysOps.globaldatapath,
            output_name=str(output_name),
        )
        child_params = dict(params)
        child_params['-path'] = filtered_path
        child_params['-filter'] = None
        if params.get('_h5ad_label_root') is not None:
            child_params['_h5ad_label_root'] = params['_h5ad_label_root']
        run_GSE(output_name, child_params, coarsen=coarsen)


def _build_augmented_h5ad(
    group_path: str,
    *,
    h5ad_filename: str = "final.h5ad",
    gse_output_name: str | None = "GSEoutput.txt",
    cluster_labels_name: str = "cluster_labels.npy",
    binary: bool = False,
) -> None:
    """Build final.h5ad at the end of run_GSE and attach downstream results.

    This is intentionally separated from GSEobj.load_data() so:
      - expensive IO does not happen during every GSEobj instantiation
      - the h5ad can include GSEoutput coordinates and (optionally) cluster labels

    Adds (when available):
      - adata.obs['GSE_1'..] from <gse_output_name> (columns 2..end, float32)
      - adata.obs['cluster'] (legacy 1D) OR method-specific cluster columns from <cluster_labels_name> (int32)


    The function is best-effort: missing optional files will be skipped.
    """
    try:
        group_path = str(group_path)
        h5ad_out = os.path.join(group_path, h5ad_filename)
        gse_path = os.path.join(group_path, gse_output_name) if gse_output_name else None
        cl_path = os.path.join(group_path, cluster_labels_name)
        index_key_path = os.path.join(group_path, "index_key.npy")

        # Ensure a local subset mapping exists before any freshness check / upward search.
        if not os.path.exists(index_key_path):
            try:
                _restore_missing_subset_index_key(group_path)
            except Exception as e:
                sysOps.throw_status("Warning: could not restore missing index_key.npy: " + str(e))

        # Rebuild only if missing or stale w.r.t downstream outputs.
        if os.path.exists(h5ad_out):
            try:
                h5_mtime = os.path.getmtime(h5ad_out)
                newest_src = h5_mtime
                for p in (gse_path, cl_path, index_key_path):
                    if p and os.path.exists(p):
                        newest_src = max(newest_src, os.path.getmtime(p))
                if newest_src <= h5_mtime:
                    sysOps.throw_status("Found up-to-date " + h5ad_out + "; skipping h5ad rebuild.")
                    return
            except Exception:
                # If mtime checks fail, fall through and rebuild.
                pass

        from annotation import build_umi_gene_anndata, _find_upwards, _looks_like_dna_seq_list

        include_sequences_cfg = getattr(sysOps, "h5ad_include_sequences", False)
        include_sequences = bool(include_sequences_cfg) if include_sequences_cfg is not None else None
        if include_sequences is None:
            # Auto: include sequences only when label_pt appears STAR-less and contains sequences.
            label_root = getattr(sysOps, "h5ad_label_root", None)
            search_root = label_root if label_root is not None else group_path

            # Locate label_pt paths the same way build_umi_gene_anndata() does.
            label_pt_paths = []
            for _amp in (0, 1):
                fn = f"label_pt{_amp}.txt"
                lp = _find_upwards(search_root, fn)
                if lp is None:
                    lp = _find_upwards(group_path, fn)
                if lp:
                    label_pt_paths.append(lp)

            def _has_nonneg_start(start_field: str) -> bool:
                if not start_field:
                    return False
                placeholders = {"NA", "N/A", "NONE", "None"}
                for sub in start_field.split(";"):
                    sub = sub.strip()
                    if not sub:
                        continue
                    for tok in sub.split("|"):
                        tok = tok.strip()
                        if (not tok) or (tok == "-1"):
                            continue
                        if tok in placeholders or tok.upper() in placeholders:
                            continue
                        try:
                            if int(tok) >= 0:
                                return True
                        except ValueError:
                            continue
                return False

            seq_like = False
            # Detect STAR having run via STARalignment*/ directories (more robust than sampling aln_start values).
            star_seen = False
            for _amp in (0, 1):
                if os.path.isdir(os.path.join(search_root, f"STARalignment{_amp}")):
                    star_seen = True
                    break

            n_probe = 0
            for lp in label_pt_paths:
                try:
                    with open(lp, "r") as _f:
                        for _line in _f:
                            _line = _line.strip()
                            if not _line:
                                continue
                            _fields = _line.split(",")
                            if len(_fields) > 8 and _looks_like_dna_seq_list(_fields[8].strip()):
                                seq_like = True
                            if len(_fields) > 1 and _has_nonneg_start(_fields[1].strip()):
                                star_seen = True
                            n_probe += 1
                            if (seq_like and star_seen) or (n_probe >= 2000):
                                break
                except OSError:
                    continue
                if (seq_like and star_seen) or (n_probe >= 2000):
                    break

            include_sequences = bool(seq_like and (not star_seen))
        include_nonunique_genes = bool(getattr(sysOps, "h5ad_include_nonunique_genes", False))
        label_root = getattr(sysOps, "h5ad_label_root", None)

        sysOps.throw_status("Building AnnData ...")
        adata = build_umi_gene_anndata(
            group_path=group_path,
            label_root=label_root,
            return_anndata=True,
            include_sequences=include_sequences,
            include_nonunique_genes=include_nonunique_genes,
            binary=bool(binary),
        )

        # If anndata isn't installed, builder can return a tuple instead of AnnData.
        # Treat that as a hard failure here; otherwise resume can appear "successful"
        # while silently leaving no usable final.h5ad behind.

        if not hasattr(adata, "obs") or not hasattr(adata, "write_h5ad"):
            raise TypeError(
                "build_umi_gene_anndata did not return an AnnData object; got "
                + str(type(adata))
            )

        # -------------------------
        # Attach GSE coordinates
        # -------------------------
        if gse_path and os.path.exists(gse_path):
            try:
                # Determine number of columns from the first line (cheap)
                with open(gse_path, "r") as f:
                    first = f.readline()
                ncols = len(first.rstrip().split(",")) if first else 0
                if ncols >= 2:
                    usecols = tuple(range(1, ncols))  # skip node_index col
                    X_gse = np.loadtxt(gse_path, delimiter=",", dtype=np.float32, usecols=usecols)
                    if X_gse.ndim == 1:
                        X_gse = X_gse.reshape(-1, 1)

                    if X_gse.shape[0] != adata.n_obs:
                        sysOps.throw_status("Warning: GSE coordinate array size does not match AnnData, mapping via explicit node_index column.")
                        full = np.loadtxt(gse_path, delimiter=",", dtype=np.float32)
                        idx = full[:, 0].astype(np.int64)
                        coords = full[:, 1:].astype(np.float32)
                        X_gse = np.full((adata.n_obs, coords.shape[1]), np.nan, dtype=np.float32)
                        mask = (idx >= 0) & (idx < adata.n_obs)
                        X_gse[idx[mask]] = coords[mask]

                    gse_cols = [f"GSE_{i+1}" for i in range(X_gse.shape[1])]
                    for i, col in enumerate(gse_cols):
                        adata.obs[col] = X_gse[:, i]
                    adata.obsm["X_gse"] = np.asarray(X_gse, dtype=np.float32)
                    adata.uns["GSEoutput_source"] = os.path.basename(gse_path)
            except Exception as e:
                sysOps.throw_status("ERROR: could not attach GSEoutput to AnnData: " + str(e))
                raise

        # -------------------------
        # Attach cluster labels
        # -------------------------
        if os.path.exists(cl_path):
            try:
                cl_raw = np.asarray(np.load(cl_path))
                if cl_raw.ndim == 0:
                    raise ValueError("cluster_labels.npy contained a scalar; expected (N,) or (N,K)")

                # Normalize to 2D: (n_points, n_clusterings)
                if cl_raw.ndim == 1:
                    cl_mat = cl_raw.reshape(-1, 1)
                elif cl_raw.ndim == 2:
                    cl_mat = cl_raw

                else:
                    # Best-effort: preserve first axis as points, flatten the rest.
                    cl_mat = cl_raw.reshape(cl_raw.shape[0], -1)

                # If saved transposed (K, N), try to fix orientation against AnnData.
                if cl_mat.shape[0] != adata.n_obs and cl_mat.shape[1] == adata.n_obs:
                    cl_mat = cl_mat.T

                mapped = None

                # Direct attach when sizes match.
                if cl_mat.shape[0] == adata.n_obs:
                    mapped = np.asarray(cl_mat, dtype=np.int32)
                else:
                    # Attempt mapping via GSEoutput node_index (first column).

                    if gse_path and os.path.exists(gse_path):
                        idx = np.loadtxt(gse_path, delimiter=",", dtype=np.int64, usecols=0)

                        # If labels are transposed relative to GSEoutput, fix that too.
                        if idx.shape[0] != cl_mat.shape[0] and idx.shape[0] == cl_mat.shape[1]:
                            cl_mat = cl_mat.T

                        if idx.shape[0] == cl_mat.shape[0]:
                            out = np.full((adata.n_obs, cl_mat.shape[1]), -1, dtype=np.int32)

                            mask = (idx >= 0) & (idx < adata.n_obs)
                            out[idx[mask], :] = np.asarray(cl_mat[mask, :], dtype=np.int32)
                            mapped = out
                        else:
                            sysOps.throw_status(
                                "Warning: cluster_labels.npy does not match AnnData, and its first dimension does not match GSEoutput; skipping."
                            )
                    else:
                        sysOps.throw_status("Warning: cluster_labels.npy does not match AnnData and no GSEoutput present; skipping.")
                if mapped is not None:
                    n_cols = int(mapped.shape[1])
                    # Naming policy:
                    # - Legacy vector -> write 'cluster'
                    # - Dual labels -> write method-specific names (no privileged default)
                    if n_cols == 1:
                        colnames = ["cluster"]
                    elif n_cols == 2:
                        colnames = ["cluster_hdbscan", "cluster_infomap"]
                        adata.uns["cluster_labels_methods"] = ["hdbscan", "infomap"]
                    elif n_cols == 3:
                        colnames = [
                            "cluster_hdbscan",
                            "cluster_infomap",
                            "cluster_infomap_final",
                        ]
                        adata.uns["cluster_labels_methods"] = [
                            "hdbscan",
                            "infomap",
                            "infomap_final",
                        ]
                    elif n_cols == 4:
                        colnames = [
                            "cluster_hdbscan",
                            "cluster_infomap",
                            "cluster_infomap_final",
                            "cluster_infomap_final_hybrid",
                        ]
                        adata.uns["cluster_labels_methods"] = [
                            "hdbscan",
                            "infomap",
                            "infomap_final",
                            "infomap_final_hybrid",
                        ]
                    else:
                        colnames = [f"cluster_{i}" for i in range(n_cols)]


                    for j, name in enumerate(colnames):
                        adata.obs[name] = np.asarray(mapped[:, j], dtype=np.int32)

                    adata.uns["cluster_labels_source"] = os.path.basename(cl_path)
                    # Store as a plain list, not a tuple; some anndata/h5py stacks refuse
                    # to serialize tuples in .uns and fail late during write_h5ad().
                    adata.uns["cluster_labels_shape"] = [int(x) for x in mapped.shape]
                    adata.uns["cluster_labels_columns"] = colnames

            except Exception as e:
                sysOps.throw_status("ERROR: could not attach cluster labels to AnnData: " + str(e))
                raise

        # -------------------------
        # Write h5ad
        # -------------------------
        try:
            import anndata as _anndata
            if hasattr(_anndata, "settings") and hasattr(_anndata.settings, "allow_write_nullable_strings"):
                _anndata.settings.allow_write_nullable_strings = True
        except Exception:
            pass

        tmp_h5ad_out = h5ad_out + ".tmp"
        try:
            try:
                if os.path.exists(tmp_h5ad_out):
                    os.remove(tmp_h5ad_out)
            except Exception:
                pass

            try:
                adata.write_h5ad(tmp_h5ad_out, compression="gzip", compression_opts=4)
            except TypeError:
                adata.write_h5ad(tmp_h5ad_out)

            os.replace(tmp_h5ad_out, h5ad_out)
        except Exception:
            try:
                if os.path.exists(tmp_h5ad_out):
                    os.remove(tmp_h5ad_out)
            except Exception:
                pass
            raise
        sysOps.throw_status("Saved AnnData to " + h5ad_out)

    except Exception as e:
        sysOps.throw_status("ERROR: could not build/save one-hot AnnData: " + str(e))
        raise

def _subset_index_key_by_keep_nodes(src_index_key_path: str, keep_nodes: np.ndarray) -> np.ndarray:
    """Subset a global index_key.npy onto keep_nodes while preserving (type, raw_index).

    Expected index_key layout: [type, raw_index, local_id]. Legacy shapes are normalized.
    The returned array always has shape (N_keep, 3) with column 2 set to 0..N_keep-1.
    """
    keep_nodes_i64 = np.asarray(keep_nodes, dtype=np.int64)
    ik_full = np.load(src_index_key_path, mmap_mode="r")

    # Normalize legacy formats to (N,3).
    if ik_full.ndim == 1:
        # Legacy: raw_index only.
        ik_full = np.column_stack(
            [
                np.full(int(ik_full.shape[0]), -1, dtype=np.int64),
                np.asarray(ik_full, dtype=np.int64),
                np.arange(int(ik_full.shape[0]), dtype=np.int64),
            ]
        )
    elif ik_full.ndim == 2 and ik_full.shape[1] == 2:
        ik_full = np.column_stack(
            [
                np.asarray(ik_full, dtype=np.int64),
                np.arange(int(ik_full.shape[0]), dtype=np.int64),
            ]
        )
    elif ik_full.ndim == 2 and ik_full.shape[1] >= 3:
        ik_full = np.asarray(ik_full[:, :3], dtype=np.int64)
    else:
        raise ValueError(f"Unexpected index_key.npy shape {getattr(ik_full, 'shape', None)}")

    n_keep = int(keep_nodes_i64.size)
    ik_sub = np.full((n_keep, 3), -1, dtype=np.int64)
    valid = (keep_nodes_i64 >= 0) & (keep_nodes_i64 < int(ik_full.shape[0]))
    if np.any(valid):
        ik_sub[valid, 0:2] = ik_full[keep_nodes_i64[valid], 0:2]
    ik_sub[:, 2] = np.arange(n_keep, dtype=np.int64)
    return ik_sub


def _find_upstream_index_key_path(group_path: str, *, gdp: str | None = None, max_up: int = 6) -> str | None:
    """Find the upstream index_key.npy used to derive a subset export."""
    if gdp is not None:
        cand = os.path.join(_ensure_trailing_slash(str(gdp)), "index_key.npy")
        if os.path.exists(cand):
            return cand

    p = os.path.abspath(str(group_path).rstrip(os.sep))
    for _ in range(int(max_up) + 1):
        parent = os.path.dirname(p)
        if parent == p:
            break
        cand = os.path.join(parent, "index_key.npy")
        if os.path.exists(cand):
            return cand
        p = parent

    return None


def _restore_missing_subset_index_key(group_path: str, *, gdp: str | None = None) -> bool:
    """Recreate <group_path>/index_key.npy exactly from keep_nodes_global.npy.

    This is a bookkeeping-only repair: it uses the same subsetting logic as the
    fine-export stage and does not trigger any graph/coarsen/eigensolve work.
    """
    group_path = _ensure_trailing_slash(str(group_path))
    dst = os.path.join(group_path, "index_key.npy")
    if os.path.exists(dst):
        return False

    keep_path = os.path.join(group_path, "keep_nodes_global.npy")
    if not os.path.exists(keep_path):
        return False

    src = _find_upstream_index_key_path(group_path, gdp=gdp)
    if src is None:
        return False

    keep_nodes = np.load(keep_path, mmap_mode="r")
    keep_nodes = np.asarray(keep_nodes).astype(np.int64, copy=False).ravel()
    ik_sub = _subset_index_key_by_keep_nodes(src, keep_nodes)
    np.save(dst, np.asarray(ik_sub, dtype=np.int64))
    sysOps.throw_status("Re-created missing index_key.npy at " + dst + " from " + src)
    return True


# -----------------------------------------------------------------------------
# Fast CSR submatrix extraction (keep_nodes) for very large sparse graphs
# -----------------------------------------------------------------------------


def _ensure_trailing_slash(path_str):
    if path_str is None:
        return None
    if path_str.endswith('/'):
        return path_str
    return path_str + '/'


def _build_knn_indicator_csr(nn_indices, shape):
    """Build a CSR indicator matrix from an (N x k) integer neighbor index array without a large row-repeat."""
    nn_indices = np.asarray(nn_indices)
    n_rows, k = nn_indices.shape
    indptr = np.arange(0, (n_rows * k) + 1, k, dtype=np.int64)
    indices = nn_indices.reshape(-1).astype(np.int64, copy=False)
    data = np.ones(indices.shape[0], dtype=np.int8)
    return csr_matrix((data, indices, indptr), shape=shape)


def _build_transformed_matrix_from_coords(sym_link_csr, coords, kneighbors, out_npz_path, workers):
    """Rebuild transformed_matrix.npz from coordinates (coords) using kNN + NNLS simplex weights."""
    # kNN search (uses Faiss/Annoy depending on size/dim)
    _, nn_indices = parallel_knn(coords, kneighbors, workers, specified_query=coords, return_distances=False)
    # Drop self neighbor (assumed first)
    nn_indices = nn_indices[:, 1:].astype(np.int32, copy=False)

    nn_indices_csr = _build_knn_indicator_csr(nn_indices, shape=sym_link_csr.shape)

    P = transform_matrix_optimized(sym_link_csr, coords, nn_indices_csr)
    P.eliminate_zeros()
    P = (P + P.T) * 0.5 
    save_npz(out_npz_path, P)


def _load_gse_coords_aligned(gse_path: str, n_nodes: int) -> np.ndarray:
    """Load GSEoutput-like coordinates and align rows by explicit node_index if needed."""
    coords_df = pd.read_csv(gse_path, header=None)
    if coords_df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {gse_path}, found {coords_df.shape[1]}")

    node_ids = coords_df.iloc[:, 0].to_numpy(dtype=np.int64, copy=False)
    coords = coords_df.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
    del coords_df

    if coords.shape[0] == int(n_nodes):
        expected = np.arange(int(n_nodes), dtype=np.int64)
        if np.array_equal(node_ids, expected):
            return coords

    aligned = np.full((int(n_nodes), int(coords.shape[1])), np.nan, dtype=np.float32)
    valid = (node_ids >= 0) & (node_ids < int(n_nodes))
    aligned[node_ids[valid]] = coords[valid]
    if np.any(~np.isfinite(aligned)):
        bad_rows = int(np.sum(~np.isfinite(aligned).all(axis=1)))
        raise ValueError(f"Could not align {gse_path} to {n_nodes} nodes; missing rows = {bad_rows}")
    return aligned


def _edge_distances_from_coords(coords: np.ndarray, row: np.ndarray, col: np.ndarray,
                                chunk_size: int = _FILTER_DISTANCE_CHUNK) -> np.ndarray:
    """Compute Euclidean edge lengths in chunks to limit peak memory."""
    n_edges = int(row.size)
    dists = np.empty(n_edges, dtype=np.float32)
    chunk_size = max(1, int(chunk_size))

    for start in range(0, n_edges, chunk_size):
        stop = min(start + chunk_size, n_edges)
        delta = coords[row[start:stop]] - coords[col[start:stop]]
        sq = np.einsum('ij,ij->i', delta, delta, dtype=np.float64, optimize=True)
        np.sqrt(sq, out=sq)
        dists[start:stop] = sq.astype(np.float32, copy=False)

    return dists


def _smoothed_source_nn_scale(n_nodes: int,
                              row: np.ndarray,
                              col: np.ndarray,
                              dists: np.ndarray) -> np.ndarray:
    """Graph-smoothed local distance scale built from nodewise nearest-neighbor lengths.

    The raw per-node scale is the minimum incident edge length on the implicit
    undirected graph. We then take a one-step closed-neighborhood mean to reduce
    variance, and finally floor the result against the graph-wide typical scale
    so very tight local clusters do not create pathological denominators.
    """
    dist64 = np.asarray(dists, dtype=np.float64)
    raw_nn = np.full(int(n_nodes), np.inf, dtype=np.float64)
    np.minimum.at(raw_nn, row, dist64)
    np.minimum.at(raw_nn, col, dist64)

    valid = np.isfinite(raw_nn)
    if np.any(valid):
        global_nn = float(np.median(raw_nn[valid]))
    else:
        global_nn = 1.0
    raw_nn[~valid] = global_nn

    deg = np.bincount(row, minlength=int(n_nodes)).astype(np.float64, copy=False)
    deg += np.bincount(col, minlength=int(n_nodes))

    nbr_sum = np.bincount(row, weights=raw_nn[col], minlength=int(n_nodes)).astype(np.float64, copy=False)
    nbr_sum += np.bincount(col, weights=raw_nn[row], minlength=int(n_nodes))

    smooth = (raw_nn + nbr_sum) / np.maximum(1.0, 1.0 + deg)
    denom_floor = max(1e-6, 0.1 * global_nn)
    return np.maximum(smooth, denom_floor)


def _incident_edge_ratio_stats(n_nodes: int,
                               row: np.ndarray,
                               col: np.ndarray,
                               dists: np.ndarray,
                               edge_weights: np.ndarray,
                               source_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted incident mean/std of directed distance-to-local-scale ratios."""
    wt = np.asarray(edge_weights, dtype=np.float64)
    dist64 = np.asarray(dists, dtype=np.float64)
    scale = np.asarray(source_scale, dtype=np.float64)

    ratio_row = dist64 / np.maximum(scale[row], 1e-12)
    ratio_col = dist64 / np.maximum(scale[col], 1e-12)

    sum_w = np.bincount(row, weights=wt, minlength=int(n_nodes)).astype(np.float64, copy=False)
    sum_w += np.bincount(col, weights=wt, minlength=int(n_nodes))

    wr_row = wt * ratio_row
    wr_col = wt * ratio_col
    sum_r = np.bincount(row, weights=wr_row, minlength=int(n_nodes)).astype(np.float64, copy=False)
    sum_r += np.bincount(col, weights=wr_col, minlength=int(n_nodes))

    sum_r2 = np.bincount(row, weights=wr_row * ratio_row, minlength=int(n_nodes)).astype(np.float64, copy=False)
    sum_r2 += np.bincount(col, weights=wr_col * ratio_col, minlength=int(n_nodes))

    denom = np.maximum(sum_w, 1e-12)
    mu = sum_r / denom
    var = np.maximum(0.0, (sum_r2 / denom) - (mu * mu))
    sigma = np.sqrt(var)

    valid = sum_w > 0
    if np.any(valid):
        global_mu = float(np.median(mu[valid]))
        global_sigma = float(np.median(sigma[valid]))
    else:
        global_mu = 1.0
        global_sigma = 0.0

    sigma_floor = max(1e-6, 0.1 * global_sigma, 1e-3 * global_mu)
    sigma = np.maximum(sigma, sigma_floor)
    return ratio_row, ratio_col, mu, sigma


def _export_filtered_subgraph_from_gse(base_path: str,
                                       output_name: str,
                                       *,
                                       filtered_dirname: str = "filtered",
                                       z_thresh: float = _FILTER_EDGE_ZSCORE) -> str:
    """Export a connectivity-trimmed induced subgraph into <base_path>/filtered/."""
    base_path = _ensure_trailing_slash(str(base_path))
    gse_path = os.path.join(base_path, str(output_name))
    link_path = os.path.join(base_path, "link_assoc_reindexed.npz")
    index_key_path = os.path.join(base_path, "index_key.npy")

    if not os.path.exists(gse_path):
        raise FileNotFoundError(f"Could not find embedding output for filtering: {gse_path}")
    if not os.path.exists(link_path):
        raise FileNotFoundError(f"Could not find graph for filtering: {link_path}")
    if not os.path.exists(index_key_path):
        raise FileNotFoundError(f"Could not find index key for filtering: {index_key_path}")

    link_triu = load_npz(link_path).tocsr()
    link_triu.sum_duplicates()
    link_triu.eliminate_zeros()
    n_nodes = int(link_triu.shape[0])
    if n_nodes <= 0:
        raise ValueError("Cannot filter an empty graph.")

    coords = _load_gse_coords_aligned(gse_path, n_nodes)
    link_coo = link_triu.tocoo(copy=False)
    row = np.asarray(link_coo.row, dtype=np.int64)
    col = np.asarray(link_coo.col, dtype=np.int64)
    data = np.asarray(link_coo.data, dtype=np.float64)

    dists = _edge_distances_from_coords(coords, row, col)
    source_scale = _smoothed_source_nn_scale(n_nodes, row, col, dists)
    ratio_row, ratio_col, mu_ratio, sigma_ratio = _incident_edge_ratio_stats(
        n_nodes, row, col, dists, data, source_scale
    )
    retain_mask = (
        (ratio_row <= (mu_ratio[row] + float(z_thresh) * sigma_ratio[row])) &
        (ratio_col <= (mu_ratio[col] + float(z_thresh) * sigma_ratio[col]))
    )
    if not np.any(retain_mask):
        sysOps.throw_status("Warning: filter removed all edges; falling back to the original graph.")
        retain_mask = np.ones(row.shape[0], dtype=np.bool_)

    retain_graph = csr_matrix(
        (np.ones(int(np.sum(retain_mask)), dtype=np.int8), (row[retain_mask], col[retain_mask])),
        shape=(n_nodes, n_nodes),
    )
    retain_graph = retain_graph + retain_graph.T

    n_components, labels = connected_components(retain_graph, directed=False, return_labels=True)
    comp_sizes = np.bincount(labels, minlength=max(1, int(n_components)))
    keep_label = int(np.argmax(comp_sizes))
    keep_node_mask = (labels == keep_label)
    keep_nodes = np.flatnonzero(keep_node_mask).astype(np.int64, copy=False)

    edge_keep = retain_mask & keep_node_mask[row] & keep_node_mask[col]
    if keep_nodes.size == 0 or (not np.any(edge_keep)):
        sysOps.throw_status("Warning: filtered graph became empty; falling back to the original graph.")
        keep_nodes = np.arange(n_nodes, dtype=np.int64)
        edge_keep = np.ones(row.shape[0], dtype=np.bool_)

    new_ids = np.full(n_nodes, -1, dtype=np.int64)
    new_ids[keep_nodes] = np.arange(keep_nodes.size, dtype=np.int64)

    filtered_graph = csr_matrix(
        (data[edge_keep], (new_ids[row[edge_keep]], new_ids[col[edge_keep]])),
        shape=(int(keep_nodes.size), int(keep_nodes.size)),
    )
    filtered_graph.sum_duplicates()
    filtered_graph.eliminate_zeros()

    filtered_path = _ensure_trailing_slash(os.path.join(base_path, filtered_dirname))
    os.makedirs(filtered_path, exist_ok=True)
    # Always clear stale derived artifacts so the child rerun cannot reuse
    # eigensystems/transformed matrices built for a previous filtered export.
    sysOps.prune_dir_except([], path=filtered_path)

    np.save(os.path.join(filtered_path, "keep_nodes_global.npy"), keep_nodes)
    np.save(
        os.path.join(filtered_path, "index_key.npy"),
        np.asarray(_subset_index_key_by_keep_nodes(index_key_path, keep_nodes), dtype=np.int64),
    )
    save_npz(
        os.path.join(filtered_path, "link_assoc_reindexed.npz"),
        scipy.sparse.triu(filtered_graph, format='csr'),
    )

    stats = {
        "z_threshold": float(z_thresh),
        "ratio_scale_median": float(np.median(source_scale)) if source_scale.size else 0.0,
        "ratio_mu_median": float(np.median(mu_ratio)) if mu_ratio.size else 0.0,
        "n_nodes_input": int(n_nodes),
        "n_edges_input": int(row.size),
        "n_edges_retained": int(np.sum(edge_keep)),
        "n_nodes_retained": int(keep_nodes.size),
        "node_fraction_retained": float(keep_nodes.size / max(1, n_nodes)),
        "edge_fraction_retained": float(np.sum(edge_keep) / max(1, row.size)),
        "n_components_after_truncation": int(n_components),
        "largest_component_size": int(comp_sizes[keep_label]),
    }
    with open(os.path.join(filtered_path, "filter_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)

    sysOps.throw_status(
        "Filtered export complete: kept " +
        str(stats["n_nodes_retained"]) + "/" + str(stats["n_nodes_input"]) +
        " nodes and " + str(stats["n_edges_retained"]) + "/" + str(stats["n_edges_input"]) + " edges."
    )
    return filtered_path


@njit(fastmath=True)
def fista_simplex_precomputed_fast(ATA_reg, ATb, step_init,
                                   max_iter, tol, backtracking_steps,
                                   x0):
    """
    FISTA on: min_{x in Δ} 0.5*x^T ATA_reg x - x^T ATb   (const dropped)
    Uses only ATA_reg and ATb (no A@x), with micro-backtracking.
    Δ = { x >= 0, sum x = 1 }.
    """
    k = ATA_reg.shape[0]
    if k == 0:
        return np.zeros(0, dtype=ATA_reg.dtype), 0.0

    # initialization
    if x0 is not None and x0.shape[0] == k:
        x = x0.copy()
        # ensure feasible (light projection)
        buf_u    = np.empty(k, dtype=ATA_reg.dtype)
        buf_desc = np.empty(k, dtype=ATA_reg.dtype)
        buf_csum = np.empty(k, dtype=ATA_reg.dtype)
        project_to_simplex(x, x, buf_u, buf_desc, buf_csum)
    else:
        x = np.full(k, 1.0 / k, dtype=ATA_reg.dtype)
    y = x.copy()
    x_prev = x.copy()
    t = 1.0
    step = step_init

    # workspaces
    tmp    = np.empty(k, dtype=ATA_reg.dtype)
    grad_y = np.empty(k, dtype=ATA_reg.dtype)
    buf_u    = np.empty(k, dtype=ATA_reg.dtype)
    buf_desc = np.empty(k, dtype=ATA_reg.dtype)
    buf_csum = np.empty(k, dtype=ATA_reg.dtype)

    # quadratic objective: f(z) = 0.5 z^T ATA_reg z - z^T ATb
    def f_quad(z):
        s = 0.0
        for i in range(k):
            acc = 0.0
            row = ATA_reg[i]
            for j in range(k):
                acc += row[j] * z[j]
            s += z[i] * acc
        dot = 0.0
        for i in range(k):
            dot += z[i] * ATb[i]
        return 0.5 * s - dot

    for _it in range(max_iter):
        # grad_y = ATA_reg @ y - ATb
        for i in range(k):
            acc = 0.0
            row = ATA_reg[i]
            for j in range(k):
                acc += row[j] * y[j]
            grad_y[i] = acc - ATb[i]

        # gradient step + projection
        for i in range(k):
            tmp[i] = y[i] - step * grad_y[i]
        project_to_simplex(tmp, x, buf_u, buf_desc, buf_csum)

        # micro backtracking (quadratic form only; no A@x)
        fy = f_quad(y)
        fx = f_quad(x)
        bt = 0
        while fx > fy and bt < backtracking_steps:
            step *= 0.5
            for i in range(k):
                tmp[i] = y[i] - step * grad_y[i]
            project_to_simplex(tmp, x, buf_u, buf_desc, buf_csum)
            fx = f_quad(x)
            bt += 1

        # convergence on iterate change
        max_diff = 0.0
        for i in range(k):
            dlt = abs(x[i] - x_prev[i])
            if dlt > max_diff:
                max_diff = dlt
        if max_diff < tol:
            break

        # Nesterov update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        beta = (t - 1.0) / t_new
        for i in range(k):
            y[i] = x[i] + beta * (x[i] - x_prev[i])
            x_prev[i] = x[i]
        t = t_new

    return x, 0.0


@njit(fastmath=True)
def power_iteration(ATA):
    """Fast power iteration with early stopping + safety factor."""
    d = ATA.shape[0]
    if d == 0:
        return 1e-10

    v = np.ones(d, dtype=ATA.dtype)
    v_norm = np.sqrt(np.sum(v * v))
    if v_norm < 1e-14:
        return 1e-10
    v /= v_norm

    prev_lambda = 0.0
    for _ in range(10):
        Av = ATA @ v
        lambda_est = np.sum(v * Av)
        if lambda_est < 0.0:  # PSD guard
            lambda_est = 0.0

        # early stopping with safety factor
        if lambda_est > 1e-12:
            if abs(lambda_est - prev_lambda) < 1e-4 * lambda_est:
                return max(1.1 * lambda_est, 1e-10)
        elif abs(prev_lambda) < 1e-12:
            return 1e-10

        v_norm_Av = np.sqrt(np.sum(Av * Av))
        if v_norm_Av < 1e-14:
            return 1e-10

        v = Av / v_norm_Av
        prev_lambda = lambda_est

    final_lambda_est = np.sum(v * (ATA @ v))
    if final_lambda_est < 0.0:
        final_lambda_est = 0.0
    return max(1.1 * final_lambda_est, 1e-10)


@njit(parallel=True, fastmath=True)
def transform_rowwise_simplex(N_data, N_indices, N_indptr,
                              Z, nn_flat_neighbor_indices, nn_indptr,
                              lambda_reg,
                              solver_tol, max_iter, backtracking_steps):
    """
    Per-edge version using the simplex-constrained solver (memory-lean):
    computes values directly into a pre-sized 1D array aligned with nn_* CSR.
    """
    n_nodes = len(N_indptr) - 1
    total_output_edges = len(nn_flat_neighbor_indices)

    vals_out = np.zeros(total_output_edges, dtype=np.float64)

    embedding_dim = Z.shape[1]

    for i in prange(n_nodes):
        N_row_start_ptr = N_indptr[i]
        N_row_end_ptr   = N_indptr[i+1]

        nn_row_start_ptr = nn_indptr[i]
        nn_row_end_ptr   = nn_indptr[i+1]

        # avoid slicing to reduce overhead in tight loops
        k_i = nn_row_end_ptr - nn_row_start_ptr
        deg_i = N_row_end_ptr - N_row_start_ptr

        if k_i == 0 or deg_i == 0:
            continue

        # Simplex over a single candidate is exactly alpha = [1].
        if k_i == 1:
            s = 0.0
            for ptr in range(N_row_start_ptr, N_row_end_ptr):
                s += N_data[ptr]
            vals_out[nn_row_start_ptr] = s
            continue

        # Local dictionary M_i = [Z_k - Z_i]
        M_i = np.empty((embedding_dim, k_i), dtype=Z.dtype)
        Zi = Z[i]
        for r_local_idx in range(k_i):
            nk = nn_flat_neighbor_indices[nn_row_start_ptr + r_local_idx]
            Zk = Z[nk]
            for d in range(embedding_dim):
                M_i[d, r_local_idx] = Zk[d] - Zi[d]

        # Precompute AT, ATA and add ridge ONCE per row
        AT_i = M_i.T                           # shape (k_i, d)
        ATA_i = AT_i @ M_i                     # shape (k_i, k_i)
        for j in range(k_i):
            ATA_i[j, j] += lambda_reg          # regularized Hessian

        # Single step size per row
        L_i = power_iteration(ATA_i)
        if L_i < 1e-12:
            L_i = 1e-12
        step_i = 1.0 / L_i

        alpha_sum_for_row_i = np.zeros(k_i, dtype=np.float64)
        b = np.empty(embedding_dim, dtype=Z.dtype)  # reuse per row
        ATb = np.empty(k_i, dtype=Z.dtype)
        # warm start across edges in this row
        x0 = None 

        # Loop over original neighbors j of i
        for ptr in range(N_row_start_ptr, N_row_end_ptr):
            j = N_indices[ptr]
            w_ij = N_data[ptr]

            # b = Z_j - Z_i
            Zj = Z[j]
            for d in range(embedding_dim):
                b[d] = Zj[d] - Zi[d]

            # ATb = AT_i @ b
            for r in range(k_i):
                s = 0.0
                for d in range(embedding_dim):
                    s += AT_i[r, d] * b[d]
                ATb[r] = s

            # Solve using fast precomputed solver (warm start)
            alpha, _ = fista_simplex_precomputed_fast(
                ATA_i, ATb, step_i,
                max_iter, solver_tol, backtracking_steps,
                x0
            )
            x0 = alpha  # warm-start next edge

            # Accumulate (no normalization needed)
            for r_local_idx in range(k_i):
                alpha_sum_for_row_i[r_local_idx] += w_ij * alpha[r_local_idx]

        # Write values for row i
        vals_out[nn_row_start_ptr:nn_row_end_ptr] = alpha_sum_for_row_i[:k_i]

    return vals_out


@njit(fastmath=True)
def project_to_simplex(z, out, buf_u, buf_desc, buf_csum):
    """
    Euclidean projection of z onto the probability simplex:
        Δ = { x >= 0, sum(x) = 1 }.
    Duchi et al. (2008) algorithm (sort-based). O(k log k).
    Writes result into `out` using caller-provided work buffers (no new allocation).
    """
    k = z.shape[0]
    if k == 0:
        return

    # Keep sort-based Duchi et al. (2008) for robustness and simplicity.
    # (This function signature is used across solvers; buffers reused to avoid allocs.)
    for i in range(k):
        buf_u[i] = z[i]
    buf_u.sort()  # ascending
    for i in range(k):
        buf_desc[i] = buf_u[k - 1 - i]
    s = 0.0
    for i in range(k):
        s += buf_desc[i]
        buf_csum[i] = s
    rho = 0
    for i in range(k):
        t = (buf_csum[i] - 1.0) / (i + 1.0)
        if buf_desc[i] - t > 0.0:
            rho = i
    tau = (buf_csum[rho] - 1.0) / (rho + 1.0)

    # projection and light renorm (for tiny roundoff)
    sum_out = 0.0
    for i in range(k):
        v = z[i] - tau
        out[i] = v if v > 0.0 else 0.0
        sum_out += out[i]

    if sum_out > 0.0:
        inv = 1.0 / sum_out
        for i in range(k):
            out[i] *= inv
    else:
        # pathological: distribute uniformly
        uni = 1.0 / k
        for i in range(k):
            out[i] = uni


def transform_matrix_optimized(N, Z, nn_indices_csr, lambda_reg=1e-2,
                               solver_tol=1e-6, max_iter=100, backtracking_steps=2,
                               use_fp32=False, fast_csr=False):
    """
    Optimized transform with per-row precomputation and fast precomputed solver.
    Parameters:
        solver_tol (float): stopping tolerance for FISTA (iterate diff).
        max_iter (int):     maximum FISTA iterations.
        backtracking_steps (int): micro backtracking halvings (0–2 recommended).
        use_gershgorin (bool): if True, use Gershgorin bound for step; else power iteration.
        use_fp32 (bool):    if True, use float32 for compute (bandwidth-speed tradeoff).
        fast_csr (bool):    if True, skip CSR duplicate/zero/sort housekeeping.
    """
    target_dtype = np.float32 if use_fp32 else np.float64

    Z_arr = np.asarray(Z, dtype=target_dtype)
    if isinstance(Z_arr, np.memmap):
        Z_contig = np.array(Z_arr, dtype=target_dtype, order='C', copy=True)
    elif Z_arr.flags.c_contiguous:
        Z_contig = Z_arr
    else:
        Z_contig = np.ascontiguousarray(Z_arr)

    ind = nn_indices_csr.indices
    ptr = nn_indices_csr.indptr

    if ind.dtype == np.int64 and ind.flags.c_contiguous:
        nn_flat = ind
    else:
        nn_flat = np.ascontiguousarray(ind, dtype=np.int64)

    if ptr.dtype == np.int64 and ptr.flags.c_contiguous:
        nn_ptrs = ptr
    else:
        nn_ptrs = np.ascontiguousarray(ptr, dtype=np.int64)

    vals = transform_rowwise_simplex(
        N.data, N.indices, N.indptr,
        Z_contig, nn_flat, nn_ptrs,
        lambda_reg,
        solver_tol, max_iter, backtracking_steps,
    )

    n_nodes = N.shape[0]
    N_new = csr_matrix((vals, nn_flat, nn_ptrs), shape=(n_nodes, n_nodes))
    if not fast_csr:
        N_new.sum_duplicates()
        N_new.eliminate_zeros()
        N_new.sort_indices()

    return N_new


def spec_GSEobj(sub_GSEobj, output_Xpts_filename = None, init_eig_count = None, X_init=None, tot_main_outer_iters=1):
    # perform structured "spectral GSEobj" (sGSEobj) likelihood maximization

    def _block_flat_indices(row0, row1, spat_dims):
        if row0 is None or row1 is None or row1 <= row0:
            return np.empty((0,), dtype=np.int64)
        return np.arange(row0 * spat_dims, row1 * spat_dims, dtype=np.int64)

    def _activate_new_block(obj, X_in, row0, row1, grad_tol=1e-8, curv_tol=1e-8, max_backtracks=8):
        """
        Small dense bootstrap step on just-added rows.
        Keeps the current subsample fixed: the first calc_grad() call here
        materializes pairings if obj.reset_subsample == True and all later
        evaluations reuse the same sampled objective.
        """
        x0 = np.asarray(X_in, dtype=np.float64).reshape(-1).copy()
        f0, g0 = obj.calc_grad(x0)  # also freezes the current subsample
        I = _block_flat_indices(row0, row1, obj.spat_dims)

        if I.size == 0 or (not np.isfinite(f0)):
            return X_in, float(f0)

        gB = np.asarray(g0[I], dtype=np.float64)
        if not np.all(np.isfinite(gB)):
            obj.calc_grad(x0)  # restore internal buffers to accepted iterate
            return X_in, float(f0)

        m = I.size
        HB = np.zeros((m, m), dtype=np.float64)
        basis_vec = np.zeros_like(x0)

        try:
            for j, idx in enumerate(I):
                basis_vec[idx] = 1.0
                Hj = obj.calc_hessp(x0, basis_vec)
                HB[:, j] = Hj[I]
                basis_vec[idx] = 0.0
        except Exception:
            obj.calc_grad(x0)  # restore internal buffers to accepted iterate
            return X_in, float(f0)

        HB = 0.5 * (HB + HB.T)

        try:
            evals, evecs = LA.eigh(HB)
        except LA.LinAlgError:
            obj.calc_grad(x0)  # restore internal buffers to accepted iterate
            return X_in, float(f0)

        gnorm = LA.norm(gB)
        ref_scale = 1.0
        if row0 is not None and row0 > 0:
            ref_scale = float(np.median(np.abs(X_in[:row0, :])))
            if (not np.isfinite(ref_scale)) or ref_scale <= 0.0:
                ref_scale = 1.0

        # Gradient-driven block Newton / damped Newton step.
        if gnorm > grad_tol:
            shift = max(1e-8, -float(evals[0]) + 1e-8)
            try:
                step = -LA.solve(HB + shift * np.eye(m, dtype=np.float64), gB)
            except LA.LinAlgError:
                step = -gB

            # Ensure the candidate is a descent direction.
            if np.dot(gB, step) >= 0.0:
                step = -gB

            step_norm = LA.norm(step)
            max_step_norm = max(1.0, ref_scale) * np.sqrt(float(m))
            if step_norm > max_step_norm and step_norm > 0.0:
                step *= (max_step_norm / step_norm)

            g_dot_step = float(np.dot(gB, step))
            alpha = 1.0
            for _ in range(max_backtracks):
                x_try = x0.copy()
                x_try[I] += alpha * step
                f_try, _ = obj.calc_grad(x_try)
                if np.isfinite(f_try) and (f_try <= f0 + 1e-4 * alpha * g_dot_step):
                    return x_try.reshape(X_in.shape), float(f_try)
                alpha *= 0.5

            obj.calc_grad(x0)  # restore internal buffers to accepted iterate
            return X_in, float(f0)

        # Near-stationary but negatively curved: take a tiny negative-curvature step.
        if float(evals[0]) < -curv_tol:
            base_dir = np.asarray(evecs[:, 0], dtype=np.float64)
            base_step = max(1e-3, 1e-2 * ref_scale)
            for signed_dir in (base_dir, -base_dir):
                alpha = base_step
                for _ in range(max_backtracks):
                    x_try = x0.copy()
                    x_try[I] += alpha * signed_dir
                    f_try, _ = obj.calc_grad(x_try)
                    if np.isfinite(f_try) and (f_try < f0 - 1e-12):
                        return x_try.reshape(X_in.shape), float(f_try)
                    alpha *= 0.5

        obj.calc_grad(x0)  # restore internal buffers to accepted iterate
        return X_in, float(f0)

    def _run_trust_krylov_with_accept(obj, X_in, f_in, maxiter):
        """
        Run the existing trust-krylov solve, but only accept the candidate if it
        is finite and improves the same frozen sampled objective.
        """
        x0 = np.asarray(X_in, dtype=np.float64).reshape(-1).copy()
        res = minimize(
            fun=obj.calc_grad,
            hessp=obj.calc_hessp,
            x0=x0,
            args=(),
            method='trust-krylov',
            jac=True,
            options={'maxiter': maxiter},
        )

        x_candidate = getattr(res, 'x', None)
        if x_candidate is None:
            obj.calc_grad(x0)  # restore internal buffers to accepted iterate
            return X_in, float(f_in), res

        x_candidate = np.asarray(x_candidate, dtype=np.float64).reshape(-1)
        if x_candidate.size != x0.size or (not np.all(np.isfinite(x_candidate))):
            obj.calc_grad(x0)  # restore internal buffers to accepted iterate
            return X_in, float(f_in), res

        f_candidate, _ = obj.calc_grad(x_candidate)

        if np.isfinite(f_candidate) and (f_candidate <= f_in + 1e-10):
            return x_candidate.reshape(X_in.shape), float(f_candidate), res

        obj.calc_grad(x0)  # restore internal buffers to accepted iterate
        return X_in, float(f_in), res

    sub_GSEobj.alphas_arr = None
    for main_outer_iter in range(tot_main_outer_iters):
        sub_GSEobj.reweighted_Nlink = None
        if main_outer_iter > 0:
            sysOps.sh("rm " + sysOps.globaldatapath + "iter*")
            sub_GSEobj.reset_gaussian_parameteres(
                np.loadtxt(sysOps.globaldatapath + "GSEoutput.txt", delimiter=',', dtype=np.float64)[:, 1:],
                main_outer_iter + 1,
            )
            os.rename(
                sysOps.globaldatapath + "GSEoutput.txt",
                sysOps.globaldatapath + str(main_outer_iter) + "_GSEoutput.txt",
            )

        if main_outer_iter == tot_main_outer_iters - 1:
            subGSEobj_eignum = sub_GSEobj.seq_evecs.shape[0]
        else:
            subGSEobj_eignum = int(sub_GSEobj.seq_evecs.shape[0] / 10)

        # ------------------------------------------------------------------
        # Resume support:
        # If the run was interrupted after writing iter*_GSEoutput.txt files,
        # restart from the latest snapshot instead of re-initializing.
        # (Assumes tot_main_outer_iters=1; still safe for other values.)
        # ------------------------------------------------------------------
        if output_Xpts_filename is not None:
            # If the final iter snapshot exists but the consolidated output
            # file is missing (or stale), promote it and return immediately.
            final_iter_fn = sub_GSEobj.path + 'iter' + str(subGSEobj_eignum) + '_' + output_Xpts_filename
            out_fn = sub_GSEobj.path + output_Xpts_filename
            if (main_outer_iter == tot_main_outer_iters - 1) and os.path.exists(final_iter_fn):
                try:
                    out_mtime = os.path.getmtime(out_fn)
                except Exception:
                    out_mtime = -1
                try:
                    iter_mtime = os.path.getmtime(final_iter_fn)
                except Exception:
                    iter_mtime = -1

                if (not os.path.exists(out_fn)) or (iter_mtime > out_mtime):
                    sysOps.throw_status('Resuming: promoting ' + final_iter_fn + ' -> ' + out_fn)
                    # keep iter snapshot; only (re)create the consolidated file
                    sysOps.sh('cp -p ' + final_iter_fn + ' ' + out_fn)

                # Return coordinates from the final snapshot.
                return np.loadtxt(final_iter_fn, delimiter=',', dtype=np.float64)[:, 1:]

            # If we weren't given an explicit initialization, try to resume from
            # the most advanced iter snapshot available for *this* outer loop.
            if X_init is None and init_eig_count is None:
                best_eig = None
                best_file = None
                suffix = '_' + output_Xpts_filename
                try:
                    for fn in os.listdir(sub_GSEobj.path):
                        if fn.startswith('iter') and fn.endswith(suffix):
                            mid = fn[4:-len(suffix)]
                            if mid.isdigit():
                                k = int(mid)
                                # Only consider snapshots that are valid for
                                # the current sub-problem size.
                                if k <= subGSEobj_eignum and (best_eig is None or k > best_eig):
                                    best_eig = k
                                    best_file = fn
                except Exception:
                    best_eig, best_file = None, None

                if best_file is not None:
                    sysOps.throw_status('Resuming: loading ' + best_file + ' (eig_count=' + str(best_eig) + ')')
                    X_init = np.loadtxt(sub_GSEobj.path + best_file, delimiter=',', dtype=np.float64)[:, 1:]
                    init_eig_count = int(best_eig)

        manifold_increment = sub_GSEobj.spat_dims
        sysOps.throw_status("Incrementing eigenspace: " + str(manifold_increment))

        if X_init is None:
            X = None
            pending_block_start = 0
        else:
            # X_init can be either:
            #   (Npts x spat_dims) : coordinate initialization (legacy)
            #   (init_eig_count x spat_dims) : coefficient initialization (new warm-start path)
            if init_eig_count is None:
                init_eig_count = sub_GSEobj.seq_evecs.shape[0]
            init_eig_count = int(init_eig_count)

            X_init_arr = np.asarray(X_init, dtype=np.float64)
            if X_init_arr.ndim != 2:
                raise ValueError("X_init must be a 2D array")

            # Prefer interpreting X_init as coordinates when it matches Npts (legacy behaviour).
            if X_init_arr.shape[0] == sub_GSEobj.Npts and X_init_arr.shape[1] == sub_GSEobj.spat_dims:
                X = sub_GSEobj.seq_evecs[:init_eig_count, :].dot(X_init_arr)
            elif X_init_arr.shape[0] == init_eig_count and X_init_arr.shape[1] == sub_GSEobj.spat_dims:
                X = X_init_arr.copy()
            else:
                raise ValueError(
                    "X_init shape must be either (Npts, spat_dims) or (init_eig_count, spat_dims); "
                    f"got {X_init_arr.shape}, expected ({sub_GSEobj.Npts},{sub_GSEobj.spat_dims}) or ({init_eig_count},{sub_GSEobj.spat_dims})."
                )
            pending_block_start = None  # warm start/resume is already populated

        if init_eig_count is None:
            init_eig_count = sub_GSEobj.spat_dims

        eig_count = int(init_eig_count)
        iter = 0

        while True:
            # SOLVE SUB-GSEobj
            maxiter = 10
            if eig_count == init_eig_count and (X is None):
                rows, cols = sub_GSEobj.link_data.nonzero()
                rmsq = np.sqrt(
                    np.square(
                        sub_GSEobj.seq_evecs[:sub_GSEobj.spat_dims, rows] -
                        sub_GSEobj.seq_evecs[:sub_GSEobj.spat_dims, cols]
                    ).dot(sub_GSEobj.link_data.data) / sub_GSEobj.Nlink
                )
                del rows, cols
                maxiter = 100
                X = np.zeros([init_eig_count, sub_GSEobj.spat_dims], dtype=np.float64)
                X[:sub_GSEobj.spat_dims, :sub_GSEobj.spat_dims] = np.diag(np.divide(1.0, rmsq))
                # initialize as identity matrix scaled by first-pass edge RMS
                pending_block_start = 0
            elif eig_count != init_eig_count:
                old_rows = X.shape[0]
                X = np.concatenate([X, np.zeros([1, sub_GSEobj.spat_dims], dtype=np.float64)], axis=0)
                # Keep old rows exactly fixed; newly added rows remain zero until block activation.
                if pending_block_start is None:
                    pending_block_start = old_rows

            sub_GSEobj.inference_eignum = eig_count  # set number of degrees of freedom

            # pre-calculate back-projection matrix: calculate inner-product of eigenvector matrix
            # with itself, and invert to compensate for lack of orthogonalization between eigenvectors
            sub_GSEobj.reset_subsample = True
            if eig_count >= manifold_increment and (eig_count % manifold_increment == 0 or eig_count == subGSEobj_eignum):

                if iter % 10 == 0:
                    maxiter = 100
                else:
                    maxiter = 10

                sysOps.throw_status(
                    'Optimizing eigencomponent ' + str(eig_count) + '/' + str(subGSEobj_eignum) +
                    ' in ' + str(sub_GSEobj.spat_dims) + 'D.'
                )
                sub_GSEobj.print_status = False

                if eig_count == subGSEobj_eignum:
                    sysOps.throw_status('Final gradient-descent.')
                    sub_GSEobj.reweighted_Nlink = None
                    maxiter = 100
                    tot_outer_iter = 3
                else:
                    tot_outer_iter = 1

                for outer_iter in range(tot_outer_iter):
                    block_start = pending_block_start
                    X_seed, f_seed = _activate_new_block(sub_GSEobj, X, block_start, eig_count)
                    X, f_seed, res = _run_trust_krylov_with_accept(sub_GSEobj, X_seed, f_seed, maxiter)
                    pending_block_start = None

                iter += 1

            my_Xpts = sub_GSEobj.seq_evecs[:sub_GSEobj.inference_eignum, :].T.dot(X)
            if output_Xpts_filename is not None and (
                eig_count == subGSEobj_eignum or
                (subGSEobj_eignum >= 10 and eig_count % (int(subGSEobj_eignum / 10)) == 0)
            ):  # can include to get regular updates on the solution at regular intervals
                np.savetxt(
                    sub_GSEobj.path + 'iter' + str(eig_count) + '_' + output_Xpts_filename,
                    np.concatenate([np.arange(sub_GSEobj.Npts).reshape([sub_GSEobj.Npts, 1]), my_Xpts], axis=1),
                    fmt='%i,' + ','.join(['%.10e' for i in range(my_Xpts.shape[1])]),
                    delimiter=',',
                )

            np.save(
                sub_GSEobj.path + "sample_Xpts.npy",
                my_Xpts[np.random.choice(my_Xpts.shape[0], min(500000, my_Xpts.shape[0]), replace=False), :],
            )

            if eig_count == subGSEobj_eignum:
                break

            eig_count += 1

        if not (output_Xpts_filename is None):
            sysOps.sh(
                "cp -p " + sub_GSEobj.path + 'iter' + str(subGSEobj_eignum) + '_' +
                output_Xpts_filename + " " + sub_GSEobj.path + output_Xpts_filename
            )

        del sub_GSEobj.gl_diag, sub_GSEobj.gl_innerprod

    sub_GSEobj.inference_eignum = int(subGSEobj_eignum)  # return to original value
    return my_Xpts           


def fill_params(params):
    _reject_removed_coarsen_params(params)

    # if unloaded from list, place params back in list
    for el in params:
        if type(params[el]) != list and type(params[el]) != bool:
            params[el] = list([params[el]])

    if '-inference_eignum' in params:
        params['-inference_eignum'] = int(params['-inference_eignum'][0])
    else:
        params['-inference_eignum'] = 30

    if '-inference_dim' in params:
        params['-inference_dim'] = int(params['-inference_dim'][0])
    else:
        params['-inference_dim'] = 2

    if '-scales' in params:
        params['-scales'] = int(params['-scales'][0])
    else:
        params['-scales'] = 1

    if '-final_eignum' in params:
        params['-final_eignum'] = int(params['-final_eignum'][0])
    else:
        params['-final_eignum'] = 100

    if '-shape_match' in params:
        params['-shape_match'] = str(params['-shape_match'][0])
    else:
        params['-shape_match'] = None

    if '-filter' in params:
        params['-filter'] = True
    else:
        params['-filter'] = None

    if '-bifurcate_type' in params:
        params['-bifurcate_type'] = int(params['-bifurcate_type'][0])
    else:
        params['-bifurcate_type'] = -1

    if '-iterations' in params:
        params['-iterations'] = int(params['-iterations'][0])
    else:
        params['-iterations'] = 1

    if '-nonlinear_proj' in params:
        params['-nonlinear_proj'] = int(params['-nonlinear_proj'][0])
    else:
        params['-nonlinear_proj'] = 0

    if '-exit_code' in params:
        params['-exit_code'] = str(params['-exit_code'][0])
    else:
        params['-exit_code'] = 'full'

    if '-calc_final' in params:
        v = str(params['-calc_final'][0]).strip() if len(params['-calc_final']) else ""
        params['-calc_final'] = v if v else None
    else:
        params['-calc_final'] = None

    if '-intermed_indexing_directory' in params:
        params['-intermed_indexing_directory'] = str(params['-intermed_indexing_directory'][0])
    else:
        params['-intermed_indexing_directory'] = None

    if '-path' in params:
        params['-path'] = str(params['-path'][0])
        if not params['-path'].endswith('/'):
            params['-path'] += "//"

    with open(params['-path'] + "params.txt", 'w') as paramfile:
        for el in params:
            sysOps.throw_status(el + " " + str(params[el]))
            if type(params[el]) == bool and params[el]:
                paramfile.write(el + "\n")
            elif type(params[el]) != bool:
                paramfile.write(el + ' ' + str(params[el]) + "\n")

def generalized_eigen_embedding(
    subspace,
    csr,
    *,
    r=None,
    center=True,
    l2_normalize=False,
    stationary="uniform",  # "degree" or "uniform"
    tau_rel=1e-8,
    eps_rel=1e-12,
):
    """
    Generalized eigen-embedding with adaptive diffusion-time (t) selection via
    cutoff spectral gap at rank r.

    Chooses t from a small doubling grid {1,2,4,8,16} to maximize:
        gap(t) = mu_r(t) - mu_{r+1}(t),
    where mu_i(t) are eigenvalues of M(t)=H^{-1/2}G(t)H^{-1/2} in descending order.

    Robust to subspace orientation:
      - accepts X as (N,m) or (m,N) and auto-transposes if needed.
    """

    # --- adjacency / size (authoritative N) ---
    if not scipy.sparse.isspmatrix_csr(csr):
        A = csr.tocsr()
    else:
        A = csr
    A = A.astype(np.float64, copy=False)
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError(f"csr must be square, got {A.shape}.")

    # --- subspace (accept (N,m) or (m,N)) ---
    X = np.asarray(subspace, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("subspace must be 2D.")
    if X.shape[0] != n:
        if X.shape[1] == n:
            X = X.T  # (m,N) -> (N,m)
        else:
            raise ValueError(
                f"subspace shape {X.shape} incompatible with csr shape {A.shape}."
            )
    n, m = X.shape
    if m == 0:
        return X.copy()

    # r := min(m-1, r)
    if r is None:
        r = m
    r = int(min(m - 1, int(r)))
    if r <= 0:
        return np.zeros((n, 0), dtype=np.float64)

    # --- preprocess X ---
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    if l2_normalize:
        norms = np.linalg.norm(X, axis=0)
        good = norms > 0
        X[:, good] /= norms[good]

    # --- degrees / isolate mask ---
    d = np.asarray(A.sum(axis=1)).ravel()
    u = (d > 0).astype(np.float64)

    # --- build P ---
    if stationary == "uniform":
        # Metropolis / max-degree: P_ij = A_ij / max(d_i,d_j), i!=j; then fill diag.
        coo = A.tocoo(copy=False)
        mask = coo.row != coo.col
        rows = coo.row[mask]
        cols = coo.col[mask]
        denom = np.maximum(d[rows], d[cols])

        data = coo.data[mask].astype(np.float64, copy=False)
        data = np.divide(data, denom, out=np.zeros_like(data), where=denom > 0)

        P_off = scipy.sparse.csr_matrix((data, (rows, cols)), shape=A.shape)

        row_sum = np.asarray(P_off.sum(axis=1)).ravel()
        diag = np.zeros_like(d)
        nz = d > 0
        diag[nz] = np.clip(1.0 - row_sum[nz], 0.0, 1.0)

        P = (P_off + scipy.sparse.diags(diag, 0, format="csr")).tocsr()

    elif stationary == "degree":
        inv_d = np.zeros_like(d)
        nz = d > 0
        inv_d[nz] = 1.0 / d[nz]
        P = A.multiply(inv_d[:, None]).tocsr()
    else:
        raise ValueError('stationary must be "uniform" or "degree".')

    ones = np.ones(n, dtype=np.float64)
    colsum = np.asarray(P.T @ ones).ravel()

    # --- build H (independent of t) ---
    PX = P @ X                      # (N,m)
    Q = X.T @ PX                    # (m,m) = X^T P X

    diag_term = (X.T @ (u[:, None] * X)) + (X.T @ (colsum[:, None] * X))
    H = diag_term - (Q + Q.T)
    H = 0.5 * (H + H.T)

    # stabilize and form H^{-1/2}
    mean_diag = float(np.trace(H) / m)
    if not np.isfinite(mean_diag) or mean_diag <= 0:
        mean_diag = 1.0
    tau = float(tau_rel) * mean_diag
    H_reg = H + tau * np.eye(m, dtype=np.float64)

    evals_H, U_H = np.linalg.eigh(H_reg)
    max_eval = float(np.max(evals_H))
    if not np.isfinite(max_eval) or max_eval <= 0:
        max_eval = 1.0
    eps = float(eps_rel) * max(1.0, max_eval)
    evals_H = np.clip(evals_H, eps, None)

    H_inv_sqrt = (U_H * (1.0 / np.sqrt(evals_H))) @ U_H.T

    # --- score helper: spectral gap at cutoff r ---
    def gap_and_M_from_G(G):
        M = H_inv_sqrt @ G @ H_inv_sqrt
        M = 0.5 * (M + M.T)
        ev = np.linalg.eigvalsh(M)     # ascending
        mu = ev[::-1]                 # descending
        return float(mu[r - 1] - mu[r]), M

    # t=1 uses Q directly: G1 = Sym(X^T P X)
    best_gap, best_M = gap_and_M_from_G(0.5 * (Q + Q.T))

    # --- tune t on a small doubling grid (cost ~ max(t) sparse multiplies) ---
    PtX = PX
    t_cur = 1
    best_t = 1
    for t_target in (2, 4, 8, 16):
        while t_cur < t_target:
            PtX = P @ PtX
            t_cur += 1

        XtPtX = X.T @ PtX
        gap, M = gap_and_M_from_G(0.5 * (XtPtX + XtPtX.T))
        if gap > best_gap:
            best_gap, best_M = gap, M
            best_t = int(t_target)

    # --- final embedding from best_M ---
    evals_M, V = np.linalg.eigh(best_M)
    idx = np.argsort(evals_M)[::-1]
    V_r = V[:, idx[:r]]

    Y = X @ (H_inv_sqrt @ V_r)

    sysOps.throw_status("Optimal t <- " + str(best_t))
    return Y


def parallel_nbrs(nbrs, query_subset, start_idx, return_distances=True):
    """
    Perform KNN search using sklearn's NearestNeighbors on a subset of queries.
    """
    if return_distances:
        distances, indices = nbrs.kneighbors(query_subset, return_distance=True)
        return distances, indices, start_idx

    indices = nbrs.kneighbors(query_subset, return_distance=False)
    return None, indices.astype(int, copy=False), start_idx


def parallel_annoy(index, query_subset, k, batch_size=1000, num_threads=None, search_k=-1,
                   return_distances=True, max_pending=None):
    """
    Perform approximate KNN search using Annoy in parallel, with bounded in-flight
    batches and direct writes into the final output arrays.

    This preserves version 1 neighbor semantics while reducing Python-object
    churn, lock contention, and peak scheduler pressure.
    """
    m = int(len(query_subset))
    if num_threads is None or int(num_threads) < 1:
        num_threads = 1
    else:
        num_threads = int(num_threads)

    nn_indices = np.zeros((m, k), dtype=int)
    nn_distances = np.zeros((m, k), dtype=float) if return_distances else None

    search_kwargs = {'include_distances': bool(return_distances)}
    if search_k > 0:
        search_kwargs['search_k'] = int(search_k)

    def search_batch(start_idx, end_idx):
        for i in range(start_idx, end_idx):
            if return_distances:
                indices, distances = index.get_nns_by_vector(query_subset[i], k, **search_kwargs)
                nret = min(len(indices), k)
                if nret > 0:
                    nn_indices[i, :nret] = np.asarray(indices[:nret], dtype=nn_indices.dtype)
                    nn_distances[i, :nret] = np.asarray(distances[:nret], dtype=nn_distances.dtype)
            else:
                indices = index.get_nns_by_vector(query_subset[i], k, **search_kwargs)
                nret = min(len(indices), k)
                if nret > 0:
                    nn_indices[i, :nret] = np.asarray(indices[:nret], dtype=nn_indices.dtype)

    if max_pending is None:
        max_pending = max(1, 2 * num_threads)

    next_start = 0
    pending = set()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        while next_start < m or pending:
            while next_start < m and len(pending) < max_pending:
                end = min(next_start + batch_size, m)
                pending.add(executor.submit(search_batch, next_start, end))
                next_start = end

            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                fut.result()

    return nn_distances, nn_indices

class dot2:
    def __init__(self, csr_op1, csr_op2, rownorm=True):
        # Convert operators to CSR if needed
        self.csr_op1 = csr_op1
        self.csr_op2 = csr_op2
        self.csr_op2T = csr_op2.T
        self.shape = getattr(csr_op1, "shape", None)
        self.dtype = getattr(csr_op1, "dtype", np.float64)

    def dot(self, x):
        return self.csr_op2.dot(self.csr_op1.dot(self.csr_op2T.dot(x)))

    def matvec(self, x):
        return self.dot(x)

    def matmat(self, X):
        return self.dot(X)
        

def parallel_knn(
    space,
    kneighbors,
    num_workers=-1,
    specified_query=None,
    approximate=False,
    search_k=-1,
    n_trees=None,
    batch_size=1000,
    reindex=None,
    return_distances=True,
):
    """
    Perform parallel K-Nearest Neighbors search using different libraries based on
    data characteristics.

    This preserves version 1 neighbor semantics while adding:
      * optional distance suppression when callers only need indices
      * bounded Annoy futures / direct writes
      * chunked reindexing and self-neighbor repair to avoid large temporaries
    """
    if num_workers < 0:
        num_workers = max(1, int(NTHREADS))
    sysOps.throw_status(f"Running parallel_knn on space.shape = {space.shape}, num_workers = {num_workers}")
    sysOps.throw_status("reindex = " + str(reindex))

    self_query = (specified_query is None) or (specified_query is space) or (reindex is not None)
    effective_k = kneighbors + 1 if self_query else kneighbors

    if effective_k > space.shape[0]:
        effective_k = int(space.shape[0])
    if effective_k < 1:
        raise ValueError("effective_k < 1; empty dataset?")

    queries = space if specified_query is None else specified_query
    total_queries = int(queries.shape[0])
    use_annoy = approximate or (space.shape[1] > 3 and (queries.shape[0] > 1e5 or space.shape[0] > 1e5))

    query_chunks = None
    if not use_annoy:
        chunk_size = int(np.ceil(total_queries / num_workers))
        query_chunks = [(queries[i:i + chunk_size], i) for i in range(0, total_queries, chunk_size)]

    nn_distances = None
    nn_indices = None

    try:
        if use_annoy:
            if not approximate:
                sysOps.throw_status("Using Annoy for approximate nearest neighbors due to space size.")
            sysOps.throw_status("Building Annoy index.")

            dim = int(space.shape[1])
            index = AnnoyIndex(dim, 'euclidean')
            sysOps.throw_status("Adding items to Annoy index.")
            for i in range(space.shape[0]):
                index.add_item(i, space[i])

            if n_trees is None:
                n_trees = min(50, max(10, int(np.log2(space.shape[0]) * dim)))
            sysOps.throw_status(f"Building Annoy index with n_trees = {n_trees}.")
            index.build(n_trees=n_trees)
            sysOps.throw_status("Annoy index built successfully.")

            sysOps.throw_status("Conducting parallel Annoy search.")
            nn_distances, nn_indices = parallel_annoy(
                index,
                queries,
                effective_k,
                batch_size=batch_size,
                num_threads=num_workers,
                search_k=search_k,
                return_distances=return_distances,
            )
            sysOps.throw_status("Annoy search complete.")

        elif space.shape[1] <= 3:
            sysOps.throw_status("Using sklearn's NearestNeighbors for low-dimensional data.")
            nbrs = NearestNeighbors(n_neighbors=effective_k, n_jobs=1).fit(space)
            results = Parallel(n_jobs=num_workers, prefer="threads", require="sharedmem")(
                delayed(parallel_nbrs)(nbrs, chunk, start_idx, return_distances=return_distances)
                for chunk, start_idx in query_chunks
            )

            nn_indices = np.zeros((total_queries, effective_k), dtype=int)
            if return_distances:
                nn_distances = np.zeros((total_queries, effective_k), dtype=float)

            for distances, indices, start_idx in results:
                end_idx = start_idx + indices.shape[0]
                nn_indices[start_idx:end_idx, :] = indices
                if return_distances:
                    nn_distances[start_idx:end_idx, :] = distances

        else:
            sysOps.throw_status("Using Faiss for high-dimensional data.")
            dim = int(space.shape[1])
            space_f = np.ascontiguousarray(space, dtype=np.float32)
            index = faiss.IndexFlatL2(dim)
            index.add(space_f)
            sysOps.throw_status("Conducting Faiss search (Faiss handles threading internally).")
            try:
                import sys as _sys
                import os as _os
                if _sys.platform == "darwin":
                    default_threads = 1
                else:
                    default_threads = int(num_workers)
                faiss_threads = int(_os.getenv("FAISS_NUM_THREADS", str(default_threads)))
                if faiss_threads < 1:
                    faiss_threads = 1
                faiss.omp_set_num_threads(faiss_threads)
            except Exception:
                pass

            nn_indices = np.zeros((total_queries, effective_k), dtype=int)
            if return_distances:
                nn_distances = np.zeros((total_queries, effective_k), dtype=float)

            chunk_size = int(np.ceil(total_queries / num_workers))
            for start_idx in range(0, total_queries, chunk_size):
                end_idx = min(start_idx + chunk_size, total_queries)
                chunk_f = np.ascontiguousarray(queries[start_idx:end_idx], dtype=np.float32)
                if not np.isfinite(chunk_f).all():
                    chunk_f = np.nan_to_num(chunk_f, copy=False).astype(np.float32, copy=False)
                distances, indices = index.search(chunk_f, effective_k)
                nn_indices[start_idx:end_idx, :] = indices
                if return_distances:
                    np.sqrt(distances, out=distances)
                    nn_distances[start_idx:end_idx, :] = distances

        if reindex is not None:
            for start in range(0, nn_indices.shape[0], 250000):
                end = min(nn_indices.shape[0], start + 250000)
                nn_indices[start:end, :] = reindex[nn_indices[start:end, :]]

        if self_query:
            total_misplaced = 0
            total_found = 0
            missing = []
            for start in range(0, nn_indices.shape[0], 250000):
                end = min(nn_indices.shape[0], start + 250000)
                rows = np.arange(start, end, dtype=nn_indices.dtype)
                mismatched = np.flatnonzero(nn_indices[start:end, 0] != rows)
                total_misplaced += int(mismatched.size)

                for local_r in mismatched:
                    r = int(start + local_r)
                    pos = -1
                    for c in range(1, effective_k):
                        if nn_indices[r, c] == r:
                            pos = c
                            break

                    if pos > 0:
                        tmp_i = int(nn_indices[r, 0])
                        nn_indices[r, 0] = r
                        nn_indices[r, pos] = tmp_i
                        if return_distances:
                            tmp_d = float(nn_distances[r, 0])
                            nn_distances[r, 0] = nn_distances[r, pos]
                            nn_distances[r, pos] = tmp_d
                        total_found += 1
                    elif pos < 0:
                        missing.append(r)

            sysOps.throw_status("Found " + str(total_misplaced) + " misplaced indices ...")
            if total_misplaced > 0:
                sysOps.throw_status("len(found_indices) = " + str(total_found))
                sysOps.throw_status("swapped elements")
            if len(missing) > 0:
                sysOps.throw_status(
                    f"Warning: Query points {np.asarray(missing, dtype=np.int64)} not found in their own neighbor lists."
                )

        sysOps.throw_status("returning nn_distances, nn_indices")
        sysOps.throw_status("np.sum(nn_indices < 0) = " + str(int(np.sum(nn_indices < 0))))
        return nn_distances, nn_indices

    except Exception as e:
        sysOps.throw_status(f"An error occurred during parallel_knn execution: {e}")
        raise


def full_GSE(output_name, params):
    # Primary function call for image inference and segmentation.
    # Inputs:
    #     imagemodule_input_filename: link data input file
    #     other arguments: boolean settings for which subroutine to run

    # The objective setup below matches the previous non-coarsened full solve.

    if type(params['-inference_eignum']) == list:
        fill_params(params)
    _reject_removed_coarsen_params(params)
    inference_eignum = int(params['-inference_eignum'])
    inference_dim = int(params['-inference_dim'])
    GSE_final_eigenbasis_size = int(params['-final_eignum'])
    sysOps.num_workers = NTHREADS
    sysOps.globaldatapath = str(params['-path'])
    # Default: auto. If '-h5ad_include_sequences' is present, always include.
    # Otherwise, include sequences only when label_pt appears STAR-less and contains sequences.
    sysOps.h5ad_include_nonunique_genes = ('-h5ad_include_nonunique_genes' in params)
    sysOps.h5ad_include_sequences = True if ('-h5ad_include_sequences' in params) else None

    try:
        os.mkdir(sysOps.globaldatapath + "tmp")
    except:
        pass

    this_GSEobj = GSEobj(inference_dim, inference_eignum)

    kneighbors = max(2 * inference_eignum, 10 * inference_dim)
    if not sysOps.check_file_exists("transformed_matrix.npz"):
        scaled_evecs = np.load(sysOps.globaldatapath + "orig_evecs_gapnorm.npy")
        _, nn_indices = parallel_knn(scaled_evecs, kneighbors, sysOps.num_workers, specified_query=scaled_evecs, return_distances=False)
        nn_indices = nn_indices[:, 1:].astype(np.int32, copy=False)
        sysOps.throw_status(sysOps.globaldatapath + 'transformed_matrix.npz not found. Generating from scratch.')

        # Build CSR indicator without materializing a large "rows" array (avoids O(N*k) extra ints).
        n_rows, k = nn_indices.shape
        indptr = np.arange(0, (n_rows * k) + 1, k, dtype=np.int64)
        indices = nn_indices.reshape(-1).astype(np.int64, copy=False)
        data = np.ones(indices.shape[0], dtype=np.int8)
        nn_indices_csr = csr_matrix((data, indices, indptr), shape=this_GSEobj.link_data.shape)
        output0 = transform_matrix_optimized(this_GSEobj.link_data + this_GSEobj.link_data.T, scaled_evecs, nn_indices_csr)
        output0.eliminate_zeros()
        output0 += output0.T
        output0 *= 0.5
        save_npz(sysOps.globaldatapath + 'transformed_matrix.npz', output0)
        del nn_indices_csr

    if not sysOps.check_file_exists('evecs.npy'):
        this_GSEobj.inference_eignum = int(GSE_final_eigenbasis_size)
        sysOps.throw_status("Generating final eigenbasis ...")
        this_GSEobj.eigen_decomp(orth=True, pmax=1)
    else:
        this_GSEobj.seq_evecs = np.load(sysOps.globaldatapath + "evecs.npy").T

    # IMPORTANT: always sync inference_eignum to the loaded eigenbasis.
    # Otherwise, subsequent iter*_GSEoutput.txt lookups can use the wrong
    # eigen-count when restarting.
    this_GSEobj.inference_eignum = int(this_GSEobj.seq_evecs.shape[0])

    sysOps.throw_status("this_GSEobj.seq_evecs.shape = " + str(this_GSEobj.seq_evecs.shape))
    if 'exit_code' in params and params['-exit_code'].lower() != 'gd' and params['-exit_code'].lower() != 'full':
        return
    # If the consolidated output is missing, resume from the most recent
    # iter*_GSEoutput.txt snapshot (or promote the final iter snapshot).
    if output_name is not None:
        final_iter_fn = 'iter' + str(this_GSEobj.inference_eignum) + '_' + output_name

        if not sysOps.check_file_exists(output_name):
            if sysOps.check_file_exists(final_iter_fn):
                sysOps.throw_status('Found ' + final_iter_fn + ' but missing ' + output_name + ' -- promoting iter snapshot.')
                sysOps.sh('cp -p ' + sysOps.globaldatapath + final_iter_fn + ' ' + sysOps.globaldatapath + output_name)
            else:
                sysOps.throw_status('Running spec_GSEobj with params = ' + str(params))
                spec_GSEobj(
                    this_GSEobj,
                    output_Xpts_filename=output_name,
                    tot_main_outer_iters=params['-scales'],
                )

        del this_GSEobj.seq_evecs
        this_GSEobj.seq_evecs = None

    del this_GSEobj
    sysOps.throw_status("Initial output complete.")


def reindex_input_files(path):
    
    if not sysOps.check_file_exists("link_assoc_reindexed.npz", path):

        if sysOps.check_file_exists('link_assoc.npy',path):
            data = np.load(path + 'link_assoc.npy')
        elif sysOps.check_file_exists('link_assoc.txt',path):
            data = np.loadtxt(path + 'link_assoc.txt', delimiter=',', dtype=np.float64)[:,1:]
            if sysOps.check_file_exists('link_assoc_extra.txt',path):
                prefactor = .00001
                sysOps.throw_status("Loading " + path + 'link_assoc_extra.txt')
                extra_data = np.loadtxt(path + 'link_assoc_extra.txt', delimiter=',', dtype=np.float64)[:,1:]
                extra_data[:,2] *= prefactor
                data = np.concatenate([data,extra_data],axis=0)
        else:
            raise ValueError("Unsupported file format")

        # Extract type 1 and type 2 indices
        type1_indices = data[:, 0].astype(int)
        type2_indices = data[:, 1].astype(int)

        # Find unique indices for type 1 and type 2
        unique_type1, reindexed_type1 = np.unique(type1_indices, return_inverse=True)
        unique_type2, reindexed_type2 = np.unique(type2_indices, return_inverse=True)

        # Reindex type 2 to start after type 1 indices
        reindexed_type2 += len(unique_type1)
        Npts = len(unique_type1)+len(unique_type2)

        csr = csr_matrix((data[:, 2], (reindexed_type1.astype(int), reindexed_type2.astype(int))), (Npts, Npts))
        
        save_npz(path + "link_assoc_reindexed.npz", scipy.sparse.triu(csr.tocsr()))
        del csr


        # Create index_key array
        type1_key = np.column_stack((np.zeros_like(unique_type1, dtype=np.float64), unique_type1, np.arange(len(unique_type1))))
        type2_key = np.column_stack((np.ones_like(unique_type2, dtype=np.float64), unique_type2, np.arange(len(unique_type1),len(unique_type1)+len(unique_type2))))
        index_key = np.vstack((type1_key, type2_key)).astype(np.int32)

        # Save the index_key array
        np.save(path + "index_key.npy", index_key)
    
    else:
        index_key = np.load(path + "index_key.npy")
        return np.sum(index_key[:,0] == 0),  np.sum(index_key[:,0] == 1)

    if sysOps.check_file_exists("orig_batch_array.npy"):
        np.save(sysOps.globaldatapath + "batch_array.npy", np.load(sysOps.globaldatapath + "orig_batch_array.npy")[index_key[:,1]])

    return len(unique_type1), len(unique_type2)
  
    
def select_points(this_GSEobj, nn_num):
    num_candidates = (2 ** this_GSEobj.spat_dims) * nn_num
    selected_indices = np.zeros((this_GSEobj.Npts, 2 * nn_num), dtype=int)

    # Calculate batch size based on max_mem_buffer
    max_mem_buffer = 10000000  # Max number of elements in temporary array
    Npts = this_GSEobj.Npts
    spat_dims = this_GSEobj.spat_dims
    Nbatch = max(1, max_mem_buffer // (num_candidates * spat_dims))
    sysOps.throw_status(f"Processing in batches of size {Nbatch}")

    # Initialize arrays to store results
    nearest_indices = np.zeros((this_GSEobj.Npts, nn_num), dtype=int)
    uniform_sampled_indices = np.zeros((this_GSEobj.Npts, nn_num), dtype=int)

    # Process in batches
    for start_idx in range(0, Npts, Nbatch):
        end_idx = min(start_idx + Nbatch, Npts)
        batch_size = end_idx - start_idx

        # Step 1: Randomly select candidate points for the batch
        batch_candidates = np.random.choice(
            Npts, (batch_size, num_candidates), replace=True
        )

        # Extract batch points
        batch_Xpts = this_GSEobj.Xpts[start_idx:end_idx]

        # Step 2: Compute distances for the batch
        batch_distances = np.linalg.norm(
            batch_Xpts[:, None, :] - this_GSEobj.Xpts[batch_candidates], axis=2
        )

        # Step 3: Sort distances for each point in the batch
        batch_sorted_indices = np.argsort(batch_distances, axis=1)

        # Step 4: Select nearest nn_num points
        batch_nearest_indices = np.take_along_axis(
            batch_candidates, batch_sorted_indices[:, :nn_num], axis=1
        )

        # Step 5: Select nn_num points uniformly from the remaining points
        batch_remaining_indices = np.take_along_axis(
            batch_candidates, batch_sorted_indices[:, nn_num:], axis=1
        )
        interval = max(1, batch_remaining_indices.shape[1] // nn_num)
        batch_uniform_sampled_indices = batch_remaining_indices[:, ::interval][:, :nn_num]

        # Store the results in the overall arrays
        nearest_indices[start_idx:end_idx] = batch_nearest_indices
        uniform_sampled_indices[start_idx:end_idx] = batch_uniform_sampled_indices

    # Combine both sets of selected indices
    selected_indices[:, :nn_num] = nearest_indices
    selected_indices[:, nn_num:] = uniform_sampled_indices
    return selected_indices

def get_triv_status(vecs, threshold = 1E-5):
    vars = np.var(vecs,axis=0)
    return vars/np.median(vars) < threshold

def partition_graph_csc_matrix(csc_mat, num_partitions):
    """
    Partition a graph represented as a CSC (Compressed Sparse Column) matrix
    using METIS's k-way partitioning.

    Args:
    - csc_mat (scipy.sparse.csc_matrix): The adjacency matrix of the graph.
    - num_partitions (int): The desired number of partitions.

    Returns:
    - Tuple[List[int], List[int]]: A tuple containing the partition vector and the edge cut.
    """
    # Ensure the matrix is symmetric (undirected graph) and has no self-loops

    # Convert the CSC matrix to the adjacency list format expected by pymetis
    adjacency_list = [csc_mat.indices[csc_mat.indptr[i]:csc_mat.indptr[i + 1]].tolist() for i in range(csc_mat.shape[0])]
    sysOps.throw_status('len(adjacency_list) = ' + str(len(adjacency_list)))
    sysOps.throw_status('Running pymetis graph cut with num_partitions = ' + str(num_partitions))
    # Partition the graph
    cut, partition = pymetis.part_graph(num_partitions, adjacency=adjacency_list)
    sysOps.throw_status('Edges cuts = ' + str(cut))
    return partition

def generate_fast_preorthbasis(csr_op1, k, metis_iterations = 1, spat_dims=3):

    csc = csr_op1.tocsc()
    Npts = csr_op1.shape[0]
    all_basis = list()
    for iter in range(metis_iterations):
        rowsums = np.array(csc.sum(axis=1)).flatten()
        if np.sum(rowsums == 0) > 0:
            iso0 = (rowsums == 0)
            try:
                sysOps.throw_status(
                    "WARNING: generate_fast_preorthbasis found " + str(int(np.sum(iso0))) +
                    " zero-degree rows; adding unit self-loops for stability."
                )
            except Exception:
                pass
            csc = csc + scipy.sparse.diags(iso0.astype(np.float64))
            rowsums = np.array(csc.sum(axis=1)).flatten()
        csc = scipy.sparse.diags(1.0/np.maximum(1E-10,rowsums)).dot(csc)
        csc = csc + csc.T
        num_partitions = max(k+1,int(csr_op1.shape[0] * 0.01)*(2**iter))
        
        partition_vector = partition_graph_csc_matrix(csc, num_partitions)
        del csc
        partition_vector = np.int64(partition_vector)
        used_partitions = np.zeros(num_partitions,dtype=np.bool_)
        used_partitions[partition_vector] = True
        new_partition_map = -np.ones(num_partitions,dtype=np.int64)
        num_partitions = np.sum(used_partitions) # re-set
        new_partition_map[np.where(used_partitions)[0]] = np.arange(num_partitions)
        partition_vector = new_partition_map[partition_vector]

        sysOps.throw_status('Preparing segment link-associations using re-set num_partitions = ' + str(num_partitions) + '...')
        if num_partitions < k+1:
            np.save(sysOps.globaldatapath + "preorthbasis.npy", np.random.randn(Npts,spat_dims))
            return
        link_data = csr_op1.tocoo()
        # Map the original row and column indices to their respective groups
        grouped_row = partition_vector[link_data.row]
        grouped_col = partition_vector[link_data.col]
        
        # Create a COO matrix with the new dimensions and with data aggregated according to the groups
        new_coo = scipy.sparse.coo_matrix((link_data.data, (grouped_row, grouped_col)), shape=(num_partitions, num_partitions))
        # Sum duplicates (i.e., aggregate the data) and convert to CSC format
        csc = new_coo.tocsc()
        del new_coo
        csc.sum_duplicates()
        sysOps.throw_status('Normalizing ...')
        vals = csc.dot(np.ones(num_partitions,dtype=np.float64)) # row-sums
        if np.sum(vals <= 0.0) > 0:
            sysOps.throw_status("np.sum(vals <= 0.0) = " + str(np.sum(vals <= 0.0) ))
            sysOps.throw_status(str(np.where(vals <= 0.0)[0] ))
            sysOps.exitProgram()
        csc = scipy.sparse.diags(np.power(vals,-1)).dot(csc)
        del vals
        sysOps.throw_status('Done. Calculating eigenvectors ...')

        eignum = min(csc.shape[1]-5,k) # ensure not too many eigenvalues are requested
        ncv = 2 * (eignum + 1)  # Initial guess for NCV, at least twice the NEV
        max_attempts = 5  # Maximum number of attempts to find eigenvalues

        for attempt in range(max_attempts):
            try:
                evals_large, evecs_large = scipy.sparse.linalg.eigs(csc, k=eignum + 1, which='LR', ncv=ncv)
                break
            except ArpackNoConvergence as err:
                err_k = len(err.eigenvalues)
                if err_k <= 0:
                    raise AssertionError("No eigenvalues found.")
                sysOps.throw_status('Assigning ' + str(err_k) + ' eigenvectors due to non-convergence ...')
                evecs_large = np.ones([csc.shape[1], eignum + 1], dtype=np.float64) / np.sqrt(csc.shape[1])
                evecs_large[:, :err_k] = np.real(err.eigenvectors)
                evals_large = np.ones(eignum + 1, dtype=np.float64) * np.min(err.eigenvalues)
                evals_large[:err_k] = np.real(err.eigenvalues)
                break
            except ArpackError as e:
                if 'No shifts could be applied' in str(e):
                    ncv += 20  # Increment NCV and retry
                    sysOps.throw_status('Increasing NCV to ' + str(ncv) + ' due to ARPACK error and retrying...')
                    if ncv > csc.shape[0]:
                        raise ValueError("NCV exceeds matrix dimensions, unable to compute eigenvalues with current parameters.")
                else:
                    raise  # Re-raise the exception if it's not the specific "No shifts" error

        del csc
        triv_eig_index = np.argmin(np.var(evecs_large,axis = 0))
        evecs_large = np.real(evecs_large[:,np.where(np.arange(evecs_large.shape[1]) != triv_eig_index)[0]])
        evecs_large = evecs_large[partition_vector,:]
        sysOps.throw_status('Done. Orthogonalizing and saving.')
        # center and norm
        for i in range(evecs_large.shape[1]):
            evecs_large[:,i] -= np.mean(evecs_large[:,i])
            evecs_large[:,i] /= 1E-10 + LA.norm(evecs_large[:,i])
        all_basis.append(scipy.linalg.qr(evecs_large,mode='economic')[0])
        del evecs_large
        
    np.save(sysOps.globaldatapath + "preorthbasis.npy", np.concatenate(all_basis,axis=1))
    

def parallel_krylov(krylov_num,csr_op1,diag,csr_op2,init_vector,rev_operator=False):
    subspace = np.zeros([init_vector.shape[0],krylov_num],dtype=np.float64)
    subspace[:,0] = init_vector-np.mean(init_vector)
    for i in range(1,krylov_num):
        if csr_op2 is None:
            if rev_operator:
                subspace[:,i] = diag.dot(csr_op1.dot(subspace[:,i-1]))
            else:
                subspace[:,i] = diag.dot(csr_op1.dot(subspace[:,i-1]))

        else:
            if rev_operator:
                subspace[:,i] = diag.dot(csr_op2.dot(subspace[:,i-1]))
            else:
                subspace[:,i] = diag.dot(csr_op2.dot(csr_op1.dot(csr_op2.T.dot(subspace[:,i-1]))))
    return subspace


def parallel_krylov_fill(out_block, csr_op1, diag, csr_op2, init_vector, rev_operator=False):
    # out_block shape: (N, krylov_num)
    out_block[:, 0] = init_vector
    out_block[:, 0] -= np.mean(out_block[:, 0])

    for i in range(1, out_block.shape[1]):
        prev = out_block[:, i - 1]
        if csr_op2 is None:
            out_block[:, i] = diag.dot(csr_op1.dot(prev))
        else:
            if rev_operator:
                out_block[:, i] = diag.dot(csr_op2.dot(prev))
            else:
                out_block[:, i] = diag.dot(
                    csr_op2.dot(csr_op1.dot(csr_op2.T.dot(prev)))
                )


def get_eigs(csr_op1, k, csr_op2=None, krylov_iterations=5, rev_operator=False):
    krylov_approx = sysOps.globaldatapath + "preorthbasis.npy"
    if csr_op2 is None:
        diag = scipy.sparse.diags(np.power(np.array(csr_op1.sum(axis=1)).flatten() + 1E-10, -1))
    else:
        if rev_operator:
            diag = scipy.sparse.diags(np.power(np.array(csr_op2.dot(np.ones(csr_op1.shape[0]))).flatten() + 1E-10, -1))
        else:
            deg = csr_op2.dot(
                csr_op1.dot(
                    csr_op2.T.dot(np.ones(csr_op1.shape[0], dtype=np.float64))
                )
            )
            diag = scipy.sparse.diags(np.power(np.array(deg).flatten() + 1E-10, -1))

    nseed = max(3, int(np.ceil(k / 10)))

    # Load only the needed seed vectors, not the entire preorthbasis file.
    seq_evecs = np.array(
        np.load(krylov_approx, mmap_mode="r")[:, :nseed],
        dtype=np.float64,
        order="F",
        copy=True,
    )

    krylov_iter = 0
    while krylov_iter < krylov_iterations:
        if seq_evecs.shape[1] == 0:
            break

        for i in range(seq_evecs.shape[1]):
            seq_evecs[:, i] -= np.mean(seq_evecs[:, i])
            seq_evecs[:, i] /= LA.norm(seq_evecs[:, i])

        krylov_num = int(np.ceil(2 * max(100, max(k, seq_evecs.shape[1])) / seq_evecs.shape[1]))
        sysOps.throw_status('krylov_iter = ' + str(krylov_iter) + " : krylov_num = " + str(krylov_num))

        total_cols = seq_evecs.shape[1] * krylov_num
        krylov_space = np.empty(
            (seq_evecs.shape[0], total_cols),
            dtype=np.float64,
            order="F",
        )

        def _fill_one(i):
            sl = slice(i * krylov_num, (i + 1) * krylov_num)
            parallel_krylov_fill(
                krylov_space[:, sl],
                csr_op1,
                diag,
                csr_op2,
                seq_evecs[:, i],
                rev_operator,
            )

        Parallel(
            n_jobs=min(NTHREADS, seq_evecs.shape[1]),
            prefer="threads",
            require="sharedmem",
        )(delayed(_fill_one)(i) for i in range(seq_evecs.shape[1]))

        krylov_space = scipy.linalg.qr(
            krylov_space,
            mode='economic',
            overwrite_a=True,
            check_finite=False,
        )[0]

        if csr_op2 is None:
            innerprod = krylov_space.T.dot(diag.dot(csr_op1.dot(krylov_space)))
        else:
            if rev_operator:
                innerprod = krylov_space.T.dot(diag.dot(csr_op2.dot(krylov_space)))
            else:
                innerprod = krylov_space.T.dot(diag.dot(csr_op2.dot(csr_op1.dot(csr_op2.T.dot(krylov_space)))))

        evals,evecs = LA.eig(innerprod)

        eval_order = np.argsort(-np.real(evals))[:(2*k)]
        evecs = np.real(evecs[:,eval_order])
        evals = np.real(evals[eval_order])
        seq_evecs = krylov_space.dot(evecs)

        seq_evecs -= np.mean(seq_evecs,axis=0)
        seq_evecs = seq_evecs / LA.norm(seq_evecs, axis=0, keepdims=True)

        if csr_op2 is None:
            order = np.argsort(-np.diagonal(seq_evecs.T.dot(diag.dot(csr_op1.dot(seq_evecs)))))
        else:
            if rev_operator:
                order = np.argsort(-np.diagonal(seq_evecs.T.dot(diag.dot(csr_op2.dot(seq_evecs)))))
            else:
                order = np.argsort(-np.diagonal(seq_evecs.T.dot(diag.dot(csr_op2.dot(csr_op1.dot(csr_op2.T.dot(seq_evecs)))))))

        seq_evecs = seq_evecs[:,order]
        seq_evals = evals[order]
        triv_eig_indices = get_triv_status(seq_evecs)

        seq_evecs = seq_evecs[:,~triv_eig_indices][:,:k]
        seq_evals = evals[~triv_eig_indices][:k]

        krylov_iter += 1
        sysOps.throw_status('Trivial indices ' + str(np.where(triv_eig_indices)[0]) + ' removed.')

    np.save(sysOps.globaldatapath + 'evecs.npy',seq_evecs)
    np.save(sysOps.globaldatapath + 'evals.npy',seq_evals)
    return

class GSEobj:
    # object for all image inference
    
    def __init__(self,inference_dim=None,inference_eignum=None,bipartite_data=True,inp_path=""):
        # if constructor has been called, it's assumed that link_assoc.txt is in present directory with original indices
        # we first want
        self.num_workers = sysOps.num_workers
        self.index_key = None
        self.bipartite_data = bipartite_data
        self.link_data = None
        self.sum_pt_tp1_link = None
        self.sum_pt_tp2_link = None
        self.Npts = None
        self.print_status = True
        self.subsample_pairings = None
        self.subsample_pairing_weights = None
        self.seq_evecs = None
        self.path = str(sysOps.globaldatapath)+inp_path
        
        #### variables for gradient ascent calculation ####
        self.reweighted_Nlink = None
        self.ampfactors = None
        self.task_inputs_and_outputs = None
        self.Xpts = None
        self.dXpts = None
        self.gl_diag = None
        self.gl_innerprod = None
        
        if inference_dim is None:
            self.spat_dims = 2 # default
        else:
            self.spat_dims = int(inference_dim)
            
        if inference_eignum is None:
            self.inference_eignum = None # default
        else:
            self.inference_eignum = int(inference_eignum)
        
        # counts and indices in inp_data, if this is included in input, take precedence over read-in numbers from inp_settings and imagemodule_input_filename
        
        self.load_data() # requires inputted value of Npt_tp1 if inp_data = None
    
        sysOps.throw_status('Done.')
        

    def load_data(self):
        # Load raw link data from link_assoc.txt
        # 1. link type
        # 2. pts1 cluster index
        # 3. pts2 cluster index
        # 4. link count

        # Subset exports (e.g. component0_fine) may legitimately lose only their
        # local index_key.npy during cleanup/resume workflows. Recreate it from
        # keep_nodes_global.npy if possible before falling back to legacy reindexing.
        try:
            _restore_missing_subset_index_key(self.path)
        except Exception as e:
            try:
                sysOps.throw_status(
                    "WARNING: index_key preflight restore skipped for " + str(self.path) + ": " + str(e)
                )
            except Exception:
                pass

        if not sysOps.check_file_exists("index_key.npy",self.path):
            self.Npt_tp1, self.Npt_tp2 = reindex_input_files(self.path)
            self.index_key = np.load(self.path + "index_key.npy")[:,1]
        else:
            self.index_key = np.load(self.path + "index_key.npy")
            self.Npt_tp1 = np.sum(self.index_key[:,0] == 0)
            self.Npt_tp2 = np.sum(self.index_key[:,0] == 1)
            self.index_key = self.index_key[:,1]
            
        # Primary (legacy) sparse link matrix: typically bipartite and stored one-way.
        self.link_data = load_npz(self.path + "link_assoc_reindexed.npz").tocsr()
        
        csr = self.link_data
        csr += csr.T
        csr_base_rowsum = np.array(csr.astype(bool).sum(axis=1)).flatten()
        csr_base_rownorm_alpha = np.array(csr.astype(bool).sum(axis=1)).flatten()
        alpha = 2
        for _ in range(1,alpha):
            csr_base_rownorm_alpha = scipy.sparse.diags(1.0/csr_base_rowsum) @ (csr.astype(bool) @ csr_base_rownorm_alpha)
        print(str(csr_base_rownorm_alpha))
        self.link_data = scipy.sparse.triu(scipy.sparse.diags(np.sqrt(csr_base_rownorm_alpha)) @ csr @  scipy.sparse.diags(np.sqrt(csr_base_rownorm_alpha)))
        del csr
        
        self.Npts = int(max(self.index_key.shape[0], self.link_data.shape[0]))

        self.Npts = int(self.link_data.shape[0])
    
        if self.print_status:   
            sysOps.throw_status('Data loaded with Npt_tp1=' + str(self.Npt_tp1) + ', Npt_tp2=' + str(self.Npt_tp2) + '. Adding link counts ...')
                           
        self.sum_pt_tp1_link = np.array(self.link_data.sum(axis=1)).flatten()
        self.sum_pt_tp2_link = np.array(self.link_data.sum(axis=0)).flatten()
    
        self.Nassoc = self.link_data.data.shape[0]
        self.Nlink = np.sum(self.link_data.data)
        
        # initiate amplification factors
        valid_pt_tp1_indices = np.array(self.sum_pt_tp1_link > 0)
        valid_pt_tp2_indices = np.array(self.sum_pt_tp2_link > 0)
        
        if self.print_status:
            sysOps.throw_status('Data read-in complete. Found ' + str(np.sum(~valid_pt_tp1_indices)) + ' empty type-1 indices and ' + str(np.sum(~valid_pt_tp2_indices)) + ' empty type-2 indices among ' + str(valid_pt_tp1_indices.shape[0]) + ' points.')

        # NOTE: AnnData/h5ad creation is intentionally deferred to the end of optimOps.run_GSE()
        # (see _build_augmented_h5ad) so we don't pay the IO cost during every GSEobj instantiation.
 
        return
    
    
    def eigen_decomp(self,orth=False, pmax=None, rev_operator=False):
    # Assemble linear manifold from data using "local linearity" assumption
    # assumes link_data type-1- and type-2-indices at this point has non-overlapping indices
        if self.seq_evecs is not None:
            del self.seq_evecs
            self.seq_evecs = None
        if pmax == 1:
            csr_op2 = load_npz(sysOps.globaldatapath + 'transformed_matrix.npz')
            if rev_operator:
                csr_op2 += csr_op2.T
            else:
                csr_op2 = scipy.sparse.diags(np.power(np.array(csr_op2.sum(axis=1)).flatten()+1E-20,-1)).dot(csr_op2)
        else:
            csr_op2 = None
        csr_op1 = (self.link_data + self.link_data.T).tocsr()

        # Guard against isolated nodes / zero-degree rows.
        # generate_fast_preorthbasis() assumes strictly positive row mass.
        try:
            _rs = np.asarray(csr_op1.sum(axis=1)).reshape(-1)
            _iso = _rs <= 0
            if np.any(_iso):
                sysOps.throw_status(
                    "WARNING: csr_op1 has " + str(int(np.sum(_iso))) +
                    " zero-degree rows; adding unit self-loops for stability."
                )
                csr_op1 = csr_op1 + scipy.sparse.diags(_iso.astype(np.float64))
                csr_op1.eliminate_zeros()
        except Exception:
            pass

        generate_fast_preorthbasis(csr_op1, self.inference_eignum)
        get_eigs(csr_op1=csr_op1,k=self.inference_eignum,csr_op2=csr_op2,rev_operator=rev_operator)

        
        self.seq_evecs = np.load(sysOps.globaldatapath + 'evecs.npy')
        if orth:
            self.seq_evecs = orth_preserve_order(self.seq_evecs)
            np.save(sysOps.globaldatapath + 'evecs.npy',self.seq_evecs)

        self.seq_evecs = self.seq_evecs.T
        return True
    
    def reset_gaussian_parameteres(self, Xpts, K=1):
        sysOps.throw_status("Beginning bimodal fit ...")
            
        self.alphas_arr, self.Ls_arr, opt = fit_decay_multi_scale(self.link_data + self.link_data.T, Xpts, K)
        sysOps.throw_status("Performed bimodal fit, (alpha, L): " + str([self.alphas_arr,self.Ls_arr]))
        
    def calc_grad(self,X):
    
        return self.calc_grad_and_hessp(X,None)
    
    def calc_hessp(self,X,inp_vec):
    
        return self.calc_grad_and_hessp(X,inp_vec)

    def calc_grad_and_hessp(self, X, inp_vec=None):
        do_grad  = inp_vec is None
        do_hessp = not do_grad

        if self.reweighted_Nlink is None:
            if self.alphas_arr is None: # Default parameters if not set by user
                self.alphas_arr = np.array([1.0,  0.0], dtype=np.float64)
                self.Ls_arr     = np.array([1.0,  1.0], dtype=np.float64)
            else:
                order = np.argsort(self.Ls_arr)
                self.alphas_arr = self.alphas_arr[order]
                self.Ls_arr = self.Ls_arr[order]
                self.Ls_arr /= self.Ls_arr[0]

            self.Xpts = np.zeros((self.Npts, self.spat_dims), np.float64)
            csr = (self.link_data + self.link_data.T).tocsr()
            if sysOps.check_file_exists('transformed_matrix.npz'):
                sysOps.throw_status("Loading " + sysOps.globaldatapath + 'transformed_matrix.npz')
                csr_op1 = csr.copy()
                del csr
                    
                csr_op2 = load_npz(sysOps.globaldatapath + 'transformed_matrix.npz')
                csr_op2 = csr_op2 + scipy.sparse.diags(np.ones(csr_op2.shape[0],dtype=np.float64)*1E-10)
                csr_op2 = scipy.sparse.diags(np.power(np.array(csr_op2.sum(axis=1)).flatten()+1E-20,-1)).dot(csr_op2)
            
                csr =  dot2(csr_op1,csr_op2,rownorm=False)
            
            vals = csr.dot(np.ones(self.Npts, dtype=np.float64))
            
            self.ampfactors = np.log(np.maximum(1E-10, csr.dot(1.0/np.maximum(1E-10, vals))))
            self.reweighted_Nlink = 0.5 * np.sum(vals)

            sysOps.throw_status('Calculating self.gl_innerprod')
            self.gl_innerprod = self.seq_evecs.dot(csr.dot(self.seq_evecs.T))
            sysOps.throw_status('Calculating self.gl_diag')
            self.gl_diag      = self.seq_evecs.dot(scipy.sparse.diags(vals).dot(self.seq_evecs.T))
            del csr

            self.sub_pairing_count = 2 * (self.spat_dims + 1)
            self.hashings          = max(1, int(NTHREADS))
            Pmax_est = 3 * self.sub_pairing_count * self.Npts
            if Pmax_est == 0 and self.Npts > 0 : Pmax_est = self.Npts * 10
            if self.Npts == 0: Pmax_est = 1 # Avoid zero-size array for w_buff if Npts is 0

            self.w_buff      = np.zeros((Pmax_est, self.spat_dims + 1), np.float64)
            self.dXpts_buff  = np.zeros((self.Npts, self.spat_dims, self.hashings), np.float64)
            self.hessp_buff  = np.zeros_like(self.dXpts_buff)
            self.work_vec    = np.zeros(self.spat_dims, np.float64)
            self.sumw        = 0.0
            # Bucketing state (populated when pairings are ready)
            self._pair_sorted = None
            self._wt_sorted = None
            self._pair_offsets = None

        self.Ls_arr[self.Ls_arr == 0] = 1e-9
        invL2_arr  = 1.0 / (self.Ls_arr*self.Ls_arr)
        invL4_arr  = invL2_arr*invL2_arr

        X_reshaped = X.reshape(self.inference_eignum, self.spat_dims)
        current_Xpts = self.seq_evecs[:self.inference_eignum, :].T @ X_reshaped

        if self.reset_subsample and self.Npts > 0: # Added Npts > 0 condition for safety
            self.Xpts[:] = current_Xpts
            self.reset_subsample = False
            sysOps.throw_status('Calling parallel_knn')
            _, nn_pairings = parallel_knn(self.Xpts, self.sub_pairing_count, sysOps.num_workers, return_distances=False)
            nn_pairings = nn_pairings[:,1:]
            sysOps.throw_status('Done parallel_knn.')
            selected_pairings = select_points(self, self.sub_pairing_count)
            close_pairings = selected_pairings[:,:self.sub_pairing_count]
            far_pairings = selected_pairings[:,self.sub_pairing_count:]

            # Ensure pairings are not empty before accessing ampfactors
            # This can happen if Npts is very small or sub_pairing_count is 0
            def get_weights(pairings_group):
                if pairings_group.size == 0:
                    return np.empty((self.Npts, 0), dtype=np.float64)
                return np.exp(self.ampfactors[pairings_group])

            nn_weights = get_weights(nn_pairings)
            close_weights = get_weights(close_pairings)
            far_weights = get_weights(far_pairings)
            
            epsilon = 1e-12
            if nn_weights.shape[1] > 0:
                nn_weights_sum = nn_weights.sum(axis=1, keepdims=True)
                nn_weights *= (self.sub_pairing_count/self.Npts) / (nn_weights_sum + epsilon)
            if close_weights.shape[1] > 0:
                close_weights_sum = close_weights.sum(axis=1, keepdims=True)
                close_weights *= ((1.0/(2**self.spat_dims)) - (self.sub_pairing_count/self.Npts)) / (close_weights_sum + epsilon)
            if far_weights.shape[1] > 0:
                far_weights_sum = far_weights.sum(axis=1, keepdims=True)
                far_weights *= (1.0-(1.0/(2**self.spat_dims))) / (far_weights_sum + epsilon)

            eAi = np.exp(self.ampfactors[:,np.newaxis])
            if nn_weights.shape[1] > 0: nn_weights *= eAi
            if close_weights.shape[1] > 0: close_weights *= eAi
            if far_weights.shape[1] > 0: far_weights *= eAi

            self.subsample_pairing_weights = np.concatenate([nn_weights, close_weights, far_weights],axis=1).ravel()
            all_pairings_indices = np.concatenate([nn_pairings, close_pairings, far_pairings],axis=1)
            
            if all_pairings_indices.shape[1] > 0:
                self_nn_indices = np.repeat(np.arange(self.Npts,dtype=np.int32), all_pairings_indices.shape[1])
                self.subsample_pairings = np.column_stack([self_nn_indices, all_pairings_indices.ravel()]).astype(np.int32)
            else: # Handle case with no pairings
                self.subsample_pairings = np.empty((0,2), dtype=np.int32)
                self.subsample_pairing_weights = np.empty((0,), dtype=np.float64)

            P_actual = self.subsample_pairings.shape[0]
            if self.w_buff.shape[0] < P_actual:
                self.w_buff = np.zeros((P_actual, self.spat_dims + 1), np.float64)
            elif P_actual == 0: # Ensure w_buff is at least 1 for Numba if P_actual is 0
                 self.w_buff = np.zeros((1, self.spat_dims + 1), np.float64) # Or handle P_actual=0 in Numba
            else:
                self.w_buff = self.w_buff[:P_actual, :]
            self._pair_sorted   = None
            self._wt_sorted     = None
            self._pair_offsets  = None
        elif self.Npts == 0 : # No points, no pairings
             self.subsample_pairings = np.empty((0,2), dtype=np.int32)
             self.subsample_pairing_weights = np.empty((0,), dtype=np.float64)
             self.w_buff = np.zeros((1, self.spat_dims + 1), np.float64)
             self._pair_sorted   = None
             self._wt_sorted     = None
             self._pair_offsets  = None


        self.Xpts[:] = current_Xpts
        P_actual = self.subsample_pairings.shape[0]

        if P_actual == 0 and self.Npts > 0 : # if pairings became empty unexpectedly
             # Fallback for w_buff if P_actual is 0 but we might proceed
             self.w_buff = np.zeros((1, self.spat_dims + 1), np.float64)

        # --- Bucketing: ensure we have plane-partitioned pairs/weights ---
        if P_actual > 0 and (self._pair_sorted is None or
                             self._pair_sorted.shape[0] != P_actual or
                             self._pair_offsets is None or
                             self._pair_offsets.size != (self.hashings + 1)):
            self._ensure_pair_buckets()

        if do_grad:
            if self.w_buff.shape[0] < P_actual and P_actual > 0:
                 self.w_buff = np.zeros((P_actual, self.spat_dims + 1), np.float64)
            elif P_actual == 0: # Ensure w_buff is at least size 1 for Numba if P_actual is 0
                 if self.w_buff.shape[0] < 1: self.w_buff = np.zeros((1, self.spat_dims + 1), np.float64)


            if P_actual > 0 :
                # bucketed kernel: scans each pair exactly once
                self.sumw = get_dxpts_bucketed(
                    self._pair_sorted, self._wt_sorted, self._pair_offsets,
                    self.w_buff, self.dXpts_buff, self.Xpts,
                    P_actual, self.spat_dims, self.Npts, self.hashings,
                    self.alphas_arr, invL2_arr)
            else:
                self.sumw = 0.0 # No pairs, sum of weights is 0
                # Ensure dXpts_buff is zeroed if P_actual is 0, as get_dxpts won't run or zero it
                self.dXpts_buff[:] = 0.0


        if do_hessp:
            inp_vec_reshaped = inp_vec.reshape(self.inference_eignum, self.spat_dims)
            inp_pts = self.seq_evecs[:self.inference_eignum, :].T @ inp_vec_reshaped
            if P_actual > 0:
                get_hessp_bucketed(
                    self.hessp_buff,
                    self._pair_sorted, self._wt_sorted, self._pair_offsets,
                    self.w_buff,
                    self.dXpts_buff[:, :, 0],
                    self.work_vec, inp_pts, self.sumw,
                    P_actual, self.spat_dims, self.Npts, self.hashings,
                    self.alphas_arr, invL2_arr, invL4_arr)
            else:
                self.hessp_buff[:] = 0.0 # No pairs, hessian contribution is 0

        L_sq_inv_factor = 1.0  
        G    = self.gl_innerprod[:self.inference_eignum,:self.inference_eignum]
        Dmat = self.gl_diag[:self.inference_eignum,:self.inference_eignum]

        if do_grad:
            dX   = np.zeros_like(X_reshaped)
            safe_sumw = self.sumw if self.sumw > 1e-12 else 1e-12
            ll   = -np.log(safe_sumw) * self.reweighted_Nlink
            
            for d in range(self.spat_dims):
                # Contribution from sum n_ij * log w_ij (where log w_ij involves -||x-x'||^2 / L_eff^2)
                ll_dmat_g_contribution = - (X_reshaped[:, d] @ (Dmat @ X_reshaped[:, d])) \
                                     + (X_reshaped[:, d] @ (G @ X_reshaped[:, d]))
                ll += L_sq_inv_factor * ll_dmat_g_contribution

                # Gradient from sum n_ij * log w_ij
                current_dx_d_term = (Dmat @ X_reshaped[:, d] - G @ X_reshaped[:, d])
                dX[:, d] -= (2.0 * L_sq_inv_factor) * current_dx_d_term

                # Gradient from -n.. log w.. (already calculated in dXpts_buff by get_dxpts)
                if self.Npts > 0: # Avoid matmul if seq_evecs is empty due to Npts=0
                    dX[:, d] -= self.seq_evecs[:self.inference_eignum, :] @ (
                        self.dXpts_buff[:, d, 0] * (self.reweighted_Nlink / safe_sumw))
            return -ll, -dX.ravel()

        # Hessian·v
        Hvec = np.zeros_like(X_reshaped) # Initialize Hvec correctly for accumulation
        inp_vec_reshaped_for_dmat_g = inp_vec.reshape(self.inference_eignum, self.spat_dims) # Ensure this is available

        # Contribution from -n.. log w..
        if self.Npts > 0: # Avoid matmul if seq_evecs is empty due to Npts=0
             Hvec += self.reweighted_Nlink * (
                self.seq_evecs[:self.inference_eignum, :] @ self.hessp_buff[:, :, 0])
        
        # Contribution from sum n_ij * log w_ij
        for d in range(self.spat_dims): # Assuming Hessian from this part is diagonal in d blocks
            Hvec_Dmat_G_d_part = (Dmat @ inp_vec_reshaped_for_dmat_g[:, d] - G @ inp_vec_reshaped_for_dmat_g[:, d])
            Hvec[:, d] -= (2.0 * L_sq_inv_factor) * Hvec_Dmat_G_d_part

        return -Hvec.ravel()

    # -------------------- Bucketing helper --------------------
    def _ensure_pair_buckets(self):
        """
        Build plane buckets so each thread processes a contiguous slice.
        Uses modulo hashing to avoid dropping pairs when H is not a power of two.
        If you must exactly emulate old bitmask partitioning, set USE_COMPAT_MASK=True.
        """
        pair = np.ascontiguousarray(self.subsample_pairings, dtype=np.int32)
        wt   = np.ascontiguousarray(self.subsample_pairing_weights, dtype=np.float64)
        H    = int(self.hashings)

        USE_COMPAT_MASK = False  # set True to reproduce old bitmask-based plane assignment
        if USE_COMPAT_MASK:
            # Old logic: ((k+j) & hmask). WARNING: drops pairs if H is not a power of two.
            mask = 1
            while mask < H: mask <<= 1
            hmask = mask - 1
            planes = ((pair[:, 0] + pair[:, 1]) & hmask).astype(np.int64)
            # Clip to [0, H-1] so we don't overrun; this mimics old behavior that *ignored* those pairs.
            planes = np.minimum(planes, H - 1)
        else:
            # Correct, collision-uniform plane assignment
            planes = ((pair[:, 0] + pair[:, 1]) % H).astype(np.int64)

        counts = np.bincount(planes, minlength=H).astype(np.int64)
        offsets = np.empty(H + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        pair_sorted = np.empty_like(pair)
        wt_sorted   = np.empty_like(wt)
        _bucket_pairs_stable(pair, wt, planes, offsets, pair_sorted, wt_sorted)
        self._pair_sorted   = pair_sorted
        self._wt_sorted     = wt_sorted
        self._pair_offsets  = offsets

@njit(
    "void(int32[:,:], float64[:], int64[:], int64[:], int32[:,:], float64[:])",
    cache=True)
def _bucket_pairs_stable(pair, wt, planes, offsets, out_pair, out_wt):
    """Stable O(P) bucketization of (pair, wt) into contiguous plane blocks.
    planes must be in [0, H-1], offsets length H+1.
    """
    H = offsets.size - 1
    write_pos = offsets[:H].copy()
    for idx in range(pair.shape[0]):
        h = planes[idx]
        pos = write_pos[h]
        out_pair[pos, 0] = pair[idx, 0]
        out_pair[pos, 1] = pair[idx, 1]
        out_wt[pos]      = wt[idx]
        write_pos[h] = pos + 1
 


@njit(
    "float64(int32[:,:], float64[:], int64[:], float64[:,:], float64[:,:,:], "
    "float64[:,:], int64, int64, int64, int64, float64[:], float64[:])",
    fastmath=True, parallel=True, cache=True)
def get_dxpts_bucketed(pair_sorted, wt_sorted, offsets, wbuf, dX, X,
                       P, D, N, H, alphas, invL2):
    # zero output planes efficiently
    for n_idx in prange(N):
        for d_idx in range(D):
            for h_idx in range(H):
                dX[n_idx, d_idx, h_idx] = 0.0
    sumw_global = 0.0
    thread_sumws = np.zeros(H, dtype=np.float64)
    for h in prange(H):
        start = offsets[h]
        stop  = offsets[h+1]
        sw_local_to_h = 0.0
        for p in range(start, stop):
            k = pair_sorted[p, 0]
            j = pair_sorted[p, 1]
            diff_sq = 0.0
            for d_coord in range(D):
                dv = X[k, d_coord] - X[j, d_coord]
                wbuf[p, d_coord] = dv
                diff_sq += dv * dv
            wsum_mix_exp = 0.0
            wdot_mix_exp_L2 = 0.0
            for m_idx in range(alphas.size):
                w_component = alphas[m_idx] * math.exp(-diff_sq * invL2[m_idx])
                wsum_mix_exp += w_component
                wdot_mix_exp_L2 += invL2[m_idx] * w_component
            wt_pair_user = wt_sorted[p]
            wmix_final = wt_pair_user * wsum_mix_exp
            wbuf[p, D] = wmix_final  # keep for compatibility (not read by Hessian)
            sw_local_to_h += wmix_final
            gfac = -2.0 * wt_pair_user * wdot_mix_exp_L2
            for d_coord in range(D):
                g_val = gfac * wbuf[p, d_coord]
                dX[k, d_coord, h] += g_val
                dX[j, d_coord, h] -= g_val
        thread_sumws[h] = sw_local_to_h
    for h_idx in range(H):
        sumw_global += thread_sumws[h_idx]
    # Reduce all planes into plane 0 in a single pass (parallel over nodes)
    for n_idx in prange(N):
        for d_coord in range(D):
            s = dX[n_idx, d_coord, 0]
            for h_idx in range(1, H):
                s += dX[n_idx, d_coord, h_idx]
            dX[n_idx, d_coord, 0] = s
    return sumw_global

@njit(
    "void(float64[:,:,:], int32[:,:], float64[:], int64[:], float64[:,:], "
    "float64[:,:], float64[:], float64[:,:], float64, "
    "int64, int64, int64, int64, float64[:], float64[:], float64[:])",
    fastmath=True, parallel=True, cache=True)
def get_hessp_bucketed(out, pair_sorted, wt_sorted, offsets, wbuf,
                       grad_sum_dxpts, tmp_work_vec, v_pts, sumw_total,
                       P, D, N, H, alphas, invL2, invL4):
    # zero output planes
    for n_idx in prange(N):
        for d_idx in range(D):
            for h_idx in range(H):
                out[n_idx, d_idx, h_idx] = 0.0
    for h_plane_idx in prange(H):
        start = offsets[h_plane_idx]
        stop  = offsets[h_plane_idx+1]
        for p_idx in range(start, stop):
            k_node = pair_sorted[p_idx, 0]
            j_node = pair_sorted[p_idx, 1]
            diff_sq_val = 0.0
            for d_coord in range(D):
                dv = wbuf[p_idx, d_coord]
                diff_sq_val += dv * dv
            wsum_mix_exp = 0.0
            gamma_sum = 0.0
            beta_sum  = 0.0
            for m_idx in range(alphas.size):
                w_component = alphas[m_idx] * math.exp(-diff_sq_val * invL2[m_idx])
                wsum_mix_exp += w_component
                gamma_sum += invL2[m_idx] * w_component
                beta_sum  += invL4[m_idx] * w_component
            wt_pair_user = wt_sorted[p_idx]
            coeff_hsec = wt_pair_user
            for d1_coord in range(D):
                delta_X_d1 = wbuf[p_idx, d1_coord]
                acc_k_val = 0.0
                acc_j_val = 0.0
                for d2_coord in range(D):
                    is_diagonal_term = 1.0 if d1_coord == d2_coord else 0.0
                    delta_X_d2 = wbuf[p_idx, d2_coord]
                    hsec_bracket_val = coeff_hsec * (
                        4.0 * beta_sum * delta_X_d1 * delta_X_d2 -
                        2.0 * gamma_sum * is_diagonal_term)
                    delta_v_d2 = v_pts[j_node, d2_coord] - v_pts[k_node, d2_coord]
                    acc_k_val += hsec_bracket_val * delta_v_d2
                    acc_j_val += hsec_bracket_val * (-delta_v_d2)
                out[k_node, d1_coord, h_plane_idx] += acc_k_val
                out[j_node, d1_coord, h_plane_idx] += acc_j_val
    inv_sumw_total = 1.0 / (sumw_total + 1e-12)
    for n_idx in prange(N):
       for d_coord in range(D):
            sum_over_h_planes = 0.0
            for h_idx in range(H):
               sum_over_h_planes += out[n_idx, d_coord, h_idx]
            out[n_idx, d_coord, 0] = sum_over_h_planes * inv_sumw_total
    total_dot_product_grad_v = 0.0
    for n_idx in range(N):
        for d_coord in range(D):
            total_dot_product_grad_v += grad_sum_dxpts[n_idx, d_coord] * v_pts[n_idx, d_coord]
    cross_term_coeff = inv_sumw_total * inv_sumw_total * total_dot_product_grad_v
    for n_idx in prange(N):
        for d_coord in range(D):
            out[n_idx, d_coord, 0] += grad_sum_dxpts[n_idx, d_coord] * cross_term_coeff

from numpy.random import default_rng
from scipy.optimize import minimize

# ---------------------------------------------------------------------
#  Vectorised two‑scale radial kernel
# ---------------------------------------------------------------------

def _kernel_vec_multi(r: np.ndarray,
                      shell: np.ndarray,
                      alphas: np.ndarray,
                      Ls:     np.ndarray) -> np.ndarray:
    """
    Radial kernel

        w(r) = r^(d‑1) · Σ_k  α_k · exp(‑r² / L_k²)

    broadcast‑vectorised for r.shape == shell.shape.

    Parameters
    ----------
    r, shell : 1‑D arrays  (same shape)
    alphas   : (K,) non‑negative, usually summing to 1
    Ls       : (K,) positive

    Returns
    -------
    w : 1‑D array  (same shape as r)
    """
    # r[:,None] → (n,1)  so broadcasting over K
    gaussians = np.exp(- (r[:, None] ** 2) / (Ls[None, :] ** 2))      # (n,K)
    return shell * (gaussians @ alphas)                               # (n,)

def _idx_to_pair(idx: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert condensed‑form indices (0 … N*(N‑1)//2 − 1) to matrix
    coordinates (i , j) with  0 ≤ i < j < N.

    Parameters
    ----------
    idx : 1‑D ndarray[int64]
        Flat indices into the upper‑triangular part of an N×N matrix.
    N   : int
        Number of nodes.

    Returns
    -------
    i , j : ndarrays[int64]   (same shape as idx)
    """
    idx = idx.astype(np.int64, copy=False)

    # ---- row index ---------------------------------------------------
    # Formula taken from SciPy’s distance.squareform implementation
    i = (N - 2 - np.floor(np.sqrt(4 * N * (N - 1) - 8 * idx - 7) / 2.0 - 0.5)) \
        .astype(np.int64)

    # ---- column index ------------------------------------------------
    j = idx + i + 1 \
        - N * (N - 1) // 2 \
        + ((N - i) * (N - i - 1)) // 2

    # Safety check (optional – remove for production)
    # if np.any(j >= N) or np.any(j <= i) or np.any(i < 0):
    #     raise RuntimeError("Internal error in _idx_to_pair")

    return i, j


# ---------------------------------------------------------------------
#  Main fitting routine
# ---------------------------------------------------------------------
from numpy.random import default_rng
from scipy.optimize import minimize

# ---------- include _idx_to_pair from the previous answer here ------------

def fit_decay_multi_scale(csr_counts: csr_matrix,
                          coords: np.ndarray,
                          K: int,
                          *,
                          sample_pairs: int = 200_000,
                          rng_seed: int = 42,
                          initial: tuple[np.ndarray, np.ndarray] | None = None,
                          dist_eps: float = 1e-10,
                          log_floor: float = 1e-30):
    """
    Fast Poisson MLE for

        w(r) = r^(d‑1) Σ_{k=1..K} α_k exp(‑r² / L_k²)

    with α_k ≥ 0,  Σ α_k = 1.

    Returns
    -------
    alphas_arr : (K,)  α_k
    Ls_arr     : (K,)  L_k
    result     : scipy.optimize.OptimizeResult
    """
    rng = default_rng(rng_seed)
    N, d = coords.shape
    if csr_counts.shape != (N, N):
        raise ValueError("csr_counts shape must equal coords.shape[0].")
    if K < 1:
        raise ValueError("K must be ≥1.")

    # ---------- 1. Exact numerator (observed edges) -----------------
    coo = csr_counts.tocoo()
    mask = (coo.row < coo.col) & (coo.data > 0)
    if not np.any(mask):
        raise ValueError("No positive edges in csr_counts.")

    i_obs = coo.row[mask]
    j_obs = coo.col[mask]
    N_ij  = coo.data[mask].astype(np.float64)

    r_obs = np.linalg.norm(coords[i_obs] - coords[j_obs], axis=1).clip(min=dist_eps)
    shell_obs = 1.0 if d == 1 else r_obs ** (d - 1)
    N_tot = N_ij.sum()

    # ---------- 2. Monte‑Carlo panel for Σ w ------------------------
    total_pairs = N * (N - 1) // 2
    flat_idx    = rng.choice(total_pairs, size=sample_pairs, replace=False)
    i_samp, j_samp = _idx_to_pair(flat_idx, N)

    r_samp     = np.linalg.norm(coords[i_samp] - coords[j_samp], axis=1).clip(min=dist_eps)
    shell_samp = 1.0 if d == 1 else r_samp ** (d - 1)
    pair_scale = total_pairs / sample_pairs

    # ---------- 3.  Parameterisation --------------------------------
    # We work with:
    #   θ = (η_1…η_K) unconstrained  → α_k = softmax(η)_k
    #   φ = log L_k                 → L_k = exp(φ_k)
    #
    # Concatenate into one vector of length 2K.
    def unpack(theta):
        eta   = theta[:K]
        phi   = theta[K:]
        alphas = np.exp(eta)
        alphas /= alphas.sum()               # simplex (∑α=1, α>0)
        Ls = np.exp(phi)                     # positive
        return alphas, Ls

    # ---------- 4.  Negative log‑likelihood -------------------------
    def nll(theta):
        alphas, Ls = unpack(theta)

        # Numerator (exact)
        log_w_obs = np.log(np.maximum(
            _kernel_vec_multi(r_obs, shell_obs, alphas, Ls), log_floor))
        term1 = (N_ij * log_w_obs).sum()

        # Denominator (Monte‑Carlo)
        Sw = pair_scale * _kernel_vec_multi(r_samp, shell_samp, alphas, Ls).sum()
        if Sw <= 0 or not np.isfinite(Sw):
            return np.inf
        term2 = N_tot * np.log(Sw)

        return term2 - term1           # Poisson NLL (const. terms dropped)

    # ---------- 5.  Initial guess -----------------------------------
    if initial is None:
        # α_k = 1/K  ;  L_k spaced logarithmically around median distance
        med = np.median(r_obs)
        alphas0 = np.full(K, 1.0 / K)
        # geometric progression centred at med
        span = 4.0                         # default span over log‑space
        exps = np.linspace(-span, span, K)
        Ls0  = med * (2.0 ** exps)
        initial = (np.log(alphas0), np.log(Ls0))     # (η, φ)

    # flatten θ
    theta0 = np.concatenate(initial)

    # ---------- 6.  Optimisation ------------------------------------
    result = minimize(nll, theta0, method='L-BFGS-B',
                      options={'maxiter': 800, 'ftol': 1e-9, 'gtol': 1e-6})

    alphas_hat, Ls_hat = unpack(result.x)
    return alphas_hat, Ls_hat, result



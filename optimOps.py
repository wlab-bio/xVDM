from __future__ import annotations
from threads_bootstrap import NTHREADS
import numpy as np
import scipy
from annoy import AnnoyIndex
import sysOps
import os
import shutil
import faiss
import pymetis
from numpy import linalg as LA
from numpy.random import default_rng
from scipy.sparse.linalg import ArpackNoConvergence, ArpackError
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from numba import njit, prange
import json
import hashlib
import tempfile
from joblib import Parallel, delayed
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import math
from scipy.sparse.csgraph import connected_components
from contextlib import contextmanager

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    @contextmanager
    def threadpool_limits(limits=None):
        yield

# Final post-embedding clustering layout/version.
_FINAL_CLUSTER_LABELS_LAYOUT_VERSION = 2
_FINAL_CLUSTER_MIN_CLUSTER_SIZE = 50
_FINAL_HDBSCAN_MIN_SAMPLES = 50

# Shared Infomap transport-graph pruning policy used by final post-embedding
# clustering and by the final coarsening/alignment stage.
_INFOMAP_K_PER_ROW = 10
_INFOMAP_WEIGHT_FRACTION_PER_ROW = 0.90
_INFOMAP_MIN_EDGES_PER_ROW = 3
_INFOMAP_NUM_TRIALS = 5

# Infomap policy for final post-GSEoutput clustering.
_FINAL_INFOMAP_ENFORCE_MIN_CLUSTER_SIZE = True
_FINAL_INFOMAP_MARKOV_TIME = 2.0
_FINAL_INFOMAP_PROTECT_LCC = True

# Final coarsening/alignment layout.  Coarsen-and-align no longer changes the
# pre-GSE or full_GSE route; it only builds this directory after GSEoutput.txt
# and final Infomap clustering have been produced.
_FINAL_COARSEN_DIRNAME = 'final_coarsening'
_FINAL_COARSEN_COMPONENT_DIRNAME = 'component0'
_FINAL_COARSEN_META_FILENAME = 'final_coarsening.meta.json'
_FINAL_COARSEN_ALIGN_META_FILENAME = 'final_coarsen_alignment.meta.json'

_COARSEN_ANNOTATION_BINARIZE_THRESHOLD = 2

# register_zf public run choices parsed/written by fill_params():
#   -register_zf, -slice_path, -register_zf_match_lam_dir,
#   -register_zf_match_refine_iter, -register_zf_ensemble_size.
# Ensemble process/thread counts are runtime controls set via
# REGISTER_ZF_ENSEMBLE_N_JOBS / REGZF_ENSEMBLE_N_JOBS and
# REGISTER_ZF_ENSEMBLE_THREADS_PER_WORKER / REGZF_ENSEMBLE_THREADS_PER_WORKER.
# In coarsen-and-align mode register_zf is deferred until after final Infomap
# clustering; it operates on the final-coarsened aggregate graph and leaves the
# slice-to-node map in coarsened form.
_REGISTER_ZF_MATCH_LAM_DIR = 2.0
_REGISTER_ZF_MATCH_REFINE_ITER = 2
_REGISTER_ZF_ENSEMBLE_SIZE = 16
_REGISTER_ZF_ENSEMBLE_SEED = 0
_REGISTER_ZF_ENSEMBLE_MODE = 'lexicographic'
_REGISTER_ZF_ENSEMBLE_TIE_MAX = 1023
_REGISTER_ZF_ENSEMBLE_PERTURB_UNITS = 0
_REGISTER_ZF_ENSEMBLE_REL_TOL = 0.0
_REGISTER_ZF_ENSEMBLE_ABS_TOL = 0.0
_REGISTER_ZF_ENSEMBLE_THREADS_PER_WORKER = 1
_REGISTER_ZF_NUM_POLE_PAIRS = 3
_REGISTER_ZF_GENES_PER_POLE = 3
_REGISTER_ZF_SLICE_CAPACITY_MODE = 'mass_exact'
_REGISTER_ZF_MOMENT_COV_FLOOR = 1.0e-4
_REGISTER_ZF_COARSE_MOMENTS_FILENAME = 'coarsen_align_coarse_node_mu_cov.npz'
_REGISTER_ZF_COARSE_MOMENTS_META_FILENAME = 'coarsen_align_coarse_node_mu_cov.meta.json'


def min_contig_edges(index_link_array, dataset_index_array, link_data):
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


def interleave_concat(arr1_fname, arr2_fname, out_fname, d):
    arr1 = np.load(arr1_fname, mmap_mode='r')
    arr2 = np.load(arr2_fname, mmap_mode='r')
    n = arr1.shape[0]
    c1, c2 = arr1.shape[1], arr2.shape[1]
    b1, b2 = (c1 + d - 1) // d, (c2 + d - 1) // d
    total_cols = c1 + c2
    out = np.empty((n, total_cols), dtype=arr1.dtype)
    col = 0
    for i in range(max(b1, b2)):
        if i < b1:
            s, e = i * d, min((i + 1) * d, c1)
            out[:, col:col + e - s] = arr1[:, s:e]
            col += e - s
        if i < b2:
            s, e = i * d, min((i + 1) * d, c2)
            out[:, col:col + e - s] = arr2[:, s:e]
            col += e - s
    np.save(out_fname, out)

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


def run_GSE(output_name, params, coarsen=True):
    # `coarsen` is retained for API compatibility.  The coarsen-and-align
    # route is now intentionally identical to the non-coarsen route through
    # GSEoutput.txt; final coarsening/alignment runs only after final Infomap.
    if type(params['-inference_eignum']) == list:
        fill_params(params)
    else:
        _drop_nonpublic_register_zf_params(params)

    # When run_GSE() is called programmatically (already-parsed params), some
    # supported public keys may be absent. Set minimal defaults here without
    # adding hidden register_zf implementation knobs to the run dictionary.
    params.setdefault('-final_eignum', 100)
    params.setdefault('-calc_final', None)
    params.setdefault('-scales', 1)
    params.setdefault('-coarsen_infomap', None)
    if '-coarsen_K' in params:
        try:
            sysOps.throw_status('Ignoring legacy -coarsen_K; final coarsening uses final Infomap labels directly.')
        except Exception:
            pass
        params.pop('-coarsen_K', None)
    params.setdefault('-register_zf', None)
    params.setdefault('-slice_path', None)
    params['-register_zf'] = _param_optional_str(params, '-register_zf')
    params['-slice_path'] = _param_optional_str(params, '-slice_path')
    _validate_coarsen_alignment_mode(params)
    inference_eignum = int(params['-inference_eignum'])
    inference_dim = int(params['-inference_dim'])
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
    _h5ad_label_root = params.get('_h5ad_label_root', None)
    # Match v1 non-coarsen behavior: do not build final.h5ad during GSEobj
    # construction.  Final coarsening/alignment, when requested, builds only
    # the aggregate h5ad it needs after GSEoutput.txt has been produced.
    sysOps.h5ad_build_initial = False

    sysOps.globaldatapath = str(params['-path'])
    sysOps.h5ad_include_nonunique_genes = False
    sysOps.h5ad_include_sequences = None

    sysOps.num_workers = worker_processes
    sysOps.throw_status("params = " + str(params))

    this_GSEobj = GSEobj(inference_dim, inference_eignum)

    if not sysOps.check_file_exists("orig_evecs.npy"):
        sysOps.throw_status("this_GSEobj.link_data.data.shape = " + str(this_GSEobj.link_data.data.shape))
        this_GSEobj.inference_eignum = inference_eignum
        this_GSEobj.eigen_decomp(orth=False, pmax=0)
        os.rename(sysOps.globaldatapath + "evecs.npy", sysOps.globaldatapath + "orig_evecs.npy")
        os.rename(sysOps.globaldatapath + "evals.npy", sysOps.globaldatapath + "orig_evals.npy")
    else:
        this_GSEobj.seq_evecs = np.load(sysOps.globaldatapath + "orig_evecs.npy").T

    if not sysOps.check_file_exists("orig_evecs_gapnorm.npy"):
        Y = rank_rotation_embedding(this_GSEobj.seq_evecs.T.dot(np.diag(1.0/np.maximum(1E-20,np.sqrt(1-np.load(sysOps.globaldatapath + "orig_evals.npy"))))))
        np.save(sysOps.globaldatapath + "orig_evecs_gapnorm.npy", Y)

    # Do not run any coarsening or alignment before full_GSE.  This preserves
    # bit-for-bit route identity with ordinary non-coarsen mode up through
    # production of GSEoutput.txt, apart from cleanup of stale legacy aligned
    # second-pass artifacts handled inside full_GSE().
    del this_GSEobj

    full_GSE(output_name, params)

    final_cluster_result = None
    need_final_clustering = bool(params['-calc_final'] is not None or (coarsen and _final_coarsen_align_enabled(params)))
    if need_final_clustering:
        # ------------------------------------------------------------------
        # Final post-embedding clustering
        #
        # This stage is intentionally isolated from the main GSE solve.
        # We keep the original HDBSCAN parameterization on FINAL coords, and
        # replace the separate hybrid_cluster dependency with several runs of
        # the updated execute_clusters_infomap() on the FINAL-embedding
        # transport graph.
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

        desired_meta = {
            "layout_version": int(_FINAL_CLUSTER_LABELS_LAYOUT_VERSION),
            "hdbscan_min_cluster_size": int(_FINAL_CLUSTER_MIN_CLUSTER_SIZE),
            "hdbscan_min_samples": int(_FINAL_HDBSCAN_MIN_SAMPLES),
            "final_infomap_min_cluster_size": int(_FINAL_CLUSTER_MIN_CLUSTER_SIZE),
            "final_infomap_enforce_min_cluster_size": bool(_FINAL_INFOMAP_ENFORCE_MIN_CLUSTER_SIZE),
            "final_infomap_markov_time": float(_FINAL_INFOMAP_MARKOV_TIME),
            "final_infomap_num_trials": int(_INFOMAP_NUM_TRIALS),
            "final_infomap_protect_lcc": bool(_FINAL_INFOMAP_PROTECT_LCC),
            "infomap_k_per_row": int(_INFOMAP_K_PER_ROW),
            "infomap_weight_fraction_per_row": float(_INFOMAP_WEIGHT_FRACTION_PER_ROW),
            "infomap_min_edges_per_row": int(_INFOMAP_MIN_EDGES_PER_ROW),

            "final_raw_connected_component_split": True,
            "final_raw_connected_component_split_graph": os.path.basename(link_path),
            "final_hdbscan_raw_connected_component_split_min_component_size": int(
                _FINAL_CLUSTER_MIN_CLUSTER_SIZE
            ),
            "final_infomap_raw_connected_component_split_min_component_size": int(
                _FINAL_CLUSTER_MIN_CLUSTER_SIZE
            ),

        }
        expected_label_cols = 2
        reuse_final_clusters = (
            (ex_mat is not None)
            and (ex_mat.shape[1] == expected_label_cols)
            and isinstance(ex_meta, dict)
            and (ex_meta == desired_meta)
        )
        need_hdbscan = not reuse_final_clusters
        need_any_final_infomap = not reuse_final_clusters
        need_final_tm = bool(need_any_final_infomap or (coarsen and _final_coarsen_align_enabled(params)))

        from clusterplot import (
            execute_clusters_hdbscan,
            execute_clusters_infomap,
            split_clusters_by_raw_connected_components,
        )

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
            labels_hdbscan = split_clusters_by_raw_connected_components(
                labels_hdbscan,
                link_csr,
                min_component_size=_FINAL_CLUSTER_MIN_CLUSTER_SIZE,
                route_name="final_hdbscan",
            ).astype(np.int32, copy=False)

        else:
            labels_hdbscan = ex_mat[:, 0].astype(np.int32, copy=False)

        # Build transformed_matrix from FINAL embedding once; reuse it across all
        # final Infomap parameterizations.
        base = os.path.splitext(os.path.basename(str(output_name)))[0]
        tm_final_name = f"transformed_matrix_final_{base}.npz"
        tm_final_path = os.path.join(sysOps.globaldatapath, tm_final_name)

        if need_final_tm:
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
        if reuse_final_clusters:
            labels_infomap = ex_mat[:, 1].astype(np.int32, copy=False)
        else:
            sysOps.throw_status(
                "Running final Infomap on transformed matrix derived from FINAL embedding."
            )
            labels_infomap = execute_clusters_infomap(
                transformed_matrix_path=tm_final_path,
                out_dir=os.path.join(sysOps.globaldatapath, "tmp", "infomap_tm_final"),
                out_name=f"tm_modules_final_{base}",
                seed=1,
                num_trials=int(_INFOMAP_NUM_TRIALS),
                silent=True,
                k_per_row=int(_INFOMAP_K_PER_ROW),
                weight_fraction_per_row=float(_INFOMAP_WEIGHT_FRACTION_PER_ROW),
                min_edges_per_row=int(_INFOMAP_MIN_EDGES_PER_ROW),
                min_cluster_size=int(_FINAL_CLUSTER_MIN_CLUSTER_SIZE),
                enforce_min_cluster_size=bool(_FINAL_INFOMAP_ENFORCE_MIN_CLUSTER_SIZE),
                infomap_markov_time=float(_FINAL_INFOMAP_MARKOV_TIME),
                protect_lcc=bool(_FINAL_INFOMAP_PROTECT_LCC),
                num_threads=num_threads,
            ).astype(np.int32, copy=False)
            labels_infomap = split_clusters_by_raw_connected_components(
                labels_infomap,
                link_csr,
                min_component_size=_FINAL_CLUSTER_MIN_CLUSTER_SIZE,
                route_name="final_infomap",
            ).astype(np.int32, copy=False)

        # Combine: hdbscan + FINAL-transport Infomap.
        cluster_labels = np.column_stack([labels_hdbscan, labels_infomap]).astype(np.int32, copy=False)

        np.save(cl_path, cluster_labels)
        try:
            with open(cl_meta_path, "w") as fh:
                json.dump(desired_meta, fh, indent=2, sort_keys=True)
        except Exception as e:
            sysOps.throw_status("Warning: could not write cluster_labels_meta.json: " + str(e))

        final_cluster_result = {
            'Xpts_final': Xpts_final,
            'link_csr': link_csr,
            'link_path': link_path,
            'tm_final_path': tm_final_path,
            'labels_infomap': labels_infomap,
            'cluster_labels_path': cl_path,
            'cluster_labels_meta_path': cl_meta_path,
            'output_base': base,
        }

    if params['-calc_final'] is not None:
        sysOps.throw_status("Calculating final : " + str(sysOps.h5ad_label_root))
        # Match v1 non-coarsen behavior: final.h5ad is deferred until the
        # end and gated by the resolved label-root directory, not by the
        # h5ad_build_initial flag used only for early/base AnnData construction.
        if sysOps.h5ad_label_root is not None and os.path.isdir(str(sysOps.h5ad_label_root)):
            _build_augmented_h5ad(
                group_path=sysOps.globaldatapath,
                gse_output_name=str(output_name),
            )

    if coarsen and _final_coarsen_align_enabled(params):
        if final_cluster_result is None:
            raise RuntimeError('Final coarsen-and-align requested, but final Infomap clustering did not run.')
        _run_final_coarsen_align_pipeline(
            params=params,
            output_name=str(output_name),
            final_cluster_result=final_cluster_result,
        )


def _resolve_h5ad_builder_options(
    group_path: str,
    *,
    label_root: str | None = None,
    binary: bool = False,
) -> dict:
    from annotation import _find_upwards, _looks_like_dna_seq_list

    include_sequences_cfg = getattr(sysOps, "h5ad_include_sequences", False)
    include_sequences = bool(include_sequences_cfg) if include_sequences_cfg is not None else None
    if include_sequences is None:
        search_root = label_root if label_root is not None else group_path

        label_pt_paths = []
        for _amp in (0, 1):
            for fn in (f"label_pt{_amp}.txt", f"label_pts{_amp}.txt"):
                lp = _find_upwards(search_root, fn)
                if lp is None:
                    lp = _find_upwards(group_path, fn)
                if lp:
                    label_pt_paths.append(lp)
                    break

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
    return {
        "label_root": label_root,
        "include_sequences": include_sequences,
        "include_nonunique_genes": include_nonunique_genes,
        "binary": bool(binary),
    }



def _find_h5ad_annotation_source_paths(
    group_path: str,
    *,
    label_root: str | None = None,
) -> list[str]:
    from annotation import _find_upwards

    group_path = str(group_path)
    out: list[str] = []
    seen: set[str] = set()

    index_key_path = os.path.join(group_path, "index_key.npy")
    if not os.path.exists(index_key_path):
        try:
            _restore_missing_subset_index_key(group_path)
        except Exception as e:
            sysOps.throw_status("Warning: could not restore missing index_key.npy: " + str(e))
    if os.path.exists(index_key_path):
        out.append(index_key_path)
        seen.add(index_key_path)

    search_roots = []
    if label_root is not None:
        search_roots.append(str(label_root))
    search_roots.append(group_path)

    for search_root in search_roots:
        for _amp in (0, 1):
            for fn in (f"label_pt{_amp}.txt", f"label_pts{_amp}.txt"):
                lp = _find_upwards(search_root, fn)
                if lp and lp not in seen:
                    out.append(lp)
                    seen.add(lp)
                    break

    return out

def _write_h5ad_atomic(adata, h5ad_out: str) -> None:
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



def _ensure_initial_h5ad(
    group_path: str,
    *,
    h5ad_filename: str = "final.h5ad",
    binary: bool = False,
) -> str:
    from annotation import build_umi_gene_anndata

    group_path = _ensure_trailing_slash(str(group_path))
    h5ad_out = os.path.join(group_path, h5ad_filename)
    label_root = getattr(sysOps, "h5ad_label_root", None)

    if os.path.exists(h5ad_out):
        return h5ad_out

    annotation_sources = _find_h5ad_annotation_source_paths(group_path, label_root=label_root)

    opts = _resolve_h5ad_builder_options(group_path, label_root=label_root, binary=binary)
    sysOps.throw_status("Building initial AnnData ...")
    adata = build_umi_gene_anndata(
        group_path=group_path,
        label_root=opts["label_root"],
        return_anndata=True,
        include_sequences=opts["include_sequences"],
        include_nonunique_genes=opts["include_nonunique_genes"],
        binary=opts["binary"],
    )

    if not hasattr(adata, "obs") or not hasattr(adata, "write_h5ad"):
        raise TypeError(
            "build_umi_gene_anndata did not return an AnnData object; got "
            + str(type(adata))
        )

    adata.uns["h5ad_build_stage"] = "base"
    adata.uns["h5ad_annotation_only"] = True
    adata.uns["h5ad_annotation_sources"] = [os.path.abspath(p) for p in annotation_sources]
    _write_h5ad_atomic(adata, h5ad_out)
    sysOps.throw_status("Saved initial AnnData to " + h5ad_out)
    return h5ad_out



def _drop_existing_gse_from_adata(adata) -> None:
    for col in [c for c in list(adata.obs.columns) if str(c).startswith("GSE_")]:
        del adata.obs[col]
    if hasattr(adata, "obsm") and "X_gse" in adata.obsm:
        del adata.obsm["X_gse"]
    for key in (
        "GSEoutput_source",
        "GSEoutput_source_kind",
        "GSEoutput_pre_coarsen_align_prior_source",
        "GSEoutput_coarsen_align_prior_refine_meta",
        "GSEoutput_coarsen_align_second_pass_operator_meta",
    ):
        if key in adata.uns:
            del adata.uns[key]



def _drop_existing_cluster_labels_from_adata(adata) -> None:
    colnames = []
    try:
        colnames.extend([str(c) for c in adata.uns.get("cluster_labels_columns", [])])
    except Exception:
        pass
    for c in list(adata.obs.columns):
        cs = str(c)
        if cs == "cluster" or cs.startswith("cluster_"):
            colnames.append(cs)
    for name in sorted(set(colnames)):
        if name in adata.obs.columns:
            del adata.obs[name]
    for key in (
        "cluster_labels_source",
        "cluster_labels_shape",
        "cluster_labels_columns",
        "cluster_labels_methods",
    ):
        if key in adata.uns:
            del adata.uns[key]



def _attach_gseoutput_to_adata(adata, gse_path: str | None) -> bool:
    _drop_existing_gse_from_adata(adata)
    if not gse_path or not os.path.exists(gse_path):
        return False

    try:
        full = np.loadtxt(gse_path, delimiter=",", dtype=np.float32)
        if full.ndim == 1:
            full = full.reshape(1, -1)
        if full.shape[1] < 2:
            return False

        idx = full[:, 0].astype(np.int64)
        coords = full[:, 1:].astype(np.float32)
        if coords.shape[0] == adata.n_obs and np.array_equal(idx, np.arange(coords.shape[0], dtype=np.int64)):
            X_gse = coords
        elif coords.shape[0] == adata.n_obs:
            X_gse = coords
        else:
            sysOps.throw_status("Warning: GSE coordinate array size does not match AnnData, mapping via explicit node_index column.")
            X_gse = np.full((adata.n_obs, coords.shape[1]), np.nan, dtype=np.float32)
            mask = (idx >= 0) & (idx < adata.n_obs)
            X_gse[idx[mask]] = coords[mask]

        gse_cols = [f"GSE_{i+1}" for i in range(X_gse.shape[1])]
        for i, col in enumerate(gse_cols):
            adata.obs[col] = X_gse[:, i]
        adata.obsm["X_gse"] = np.asarray(X_gse, dtype=np.float32)
        adata.uns["GSEoutput_source"] = os.path.basename(gse_path)

        adata.uns["GSEoutput_source_kind"] = "ordinary_full_GSE"
        return True
    except Exception as e:
        sysOps.throw_status("ERROR: could not attach GSEoutput to AnnData: " + str(e))
        raise



def _attach_cluster_labels_to_adata(adata, cl_path: str | None, *, gse_path: str | None = None) -> bool:
    _drop_existing_cluster_labels_from_adata(adata)
    if not cl_path or not os.path.exists(cl_path):
        return False

    try:
        cl_raw = np.asarray(np.load(cl_path))
        if cl_raw.ndim == 0:
            raise ValueError("cluster_labels.npy contained a scalar; expected (N,) or (N,K)")

        if cl_raw.ndim == 1:
            cl_mat = cl_raw.reshape(-1, 1)
        elif cl_raw.ndim == 2:
            cl_mat = cl_raw
        else:
            cl_mat = cl_raw.reshape(cl_raw.shape[0], -1)

        if cl_mat.shape[0] != adata.n_obs and cl_mat.shape[1] == adata.n_obs:
            cl_mat = cl_mat.T

        mapped = None
        if cl_mat.shape[0] == adata.n_obs:
            mapped = np.asarray(cl_mat, dtype=np.int32)
        else:
            if gse_path and os.path.exists(gse_path):
                gse_full = np.loadtxt(gse_path, delimiter=",", dtype=np.int64)
                if gse_full.ndim == 1:
                    gse_full = gse_full.reshape(1, -1)
                idx = gse_full[:, 0].astype(np.int64, copy=False)
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

        if mapped is None:
            return False

        n_cols = int(mapped.shape[1])
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
        adata.uns["cluster_labels_shape"] = [int(x) for x in mapped.shape]
        adata.uns["cluster_labels_columns"] = colnames
        return True
    except Exception as e:
        sysOps.throw_status("ERROR: could not attach cluster labels to AnnData: " + str(e))
        raise



def _read_h5ad_sparse_matrix_to_csr(mat):
    if hasattr(mat, "to_memory"):
        mat = mat.to_memory()
    elif hasattr(mat, "__getitem__") and not scipy.sparse.issparse(mat):
        try:
            mat = mat[:]
        except Exception:
            pass
    if scipy.sparse.issparse(mat):
        return mat.tocsr()
    return csr_matrix(np.asarray(mat))



def _binarize_sparse_for_coarsening(mat, threshold: int) -> csr_matrix:
    X = _read_h5ad_sparse_matrix_to_csr(mat).copy()
    if X.nnz > 0:
        X.data = (X.data >= threshold).astype(np.int32, copy=False)
        X.eliminate_zeros()
        X.sort_indices()
    return X



def _build_local_super_membership_csr(node2super, keep_super):
    node2super = np.asarray(node2super, dtype=np.int64).ravel()
    keep_super = np.asarray(keep_super, dtype=np.int64).ravel()
    n_keep = int(keep_super.size)
    if n_keep == 0:
        node2local = np.full(node2super.shape[0], -1, dtype=np.int64)
        return node2local, _build_membership_csr(node2local, 0, dtype=np.int32)

    max_super = int(max(
        int(np.max(node2super[node2super >= 0])) if np.any(node2super >= 0) else -1,
        int(np.max(keep_super)),
    )) + 1
    super_to_local = np.full(max_super, -1, dtype=np.int64)
    super_to_local[keep_super] = np.arange(n_keep, dtype=np.int64)

    node2local = np.full(node2super.shape[0], -1, dtype=np.int64)
    valid = (node2super >= 0) & (node2super < max_super)
    if np.any(valid):
        node2local[valid] = super_to_local[node2super[valid]]
    return node2local, _build_membership_csr(node2local, n_keep, dtype=np.int32)



def _coarsen_h5ad_sparse_matrix(parent_mat, membership_csr: csr_matrix, *, threshold: int) -> csr_matrix:
    X_bin = _binarize_sparse_for_coarsening(parent_mat, int(threshold))
    if membership_csr.shape[0] != X_bin.shape[0]:
        raise ValueError(
            "Membership matrix row count " + str(membership_csr.shape[0]) +
            " does not match sparse annotation matrix row count " + str(X_bin.shape[0]) + "."
        )
    X_coarse = (membership_csr.T @ X_bin).tocsr()
    X_coarse.sum_duplicates()
    X_coarse.eliminate_zeros()
    X_coarse.sort_indices()
    return X_coarse



def _write_gseoutput_from_coords(out_path: str, coords: np.ndarray) -> None:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    local_ids = np.arange(coords.shape[0], dtype=np.int64).reshape(-1, 1)
    out = np.concatenate([local_ids, coords], axis=1)
    fmt = '%i,' + ','.join(['%.10e' for _ in range(coords.shape[1])])
    np.savetxt(out_path, out, fmt=fmt, delimiter=',')



def _build_coarsened_h5ad_from_parent(
    parent_group_path: str,
    coarse_group_path: str,
    *,
    node2super: np.ndarray,
    keep_super: np.ndarray,
    super_size: np.ndarray,
    parent_h5ad_filename: str = "final.h5ad",
    coarse_h5ad_filename: str = "final.h5ad",
    gse_output_name: str = "GSEoutput.txt",
    annotation_binary_threshold: int = _COARSEN_ANNOTATION_BINARIZE_THRESHOLD,
    extra_source_paths: list[str] | None = None,
) -> None:
    try:
        import anndata as _anndata

        parent_group_path = _ensure_trailing_slash(str(parent_group_path))
        coarse_group_path = _ensure_trailing_slash(str(coarse_group_path))
        parent_h5ad_path = os.path.join(parent_group_path, parent_h5ad_filename)
        coarse_h5ad_path = os.path.join(coarse_group_path, coarse_h5ad_filename)
        gse_path = os.path.join(coarse_group_path, gse_output_name) if gse_output_name else None

        if not os.path.exists(parent_h5ad_path):
            _ensure_initial_h5ad(parent_group_path, h5ad_filename=parent_h5ad_filename, binary=False)

        if os.path.exists(coarse_h5ad_path):
            sysOps.throw_status("Found existing " + coarse_h5ad_path + "; skipping coarse h5ad rebuild.")
            return False

        node2super = np.asarray(node2super, dtype=np.int64).ravel()
        keep_super = np.asarray(keep_super, dtype=np.int64).ravel()
        super_size = np.asarray(super_size, dtype=np.int64).ravel()
        _, membership_csr = _build_local_super_membership_csr(node2super, keep_super)

        parent_adata = _anndata.read_h5ad(parent_h5ad_path, backed='r')
        try:
            if int(parent_adata.n_obs) != int(node2super.shape[0]):
                raise ValueError(
                    "Parent AnnData n_obs=" + str(parent_adata.n_obs) +
                    " does not match node2super length=" + str(node2super.shape[0]) + "."
                )

            X_coarse = _coarsen_h5ad_sparse_matrix(
                parent_adata.X,
                membership_csr,
                threshold=int(annotation_binary_threshold),
            )

            obs_df = pd.DataFrame(index=pd.Index(np.arange(keep_super.shape[0], dtype=np.int32), name='node_id'))
            obs_df['supernode_index'] = keep_super.astype(np.int64, copy=False)
            if keep_super.size > 0 and super_size.size > 0 and np.max(keep_super) < super_size.size:
                obs_df['n_fine_nodes'] = super_size[keep_super].astype(np.int64, copy=False)
            if 'has_label' in parent_adata.obs.columns:
                has_label = np.asarray(parent_adata.obs['has_label']).astype(np.int64, copy=False)
                obs_df['n_labeled_fine_nodes'] = np.asarray(membership_csr.T @ has_label).reshape(-1)
            if 'total_sub_reads' in parent_adata.obs.columns:
                total_sub_reads = np.asarray(parent_adata.obs['total_sub_reads']).astype(np.int64, copy=False)
                obs_df['total_sub_reads'] = np.asarray(membership_csr.T @ total_sub_reads).reshape(-1)

            var_df = parent_adata.var.copy()
            obs_df.index = obs_df.index.astype(str)
            var_df.index = var_df.index.astype(str)
            adata_coarse = _anndata.AnnData(X=X_coarse, obs=obs_df, var=var_df)

            for layer_name in list(parent_adata.layers.keys()):
                try:
                    layer_threshold = 1 if str(layer_name) == 'seq' else int(annotation_binary_threshold)
                    adata_coarse.layers[str(layer_name)] = _coarsen_h5ad_sparse_matrix(
                        parent_adata.layers[layer_name],
                        membership_csr,
                        threshold=layer_threshold,
                    )
                except Exception as e:
                    sysOps.throw_status(
                        "Warning: could not coarsen layer '" + str(layer_name) + "': " + str(e)
                    )

            for key in (
                'label_pt_paths',
                'label_pt_amps',
                'index_key_path',
                'include_sequences',
                'include_nonunique_genes',
                'include_obs_strings',
                'genome_feature_prefix',
                'h5ad_annotation_sources',
            ):
                if key in parent_adata.uns:
                    adata_coarse.uns[key] = parent_adata.uns[key]
            adata_coarse.uns['h5ad_build_stage'] = 'coarsened'
            adata_coarse.uns['h5ad_annotation_only'] = False
            adata_coarse.uns['h5ad_coarsened_from'] = os.path.abspath(parent_h5ad_path)
            adata_coarse.uns['h5ad_coarse_index_key_path'] = os.path.join(coarse_group_path, 'index_key.npy')
            adata_coarse.uns['h5ad_annotation_binary_threshold'] = int(annotation_binary_threshold)
            adata_coarse.uns['coarse_supernode_count'] = int(keep_super.shape[0])

            _attach_gseoutput_to_adata(adata_coarse, gse_path)
            _write_h5ad_atomic(adata_coarse, coarse_h5ad_path)
            sysOps.throw_status("Saved coarsened AnnData to " + coarse_h5ad_path)
            return True
        finally:
            try:
                if getattr(parent_adata, 'file', None) is not None:
                    parent_adata.file.close()
            except Exception:
                pass
    except Exception as e:
        sysOps.throw_status("ERROR: could not build/save coarsened AnnData: " + str(e))
        raise



def _build_augmented_h5ad(
    group_path: str,
    *,
    h5ad_filename: str = "final.h5ad",
    gse_output_name: str | None = "GSEoutput.txt",
    cluster_labels_name: str = "cluster_labels.npy",
    binary: bool = False,
) -> None:
    """Complete final.h5ad from the initial annotation-only AnnData written at first GSEobj construction."""
    try:
        group_path = _ensure_trailing_slash(str(group_path))
        h5ad_out = _ensure_initial_h5ad(group_path, h5ad_filename=h5ad_filename, binary=binary)
        gse_path = os.path.join(group_path, gse_output_name) if gse_output_name else None
        cl_path = os.path.join(group_path, cluster_labels_name)

        import anndata as _anndata

        sysOps.throw_status("Loading initial AnnData from " + h5ad_out)
        adata = _anndata.read_h5ad(h5ad_out)
        _attach_gseoutput_to_adata(adata, gse_path)
        _attach_cluster_labels_to_adata(adata, cl_path, gse_path=gse_path)
        adata.uns['h5ad_build_stage'] = 'complete'
        adata.uns['h5ad_annotation_only'] = False
        _write_h5ad_atomic(adata, h5ad_out)
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


def _safe_load_json(path):
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except Exception:
        return None


def _atomic_write_json(path, payload):
    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    basename = os.path.basename(path)
    fd, tmp_path = tempfile.mkstemp(prefix=basename + ".tmp.", suffix=".json", dir=dirpath, text=True)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def _atomic_save_npy(path, arr):
    dirpath = os.path.dirname(os.path.abspath(path)) or "."
    basename = os.path.basename(path)
    fd, tmp_path = tempfile.mkstemp(prefix=basename + ".tmp.", suffix=".npy", dir=dirpath)
    os.close(fd)
    try:
        np.save(tmp_path, arr)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise



@njit(parallel=True, cache=True)
def _scale_csr_rows_inplace(indptr, data, scale):
    """In-place CSR row scaling without materializing a sparse diagonal."""
    n = indptr.shape[0] - 1
    for i in prange(n):
        s = scale[i]
        for p in range(indptr[i], indptr[i + 1]):
            data[p] *= s



def _canonicalize_transport_component(W):
    """Return a sorted, finite float64 CSR transport component with no diagonal."""
    W = csr_matrix(W).astype(np.float64, copy=False).tocsr()
    W.sum_duplicates()
    W.setdiag(0.0)
    W.eliminate_zeros()
    if W.nnz:
        finite = np.isfinite(W.data)
        if not np.all(finite):
            W.data[~finite] = 0.0
            W.eliminate_zeros()
    W.sort_indices()
    return W


















def _digest_arrays_hex(*arrays):
    h = hashlib.blake2b(digest_size=16)
    for arr in arrays:
        a = np.ascontiguousarray(np.asarray(arr))
        h.update(repr(tuple(a.shape)).encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
        if a.size > 0:
            h.update(a.view(np.uint8).tobytes())
    return h.hexdigest()


def _load_npy_checkpoint(path, *, mmap_mode='r'):
    try:
        return np.load(path, mmap_mode=mmap_mode)
    except Exception as e:
        try:
            sysOps.throw_status('Ignoring unreadable checkpoint ' + str(path) + ': ' + str(e))
        except Exception:
            pass
        return None


def _load_gseoutput_if_valid(path, expected_rows, expected_dim):
    if not os.path.exists(path):
        return None
    try:
        arr = np.loadtxt(path, delimiter=',', dtype=np.float64)
    except Exception as e:
        sysOps.throw_status('Ignoring unreadable GSEoutput checkpoint ' + str(path) + ': ' + str(e))
        return None

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    expected_rows = int(expected_rows)
    expected_dim = int(expected_dim)
    if arr.shape[0] != expected_rows or arr.shape[1] < (expected_dim + 1):
        sysOps.throw_status(
            'Ignoring incompatible GSEoutput checkpoint ' + str(path) +
            ' with shape ' + str(arr.shape) + '; expected (' + str(expected_rows) + ', >= ' + str(expected_dim + 1) + ').'
        )
        return None

    local_ids = arr[:, 0].astype(np.int64, copy=False)
    if not np.array_equal(local_ids, np.arange(expected_rows, dtype=np.int64)):
        sysOps.throw_status('Ignoring incompatible GSEoutput checkpoint ' + str(path) + ': node_index column is not 0..N-1.')
        return None

    return arr


def _param_bool_flag(params, key, default=False):
    """Parse a CLI/programmatic boolean flag without changing legacy callers."""
    if params is None or key not in params:
        return bool(default)
    val = params.get(key)
    if isinstance(val, list):
        if len(val) == 0:
            return True
        val = val[0]
    if isinstance(val, bool):
        return val
    if val is None:
        return bool(default)
    if isinstance(val, (int, np.integer, float, np.floating)):
        return bool(val)
    sval = str(val).strip().lower()
    if sval in ('', '0', 'false', 'f', 'no', 'n', 'none', 'null', 'off'):
        return False
    return True


def _param_first(params, key, default=None):
    """Return a parsed/programmatic parameter value, unwrapping CLI lists."""
    if params is None or key not in params:
        return default
    val = params.get(key, default)
    if isinstance(val, list):
        return val[0] if len(val) else default
    return val




_REGISTER_ZF_NONPUBLIC_PARAM_KEYS = {
    '-register_zf_ensemble_seed',
    '-register_zf_ensemble_mode',
    '-register_zf_ensemble_tie_max',
    '-register_zf_ensemble_perturb_units',
    '-register_zf_ensemble_rel_tol',
    '-register_zf_ensemble_abs_tol',
    '-register_zf_ensemble_n_jobs',
    '-register_zf_ensemble_threads_per_worker',
    '-register_zf_ensemble_mp_start_method',
    '-register_zf_pole_pairs_json',
    '-register_zf_slice_capacity_mode',
    '-register_zf_num_pole_pairs',
    '-register_zf_genes_per_pole',
    '-register_zf_anchor_mode',
    '-register_zf_anchor_multimodal_sd_p95_norm_threshold',
    '-register_zf_prior_mean_mode',
}


def _drop_nonpublic_register_zf_params(params):
    """Drop implementation knobs from the ordinary CLI parameter surface.

    Public register_zf choices are parsed explicitly in ``fill_params()``.
    Remaining low-level choices are fixed internally or resolved from environment
    variables and written to metadata rather than ``params.txt``.
    """
    if params is None:
        return
    for key in sorted(_REGISTER_ZF_NONPUBLIC_PARAM_KEYS):
        if key in params:
            try:
                sysOps.throw_status('Ignoring non-public register_zf parameter ' + key + '; using fixed internal policy.')
            except Exception:
                pass
            params.pop(key, None)


def _env_first_int(names, default=None):
    for name in names:
        raw = os.getenv(str(name))
        if raw is None or str(raw).strip() == '':
            continue
        try:
            return int(str(raw).strip())
        except Exception:
            sysOps.throw_status('Warning: ignoring non-integer environment override ' + str(name) + '=' + str(raw))
    return default


def _env_first_str(names, default=None):
    for name in names:
        raw = os.getenv(str(name))
        if raw is None:
            continue
        val = str(raw).strip()
        if val:
            return val
    return default


def _resolve_register_zf_ensemble_runtime(ensemble_size):
    """Resolve hidden register_zf runtime controls without exposing params."""
    threads_per_worker = _env_first_int(
        ('REGISTER_ZF_ENSEMBLE_THREADS_PER_WORKER', 'REGZF_ENSEMBLE_THREADS_PER_WORKER'),
        _REGISTER_ZF_ENSEMBLE_THREADS_PER_WORKER,
    )
    threads_per_worker = int(max(1, threads_per_worker or 1))

    requested_jobs = _env_first_int(
        ('REGISTER_ZF_ENSEMBLE_N_JOBS', 'REGZF_ENSEMBLE_N_JOBS'),
        None,
    )
    if requested_jobs is None:
        cpu_cap = int(max(1, int(getattr(sysOps, 'num_workers', NTHREADS)) // threads_per_worker))
        requested_jobs = min(int(max(1, ensemble_size)), cpu_cap)
    ensemble_n_jobs = int(max(1, requested_jobs))

    mp_start_method = _env_first_str(
        ('REGISTER_ZF_ENSEMBLE_MP_START_METHOD', 'REGZF_ENSEMBLE_MP_START_METHOD'),
        None,
    )
    return ensemble_n_jobs, threads_per_worker, mp_start_method


def _param_optional_str(params, key):
    val = _param_first(params, key, None)
    if val is None:
        return None
    sval = str(val).strip()
    return sval if sval else None



def _final_coarsen_align_enabled(params) -> bool:
    """Return True for the deferred final Infomap coarsen + register_zf route."""
    if params is None:
        return False
    return (params.get('-coarsen_infomap') is not None) and (_param_optional_str(params, '-register_zf') is not None)



def _validate_coarsen_alignment_mode(params):
    """Validate the deferred final coarsen-and-align route.

    Plain non-coarsen runs are unchanged.  Coarsen without register_zf remains
    unsupported, but the supported coarsen-and-align route no longer modifies
    the second-pass GSE operator; register_zf is run only after final Infomap
    clustering on the final embedding has completed.
    """
    if params is None:
        return
    coarsen_enabled = params.get('-coarsen_infomap') is not None
    register_zf_flag = _param_optional_str(params, '-register_zf')
    slice_path = _param_optional_str(params, '-slice_path')

    if coarsen_enabled and register_zf_flag is None:
        raise ValueError(
            'coarsen-no-align mode has been removed.  Use ordinary non-coarsen '
            'mode, or specify -coarsen_infomap together with -register_zf and -slice_path.'
        )
    if register_zf_flag is not None and not coarsen_enabled:
        raise ValueError(
            '-register_zf is only supported as part of coarsen-and-align; '
            'also pass -coarsen_infomap, or remove -register_zf.'
        )
    if register_zf_flag is not None and slice_path is None:
        raise ValueError('-register_zf requires -slice_path pointing to the raw slice h5ad.')
    if coarsen_enabled and register_zf_flag is not None and params.get('-calc_final') is None:
        raise ValueError(
            '-coarsen_infomap + -register_zf now runs after final Infomap clustering and requires -calc_final '
            'so the final annotated h5ad inputs are available for register_zf.'
        )


def _remove_removed_align_kernel_artifacts(path, output_name=None):
    """Delete artifacts from the removed align-kernel/two-operator/coarsen-seed routes.

    This is a cleanup guard only.  It prevents stale aligned transformed matrices,
    stacked eigenbases, or coarse_fine_embedding seeds from changing the ordinary
    v1-compatible scalar final solve.
    """
    path = _ensure_trailing_slash(str(path))
    stale_solver_names = (
        'transformed_matrix_inner_product.npz',
        'transformed_matrix_simplex.npz',
        'transformed_matrix.two_operator.meta.json',
        'evecs.two_operator.meta.json',
        'transformed_matrix.register_zf.meta.json',
        'evecs_inner_product.npy',
        'evals_inner_product.npy',
        'evecs_simplex.npy',
        'evals_simplex.npy',
        'coarsen_spec_basis.npy',
        'coarsen_spec_basis.meta.json',
        'coarse_fine_embedding.npy',
        'coarse_fine_embedding.meta.json',
        'coarse_stack_scale_factor.npy',
        'coarsen_stack_ready.meta.json',
        # Legacy post-GSE/pre-GSE coarsen-align artifacts.  These are no
        # longer allowed to affect the ordinary full_GSE route.
        'coarsen_align_prior_refine.meta.json',
        'coarsen_align_reference_registered.npz',
        'coarsen_align_fine_node_mu_cov.npz',
        'coarsen_align_fine_node_mu_cov.meta.json',
        'orig_evecs_coarsen_align.npy',
        'orig_evecs_coarsen_align.meta.json',
        'orig_evecs_gapnorm_coarsen_align.npy',
        'orig_evecs_gapnorm_coarsen_align.meta.json',
        'orig_evecs_gapnorm_coarsen_align_stacked.npy',
        'orig_evecs_gapnorm_coarsen_align_stacked.meta.json',
        'transformed_matrix_coarsen_align.npz',
        'coarsen_align_second_pass_operator.meta.json',
    )
    had_removed_solver_artifact = False
    for name in stale_solver_names:
        fpath = os.path.join(path, name)
        if os.path.exists(fpath):
            had_removed_solver_artifact = True
            try:
                os.remove(fpath)
            except Exception:
                pass

    if output_name is not None:
        pre_refine_path = os.path.join(path, str(output_name) + '.pre_coarsen_align_prior')
        if os.path.exists(pre_refine_path):
            had_removed_solver_artifact = True
            try:
                os.remove(pre_refine_path)
            except Exception:
                pass

    if had_removed_solver_artifact:
        sysOps.throw_status(
            'Removed stale align-kernel/two-operator/coarsen-seed artifacts; '
            'forcing a v1-compatible scalar second-pass solve.'
        )
        for name in (
            'transformed_matrix.npz',
            'evecs.npy',
            'evals.npy',
            'init_evecs.npy',
            'rank_evecs.npy',
            'new_evecs.npy',
            'preorthbasis.npy',
        ):
            fpath = os.path.join(path, name)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass
        _purge_final_resume_outputs(path, output_name=output_name)
    return had_removed_solver_artifact

def _npy_shape_safely(path):
    try:
        arr = np.load(path, mmap_mode='r')
        return [int(x) for x in arr.shape]
    except Exception:
        return None


def _file_stat_payload(path):
    if not path or not os.path.exists(path):
        return None
    try:
        st = os.stat(path)
        return {
            'path': os.path.basename(path),
            'size': int(st.st_size),
            'shape': _npy_shape_safely(path) if str(path).endswith('.npy') else None,
        }
    except Exception:
        return None


def _expected_register_zf_obs_dim(inference_dim):
    """Number of slice-observed coordinate dimensions for register_zf alignment.

    register_zf supplies raw slice coordinates only.  In the 3D coarse-alignment
    route that means xy is observed and z must remain a separate graph-only block.
    In a 2D run there is no latent coordinate, so both dimensions are observed.
    """
    d = int(inference_dim)
    if d <= 0:
        raise ValueError('inference_dim must be positive; got ' + str(inference_dim))
    return d - 1 if d > 2 else d



def _register_zf_ensemble_medoid_index(ens):
    """Return the ensemble member closest to the ensemble in squared-distance medoid sense."""
    ens = np.asarray(ens, dtype=np.float64)
    if ens.ndim != 3 or ens.shape[0] <= 1:
        return 0
    E = int(ens.shape[0])
    F = ens.reshape(E, -1)
    G = F @ F.T
    sq = np.sum(F * F, axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * G, 0.0)
    score = np.nanmean(D2, axis=1)
    if not np.any(np.isfinite(score)):
        return 0
    return int(np.nanargmin(score))


def _choose_register_zf_anchor_xy_from_ensemble(ens, *, a_to_slice_stack=None):
    """Choose a concrete ensemble member for the saved exact slice-row witness.

    The witness is always a real sparse-transport assignment.  The prior used
    downstream is the ensemble mean and is computed separately.
    """
    ens = np.asarray(ens, dtype=np.float64)
    if ens.ndim != 3:
        raise ValueError('register_zf ensemble must have shape (E,M,obs_dim); got ' + str(ens.shape))

    a_to_slice_stack_arr = None
    if a_to_slice_stack is not None:
        a_to_slice_stack_arr = np.asarray(a_to_slice_stack, dtype=np.int64)
        if a_to_slice_stack_arr.ndim != 2 or a_to_slice_stack_arr.shape[:2] != ens.shape[:2]:
            raise ValueError(
                'register_zf a_to_slice stack has shape ' + str(a_to_slice_stack_arr.shape) +
                '; expected ' + str(ens.shape[:2]) + '.'
            )

    if ens.shape[0] <= 1:
        member_index = 0
        coord_sd_mean = coord_sd_p95 = coord_sd_p95_norm = 0.0
        medoid_idx = 0
        used = 'single_member'
    else:
        coord_sd = np.sqrt(np.mean(np.var(ens, axis=0, ddof=1), axis=1))
        flat = ens.reshape(-1, ens.shape[2])
        span = float(LA.norm(np.ptp(flat, axis=0)))
        if (not np.isfinite(span)) or span <= 1.0e-12:
            span = 1.0
        coord_sd_mean = float(np.nanmean(coord_sd)) if coord_sd.size else 0.0
        coord_sd_p95 = float(np.nanquantile(coord_sd, 0.95)) if coord_sd.size else 0.0
        coord_sd_p95_norm = float(coord_sd_p95 / max(span, 1.0e-12))
        medoid_idx = _register_zf_ensemble_medoid_index(ens)
        member_index = int(medoid_idx)
        used = 'medoid_member'

    selected_slice_indices = None
    if a_to_slice_stack_arr is not None:
        selected_slice_indices = np.asarray(a_to_slice_stack_arr[int(member_index)], dtype=np.int64)

    meta = {
        'anchor_coord_mode_used': used,
        'anchor_member_index': int(member_index),
        'anchor_is_exact_slice_member': True,
        'anchor_slice_indices_available': bool(selected_slice_indices is not None),
        'prior_mean_semantics': 'ensemble_mean_not_anchor_witness',
        'ensemble_coord_sd_mean': float(coord_sd_mean),
        'ensemble_coord_sd_p95': float(coord_sd_p95),
        'ensemble_coord_sd_p95_norm': float(coord_sd_p95_norm),
        'ensemble_medoid_member_index': int(medoid_idx),
    }
    return np.asarray(ens[int(member_index)], dtype=np.float64), meta, selected_slice_indices


def _save_register_zf_visualization_artifacts(
    *,
    gdp,
    coarse_registered_xy_ensemble,
    anchor_prior_coords,
    anchor_gse_coords,
    register_zf_payload,
    register_zf_resolved_config,
    inference_dim,
    obs_dim,
):
    """Persist register_zf witness/prior files for visualization/provenance.

    In the patched route these files live under final_coarsening/ and document
    the coarsened aggregate-node registration ensemble.  They are not consumed
    by full_GSE and are not lifted back to fine nodes.
    """
    gdp = _ensure_trailing_slash(str(gdp))
    ens = np.asarray(coarse_registered_xy_ensemble, dtype=np.float64)
    if ens.ndim != 3:
        raise ValueError('coarse_registered_xy_ensemble must have shape (E,M,obs_dim); got ' + str(ens.shape))
    obs_dim = int(obs_dim)
    inference_dim = int(inference_dim)
    a_to_slice_stack = register_zf_payload.get('a_to_slice') if isinstance(register_zf_payload, dict) else None
    witness_xy, anchor_meta, selected_slice_indices = _choose_register_zf_anchor_xy_from_ensemble(
        ens,
        a_to_slice_stack=a_to_slice_stack,
    )
    prior_mean_xy = np.mean(ens, axis=0)
    prior_meta = {
        'prior_mean_mode_used': 'ensemble_mean',
        'prior_mean_is_exact_slice_member': False,
    }

    anchor_prior_coords = np.asarray(anchor_prior_coords, dtype=np.float64)
    anchor_gse_coords = np.asarray(anchor_gse_coords, dtype=np.float64)
    if anchor_prior_coords.ndim != 2 or anchor_gse_coords.ndim != 2:
        raise ValueError('anchor_prior_coords and anchor_gse_coords must both be 2D arrays.')
    if anchor_prior_coords.shape != anchor_gse_coords.shape:
        raise ValueError(
            'anchor_prior_coords shape ' + str(anchor_prior_coords.shape) +
            ' does not match anchor_gse_coords shape ' + str(anchor_gse_coords.shape) + '.'
        )
    if anchor_prior_coords.shape[1] < inference_dim:
        raise ValueError('anchor coordinate arrays have fewer columns than inference_dim.')

    anchor_witness_coords = np.asarray(anchor_gse_coords, dtype=np.float64).copy()
    anchor_witness_coords[:, :obs_dim] = np.asarray(witness_xy, dtype=np.float64)

    np.save(os.path.join(gdp, 'coarse_anchor_coords_prior_mean.npy'), anchor_prior_coords)
    np.save(os.path.join(gdp, 'coarse_anchor_coords_witness.npy'), anchor_witness_coords)
    np.save(os.path.join(gdp, 'coarse_anchor_coords_registered.npy'), anchor_witness_coords)

    anchor_slice_indices_path = os.path.join(gdp, 'coarse_anchor_slice_indices.npy')
    if selected_slice_indices is not None:
        selected_slice_indices = np.asarray(selected_slice_indices, dtype=np.int64)
        if selected_slice_indices.shape[0] != anchor_witness_coords.shape[0]:
            raise ValueError(
                'coarse anchor slice-index witness length ' + str(selected_slice_indices.shape[0]) +
                ' does not match anchor row count ' + str(anchor_witness_coords.shape[0]) + '.'
            )
        np.save(anchor_slice_indices_path, selected_slice_indices)
    elif os.path.exists(anchor_slice_indices_path):
        try:
            os.remove(anchor_slice_indices_path)
        except Exception:
            pass

    _atomic_write_json(
        os.path.join(gdp, 'coarse_anchor_coords_registered.meta.json'),
        {
            'layout_version': 3,
            'source': 'register_zf visualization/provenance artifacts',
            'solver_path': 'final Infomap clustering -> final_coarsening aggregate h5ad -> register_zf',
            'coords_frame': 'raw_slice_spatial_xy_for_observed_block',
            'obs_dim': int(obs_dim),
            'inference_dim': int(inference_dim),
            'prior_mean_path': 'coarse_anchor_coords_prior_mean.npy',
            'witness_path': 'coarse_anchor_coords_witness.npy',
            'registered_alias_of': 'coarse_anchor_coords_witness.npy',
            'coarse_anchor_coords_semantics': 'ensemble_prior_mean_on_final_coarsened_nodes',
            'unregistered_gse_path': 'coarse_anchor_coords_final_gse.npy',
            'register_zf_output_dir': None if register_zf_payload is None else register_zf_payload.get('output_dir'),
            'register_zf_ensemble_npz_path': None if register_zf_payload is None else register_zf_payload.get('ensemble_npz_path'),
            'register_zf_anchor_slice_indices_path': (
                os.path.basename(anchor_slice_indices_path)
                if selected_slice_indices is not None
                else None
            ),
            'anchor_selection': anchor_meta,
            'prior_selection': prior_meta,
            'resolved_internal_config': register_zf_resolved_config,
            'no_fine_lift': True,
        },
    )
    return {
        'anchor_selection': anchor_meta,
        'prior_selection': prior_meta,
        'anchor_slice_indices_available': bool(selected_slice_indices is not None),
        'no_fine_lift': True,
    }




def _regularize_covariance_stack(cov_stack, *, cov_floor):
    """Symmetrize and add a small isotropic floor to a stack of tiny covariances."""
    cov_stack = np.asarray(cov_stack, dtype=np.float64)
    if cov_stack.ndim != 3 or cov_stack.shape[1] != cov_stack.shape[2]:
        raise ValueError('cov_stack must have shape (n,b,b); got ' + str(cov_stack.shape))
    n, b, _ = cov_stack.shape
    cov_reg = np.array(0.5 * (cov_stack + np.swapaxes(cov_stack, 1, 2)), dtype=np.float64, copy=True)
    bad = ~np.isfinite(cov_reg).all(axis=(1, 2))
    if np.any(bad):
        cov_reg[bad, :, :] = 0.0
    traces = np.trace(cov_reg, axis1=1, axis2=2)
    good = traces[np.isfinite(traces) & (traces > 0.0)] / float(max(1, b))
    scale = float(np.median(good)) if good.size else 1.0
    if (not np.isfinite(scale)) or scale <= 0.0:
        scale = 1.0
    floor_abs = max(float(cov_floor) * scale, 1.0e-12)
    diag = np.arange(b, dtype=np.int64)
    cov_reg[:, diag, diag] += float(floor_abs)
    return cov_reg, float(floor_abs)


def _coarse_node_moments_from_register_ensemble(
    coarse_registered_xy_ensemble,
    anchor_placement_coords,
    *,
    inference_dim,
    cov_floor,
):
    """Compute coarse-node mu/cov moments from the register_zf ensemble.

    The returned moments are deliberately in the final-coarsened node frame;
    no fine-node lift or fine-to-coarse transport is constructed.
    """
    ens = np.asarray(coarse_registered_xy_ensemble, dtype=np.float64)
    if ens.ndim != 3:
        raise ValueError('register_zf ensemble must have shape (E,M,obs_dim); got ' + str(ens.shape))
    E, M, obs_dim_raw = ens.shape
    d = int(inference_dim)
    obs_dim = _expected_register_zf_obs_dim(d)
    if int(obs_dim_raw) != int(obs_dim):
        raise ValueError(
            'register_zf ensemble returned ' + str(obs_dim_raw) +
            ' observed dimensions; inference_dim=' + str(d) +
            ' expects ' + str(obs_dim) + '.'
        )

    mu_obs = np.mean(ens[:, :, :obs_dim], axis=0)
    cov_obs = np.zeros((int(M), int(obs_dim), int(obs_dim)), dtype=np.float64)
    if E > 1:
        for e in range(int(E)):
            delta = ens[e, :, :obs_dim] - mu_obs
            cov_obs += np.einsum('ni,nj->nij', delta, delta, optimize=True)
        cov_obs /= float(E - 1)
    cov_obs, floor_abs = _regularize_covariance_stack(cov_obs, cov_floor=float(cov_floor))

    anchor = np.asarray(anchor_placement_coords, dtype=np.float64)
    mu = np.zeros((int(M), int(d)), dtype=np.float64)
    mu[:, :obs_dim] = mu_obs
    if d > obs_dim and anchor.ndim == 2 and anchor.shape[0] == M:
        use = min(d - obs_dim, max(0, anchor.shape[1] - obs_dim))
        if use > 0:
            mu[:, obs_dim:obs_dim + use] = anchor[:, obs_dim:obs_dim + use]

    return {
        'mu': mu,
        'cov_observed': cov_obs,
        'obs_dim': int(obs_dim),
        'cov_floor_abs': float(floor_abs),
    }


def _write_coarse_node_moments_file(*, out_root, moments, register_zf_resolved_config=None, register_zf_payload=None):
    """Persist coarse-node register_zf moments in final_coarsening/."""
    out_root = _ensure_trailing_slash(str(out_root))
    mu = np.asarray(moments['mu'], dtype=np.float64)
    cov_observed = np.asarray(moments['cov_observed'], dtype=np.float64)
    obs_dim = int(moments.get('obs_dim', cov_observed.shape[1]))
    if mu.ndim != 2:
        raise ValueError('coarse-node moments require 2D mu; got ' + str(mu.shape))
    if cov_observed.ndim != 3 or cov_observed.shape[0] != mu.shape[0]:
        raise ValueError('coarse-node cov_observed shape ' + str(cov_observed.shape) + ' is incompatible with mu shape ' + str(mu.shape))

    out_path = os.path.join(out_root, _REGISTER_ZF_COARSE_MOMENTS_FILENAME)
    tmp_path = out_path + '.tmp.npz'
    meta_path = os.path.join(out_root, _REGISTER_ZF_COARSE_MOMENTS_META_FILENAME)
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass

    np.savez(
        tmp_path,
        mu=mu,
        cov=cov_observed,
        cov_observed=cov_observed,
        obs_dim=np.asarray([obs_dim], dtype=np.int64),
        inference_dim=np.asarray([int(mu.shape[1])], dtype=np.int64),
        cov_floor_abs=np.asarray([float(moments.get('cov_floor_abs', np.nan))], dtype=np.float64),
    )
    os.replace(tmp_path, out_path)

    _atomic_write_json(
        meta_path,
        {
            'layout_version': 1,
            'mode': 'final_coarsened_register_zf_moments',
            'file': os.path.basename(out_path),
            'semantics': 'coarse-node centroids and observed-slice covariance from register_zf ensemble on the final-Infomap coarsened graph; not lifted to fine nodes and not consumed by full_GSE',
            'Npts': int(mu.shape[0]),
            'inference_dim': int(mu.shape[1]),
            'obs_dim': int(obs_dim),
            'cov_key': 'cov_observed',
            'cov_shape': [int(x) for x in cov_observed.shape],
            'mu_shape': [int(x) for x in mu.shape],
            'cov_floor_abs': float(moments.get('cov_floor_abs', np.nan)),
            'register_zf_output_dir': None if register_zf_payload is None else register_zf_payload.get('output_dir'),
            'register_zf_ensemble_npz_path': None if register_zf_payload is None else register_zf_payload.get('ensemble_npz_path'),
            'resolved_internal_config': register_zf_resolved_config,
            'no_fine_lift': True,
        },
    )
    return out_path, meta_path


def _copy_register_zf_coarse_alignment_aliases(out_root, register_zf_payload):
    """Copy register_zf aggregate-node outputs into final_coarsening/ with coarse-node names."""
    out_root = _ensure_trailing_slash(str(out_root))
    if not isinstance(register_zf_payload, dict):
        return {}
    reg_out = register_zf_payload.get('output_dir')
    if not reg_out:
        return {}
    reg_out = str(reg_out)
    aliases = {
        'aggregated_to_slice_match_csr.npz': 'coarse_node_to_slice_match_csr.npz',
        'slice_to_aggregated_match_csr.npz': 'slice_to_coarse_node_match_csr.npz',
        'aggregated_nodes_slice_mapped_coords.npz': 'coarse_nodes_slice_mapped_coords.npz',
        'aggregated_nodes_slice_mapped_coords_ensemble.npz': 'coarse_nodes_slice_mapped_coords_ensemble.npz',
        'slice_assigned_aggregated_feature_maps.npz': 'slice_assigned_coarse_node_feature_maps.npz',
        'matching_context_base.npz': 'coarse_matching_context_base.npz',
        'matching_refinement_context.npz': 'coarse_matching_refinement_context.npz',
        'slice_capacity_targets.npz': 'coarse_slice_capacity_targets.npz',
        'slice_capacity_spatial_diagnostics.json': 'coarse_slice_capacity_spatial_diagnostics.json',
        'run_metadata.json': 'coarse_register_zf_run_metadata.json',
        'ensemble_metadata.json': 'coarse_register_zf_ensemble_metadata.json',
    }
    copied = {}
    for src_name, dst_name in aliases.items():
        src = os.path.join(reg_out, src_name)
        if os.path.exists(src):
            dst = os.path.join(out_root, dst_name)
            try:
                shutil.copyfile(src, dst)
                copied[src_name] = dst_name
            except Exception as e:
                sysOps.throw_status('Warning: could not copy register_zf alias ' + src + ' -> ' + dst + ': ' + str(e))
    return copied

def _remove_register_zf_visualization_artifacts(gdp):
    gdp = _ensure_trailing_slash(str(gdp))
    for stale_name in (
        'coarse_anchor_coords_registered.npy',
        'coarse_anchor_coords_registered.meta.json',
        'coarse_anchor_coords_prior_mean.npy',
        'coarse_anchor_coords_witness.npy',
        'coarse_anchor_slice_indices.npy',
    ):
        stale_path = os.path.join(gdp, stale_name)
        if os.path.exists(stale_path):
            try:
                os.remove(stale_path)
            except Exception:
                pass






def _coarse_anchor_search_meta(source_digest, n_rows, n_cols):
    return {
        'layout_version': 2,
        'source_digest': str(source_digest),
        'n_rows': int(n_rows),
        'n_cols': int(n_cols),
    }


def _purge_final_resume_outputs(path, output_name=None):
    import glob

    targets = [
        'sample_Xpts.npy',
        'cluster_labels.npy',
        'cluster_labels_meta.json',
        'GSEoutput_meta.json',
    ]
    if output_name is not None:
        targets.append(str(output_name))
        iter_pattern = 'iter*_' + str(output_name)
    else:
        iter_pattern = 'iter*_GSEoutput.txt'

    for name in targets:
        fpath = os.path.join(path, name)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass

    for fpath in glob.glob(os.path.join(path, iter_pattern)):
        try:
            os.remove(fpath)
        except Exception:
            pass

    for fpath in glob.glob(os.path.join(path, 'transformed_matrix_final_*.npz')):
        try:
            os.remove(fpath)
        except Exception:
            pass













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
    del nn_indices, nn_indices_csr, P


def _fill_nonfinite_columns(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError('Expected a 2D coordinate array; got ' + str(arr.shape))
    out = np.array(arr, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(out)):
        for j in range(out.shape[1]):
            col = out[:, j]
            good = np.isfinite(col)
            fill = float(np.mean(col[good])) if np.any(good) else 0.0
            col[~good] = fill
            out[:, j] = col
    return out


def _approx_median_pairwise_distance(coords, *, max_points=2048):
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 1:
        return 1.0
    n = int(coords.shape[0])
    m = min(int(max_points), n)
    if m < n:
        idx = np.linspace(0, n - 1, m, dtype=np.int64)
        X = np.asarray(coords[idx], dtype=np.float64)
    else:
        X = np.asarray(coords, dtype=np.float64)
    X = _fill_nonfinite_columns(X)
    sq = np.sum(X * X, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    D2 = np.maximum(D2, 0.0)
    tri = np.triu_indices(int(X.shape[0]), k=1)
    vals = np.sqrt(D2[tri])
    vals = vals[np.isfinite(vals) & (vals > 1.0e-12)]
    if vals.size == 0:
        return 1.0
    med = float(np.median(vals))
    return med if np.isfinite(med) and med > 1.0e-12 else 1.0



def _robust_column_scale_median(coords):
    """Median robust per-column scale, with finite fallback, for candidate kNN only."""
    X = _fill_nonfinite_columns(coords)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        return 1.0
    med = np.median(X, axis=0)
    mad = 1.4826 * np.median(np.abs(X - med[None, :]), axis=0)
    good = mad[np.isfinite(mad) & (mad > 1.0e-12)]
    if good.size:
        scale = float(np.median(good))
    else:
        sd = np.std(X, axis=0)
        good = sd[np.isfinite(sd) & (sd > 1.0e-12)]
        scale = float(np.median(good)) if good.size else 1.0
    return scale if np.isfinite(scale) and scale > 1.0e-12 else 1.0


def _digest_arrays_hex_streaming(*arrays, chunk_rows=262144):
    """Content hash like _digest_arrays_hex, but without large full-array copies."""
    h = hashlib.blake2b(digest_size=16)
    for arr in arrays:
        if isinstance(arr, (bytes, bytearray)):
            h.update(b'raw-bytes')
            h.update(len(arr).to_bytes(8, byteorder='little', signed=False))
            h.update(bytes(arr))
            continue
        a = np.asarray(arr)
        h.update(repr(tuple(a.shape)).encode('utf-8'))
        h.update(str(a.dtype).encode('utf-8'))
        if a.size == 0:
            continue
        if a.ndim >= 1 and int(a.shape[0]) > int(chunk_rows):
            for s in range(0, int(a.shape[0]), int(chunk_rows)):
                e = min(int(a.shape[0]), s + int(chunk_rows))
                c = np.ascontiguousarray(a[s:e])
                h.update(c.view(np.uint8).tobytes())
        else:
            c = np.ascontiguousarray(a)
            h.update(c.view(np.uint8).tobytes())
    return h.hexdigest()


def _sparse_weight_sum_csr(mat):
    """Return the raw total sparse edge weight of a CSR-like matrix."""
    mat = mat.tocsr()
    if mat.nnz == 0:
        return 0.0
    return float(np.asarray(mat.data, dtype=np.float64).sum())
















def _build_membership_csr(node2super, M, dtype=np.float64):
    """Build an (N x M) CSR membership matrix C with C[i, node2super[i]] = 1."""
    node2super = np.asarray(node2super)
    N = node2super.shape[0]
    valid = node2super >= 0

    indptr = np.empty(N + 1, dtype=np.int64)
    indptr[0] = 0
    indptr[1:] = np.cumsum(valid.astype(np.int64))

    indices = node2super[valid].astype(np.int64, copy=False)
    data = np.ones(indices.shape[0], dtype=dtype)
    return csr_matrix((data, indices, indptr), shape=(N, M))


def _coarsen_sparse_by_mapping_matmul(W_csr, node2super, M, remove_diagonal=True):
    """Coarsen a sparse matrix W using quotient construction C^T W C."""
    C = _build_membership_csr(node2super, M)
    Wc = (C.T @ W_csr @ C).tocsr()
    if remove_diagonal:
        Wc.setdiag(0)
        Wc.eliminate_zeros()
    return Wc


def _build_supernodes_from_labels(labels, min_cluster_size):
    """Convert Infomap labels into a contiguous supernode mapping."""
    labels = np.asarray(labels)
    N = labels.shape[0]
    valid = labels >= 0
    if not np.any(valid):
        return np.full(N, -1, dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    u, c = np.unique(labels[valid], return_counts=True)
    keep_mask = c >= int(min_cluster_size)
    kept_labels = u[keep_mask].astype(np.int64, copy=False)
    kept_counts = c[keep_mask].astype(np.int64, copy=False)

    if kept_labels.shape[0] == 0:
        return np.full(N, -1, dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    order = np.argsort(kept_labels)
    kept_labels = kept_labels[order]
    kept_counts = kept_counts[order]

    max_lab = int(labels[valid].max())
    lab2super = np.full(max_lab + 1, -1, dtype=np.int64)
    lab2super[kept_labels] = np.arange(kept_labels.shape[0], dtype=np.int64)

    node2super = np.full(N, -1, dtype=np.int64)
    node2super[valid] = lab2super[labels[valid].astype(np.int64)]

    return node2super, kept_counts, kept_labels


def _compute_supernode_means_chunked(coords_source, node2super, anchor_super_ids, out_npy_path=None, chunk_size=250000):
    """Compute parent-space centroids for a selected set of supernodes in chunks."""
    if isinstance(coords_source, (str, os.PathLike)):
        coords = np.load(str(coords_source), mmap_mode='r')
    else:
        coords = np.asarray(coords_source)

    node2super = np.asarray(node2super, dtype=np.int64)
    anchor_super_ids = np.asarray(anchor_super_ids, dtype=np.int64)

    if coords.ndim != 2:
        raise ValueError(f"coords_source must resolve to a 2D array; got shape {coords.shape}")
    if node2super.shape[0] != coords.shape[0]:
        raise ValueError(
            f"node2super length {node2super.shape[0]} != coords rows {coords.shape[0]}"
        )

    m_anchor = int(anchor_super_ids.shape[0])
    d = int(coords.shape[1])
    if m_anchor == 0:
        means = np.zeros((0, d), dtype=np.float64)
        if out_npy_path is not None:
            np.save(out_npy_path, means)
        return means

    max_super = int(max(int(np.max(node2super[node2super >= 0])) if np.any(node2super >= 0) else -1, int(np.max(anchor_super_ids)))) + 1
    super_to_anchor = np.full(max_super, -1, dtype=np.int64)
    super_to_anchor[anchor_super_ids] = np.arange(m_anchor, dtype=np.int64)

    sums = np.zeros((m_anchor, d), dtype=np.float64)
    counts = np.zeros((m_anchor,), dtype=np.int64)
    N = int(coords.shape[0])

    for start in range(0, N, int(chunk_size)):
        end = min(N, start + int(chunk_size))
        super_chunk = node2super[start:end]
        valid = super_chunk >= 0
        if not np.any(valid):
            continue

        anchor_local = super_to_anchor[super_chunk[valid]]
        keep = anchor_local >= 0
        if not np.any(keep):
            continue

        anchor_local = anchor_local[keep]
        X = np.asarray(coords[start:end, :], dtype=np.float64)[valid, :][keep, :]

        counts += np.bincount(anchor_local, minlength=m_anchor).astype(np.int64, copy=False)
        for dim in range(d):
            sums[:, dim] += np.bincount(
                anchor_local,
                weights=X[:, dim],
                minlength=m_anchor,
            ).astype(np.float64, copy=False)

    if np.any(counts <= 0):
        missing = np.where(counts <= 0)[0]
        raise ValueError(
            "Some anchor supernodes had no contributing fine nodes when computing means: "
            + str(missing[:10])
        )

    means = sums / counts[:, None]
    if out_npy_path is not None:
        np.save(out_npy_path, means)
    return means


def _file_stat_payload_with_mtime(path):
    payload = _file_stat_payload(path)
    if payload is None:
        return None
    try:
        st = os.stat(path)
        payload = dict(payload)
        payload['mtime_ns'] = int(getattr(st, 'st_mtime_ns', int(st.st_mtime * 1.0e9)))
    except Exception:
        pass
    return payload


def _final_coarsening_paths(gdp):
    root = _ensure_trailing_slash(os.path.join(_ensure_trailing_slash(str(gdp)), _FINAL_COARSEN_DIRNAME))
    coarse_dir = _ensure_trailing_slash(os.path.join(root, _FINAL_COARSEN_COMPONENT_DIRNAME))
    return {
        'root': root,
        'coarse_dir': coarse_dir,
        'meta': os.path.join(root, _FINAL_COARSEN_META_FILENAME),
        'align_meta': os.path.join(root, _FINAL_COARSEN_ALIGN_META_FILENAME),
        'labels': os.path.join(root, 'infomap_labels.npy'),
        'labels_meta': os.path.join(root, 'infomap_labels.meta.json'),
        'node2super': os.path.join(root, 'node2super.npy'),
        'super_size': os.path.join(root, 'super_size.npy'),
        'kept_cluster_labels': os.path.join(root, 'kept_cluster_labels.npy'),
        'keep_super': os.path.join(root, 'largest_component_supernodes.npy'),
        'coarse_anchor_supernodes': os.path.join(root, 'coarse_anchor_supernodes.npy'),
        'coarse_anchor_search_coords': os.path.join(root, 'coarse_anchor_search_coords.npy'),
        'coarse_anchor_search_meta': os.path.join(root, 'coarse_anchor_search_coords.meta.json'),
        'coarse_anchor_coords': os.path.join(root, 'coarse_anchor_coords.npy'),
        'coarse_anchor_coords_final_gse': os.path.join(root, 'coarse_anchor_coords_final_gse.npy'),
        'coarse_link': os.path.join(coarse_dir, 'link_assoc_reindexed.npz'),
        'coarse_tm': os.path.join(coarse_dir, 'transformed_matrix.npz'),
        'coarse_index_key': os.path.join(coarse_dir, 'index_key.npy'),
        'coarse_gse': os.path.join(coarse_dir, 'GSEoutput.txt'),
        'coarse_gse_meta': os.path.join(coarse_dir, 'GSEoutput_meta.json'),
        'coarse_input_meta': os.path.join(coarse_dir, 'coarse_input_meta.json'),
        'coarse_h5ad': os.path.join(coarse_dir, 'final.h5ad'),
        'coarse_moments': os.path.join(root, _REGISTER_ZF_COARSE_MOMENTS_FILENAME),
        'coarse_moments_meta': os.path.join(root, _REGISTER_ZF_COARSE_MOMENTS_META_FILENAME),
    }


def _final_coarsen_align_request(gdp, params, output_name, final_cluster_result):
    labels = np.asarray(final_cluster_result['labels_infomap'], dtype=np.int32).reshape(-1)
    label_digest = _digest_arrays_hex_streaming(labels)
    return {
        'layout_version': 2,
        'mode': 'final_infomap_coarsen_register_zf_alignment',
        'output_name': str(output_name),
        'supernode_policy': 'one_supernode_per_nonnegative_final_infomap_label',
        'inference_dim': int(params['-inference_dim']),
        'inference_eignum': int(params['-inference_eignum']),
        'register_zf': _param_optional_str(params, '-register_zf'),
        'slice_path': None if _param_optional_str(params, '-slice_path') is None else os.path.abspath(_param_optional_str(params, '-slice_path')),
        'match_lam_dir': float(_param_first(params, '-register_zf_match_lam_dir', _REGISTER_ZF_MATCH_LAM_DIR)),
        'match_refine_iter': int(_param_first(params, '-register_zf_match_refine_iter', _REGISTER_ZF_MATCH_REFINE_ITER)),
        'ensemble_size': int(_param_first(params, '-register_zf_ensemble_size', _REGISTER_ZF_ENSEMBLE_SIZE)),
        'final_gse_file': _file_stat_payload_with_mtime(os.path.join(_ensure_trailing_slash(str(gdp)), str(output_name))),
        'final_transport_file': _file_stat_payload_with_mtime(final_cluster_result.get('tm_final_path')),
        'final_cluster_labels_file': _file_stat_payload_with_mtime(final_cluster_result.get('cluster_labels_path')),
        'final_cluster_labels_meta_file': _file_stat_payload_with_mtime(final_cluster_result.get('cluster_labels_meta_path')),
        'raw_link_file': _file_stat_payload_with_mtime(final_cluster_result.get('link_path')),
        'labels_infomap_digest': label_digest,
        'n_fine_nodes': int(labels.shape[0]),
        'no_fine_lift': True,
    }

def _final_coarsening_meta_matches(meta, request):
    return isinstance(meta, dict) and meta.get('request') == request


def _run_final_coarsen_align_pipeline(params, output_name, final_cluster_result, *, force=False):
    """Build final Infomap coarsening outputs and run register_zf on them.

    This is the only coarsen-and-align path after the patch.  It consumes the
    final embedding and the final-embedding NNLS/simplex transport matrix,
    forms a coarsened aggregate graph under final_coarsening/component0, builds
    the coarsened aggregate h5ad, and runs register_zf there.  It never installs
    a different transformed_matrix.npz for full_GSE and never lifts the
    coarsened slice assignment back to fine nodes.
    """
    gdp = _ensure_trailing_slash(str(sysOps.globaldatapath))
    _validate_coarsen_alignment_mode(params)

    register_zf_flag = _param_optional_str(params, '-register_zf')
    slice_path = _param_optional_str(params, '-slice_path')
    if register_zf_flag is None:
        raise ValueError('Final coarsen-and-align requires -register_zf.')
    if slice_path is None:
        raise ValueError('-register_zf requires -slice_path pointing to the raw slice h5ad.')

    inference_dim = int(params['-inference_dim'])
    if inference_dim not in (2, 3):
        raise ValueError(
            '-register_zf direct probing is only supported for inference_dim in {2, 3}; '
            'got ' + str(inference_dim) + '.'
        )

    annotation_binary_threshold = int(params.get('-coarsen_annotation_binary_threshold', _COARSEN_ANNOTATION_BINARIZE_THRESHOLD))
    paths = _final_coarsening_paths(gdp)
    os.makedirs(paths['root'], exist_ok=True)
    os.makedirs(paths['coarse_dir'], exist_ok=True)
    os.makedirs(os.path.join(paths['coarse_dir'], 'tmp'), exist_ok=True)

    request = _final_coarsen_align_request(gdp, params, output_name, final_cluster_result)
    prev_meta = _safe_load_json(paths['meta']) if os.path.exists(paths['meta']) else None
    inputs_match = bool((not force) and _final_coarsening_meta_matches(prev_meta, request))

    Xpts_final = np.asarray(final_cluster_result['Xpts_final'], dtype=np.float64)
    labels_infomap = np.asarray(final_cluster_result['labels_infomap'], dtype=np.int64).reshape(-1)
    link_csr = final_cluster_result.get('link_csr')
    if link_csr is None:
        link_csr = load_npz(os.path.join(gdp, 'link_assoc_reindexed.npz')).tocsr()
    else:
        link_csr = link_csr.tocsr()
    tm_final_path = str(final_cluster_result['tm_final_path'])
    if not os.path.exists(tm_final_path):
        raise FileNotFoundError('Final-embedding transformed matrix not found: ' + tm_final_path)

    if labels_infomap.shape[0] != int(link_csr.shape[0]):
        raise ValueError(
            'Final Infomap labels length ' + str(labels_infomap.shape[0]) +
            ' does not match link graph rows ' + str(link_csr.shape[0]) + '.'
        )
    if Xpts_final.shape[0] != int(link_csr.shape[0]):
        raise ValueError(
            'Final GSE coordinates rows ' + str(Xpts_final.shape[0]) +
            ' does not match link graph rows ' + str(link_csr.shape[0]) + '.'
        )

    coarse_core_ready = all(os.path.exists(paths[k]) for k in (
        'node2super', 'super_size', 'kept_cluster_labels', 'keep_super',
        'coarse_link', 'coarse_tm', 'coarse_index_key', 'coarse_gse',
        'coarse_anchor_supernodes', 'coarse_anchor_coords_final_gse',
    ))

    if (not inputs_match) or (not coarse_core_ready):
        sysOps.throw_status('Building final_coarsening/component0 from final Infomap labels and final embedding transport matrix.')
        _remove_register_zf_visualization_artifacts(paths['root'])
        # If the same final_coarsening path is being reused for changed inputs,
        # force the aggregate h5ad and register_zf outputs to be regenerated.
        for stale_path in (paths['coarse_h5ad'],):
            if os.path.exists(stale_path):
                try:
                    os.remove(stale_path)
                except Exception:
                    pass
        match_dir = os.path.join(paths['coarse_dir'], f'match_result_{str(register_zf_flag)}')
        if os.path.isdir(match_dir):
            try:
                shutil.rmtree(match_dir)
            except Exception:
                pass

        node2super, super_size, kept_labels = _build_supernodes_from_labels(labels_infomap, min_cluster_size=1)
        M = int(super_size.shape[0])
        if M < 1:
            raise RuntimeError('Final Infomap coarsening produced no supernodes; cannot run register_zf alignment.')

        _atomic_save_npy(paths['labels'], labels_infomap.astype(np.int64, copy=False))
        _atomic_write_json(
            paths['labels_meta'],
            {
                'layout_version': 1,
                'source': 'final_post_GSEoutput_infomap_labels',
                'cluster_labels_path': os.path.basename(str(final_cluster_result.get('cluster_labels_path'))),
                'final_transport_path': os.path.basename(tm_final_path),
                'supernode_policy': 'one_supernode_per_nonnegative_final_infomap_label',
            },
        )
        _atomic_save_npy(paths['node2super'], np.asarray(node2super, dtype=np.int64))
        _atomic_save_npy(paths['super_size'], np.asarray(super_size, dtype=np.int64))
        _atomic_save_npy(paths['kept_cluster_labels'], np.asarray(kept_labels, dtype=np.int64))

        P_final = load_npz(tm_final_path).tocsr()
        if P_final.shape != link_csr.shape:
            raise ValueError(
                'Final transformed matrix shape ' + str(P_final.shape) +
                ' does not match link graph shape ' + str(link_csr.shape) + '.'
            )

        sysOps.throw_status('Coarsening raw link graph and final-embedding NNLS transport graph (C^T W C).')
        with threadpool_limits(limits=1):
            Lc = _coarsen_sparse_by_mapping_matmul(link_csr, node2super, M, remove_diagonal=True)
            Pc = _coarsen_sparse_by_mapping_matmul(P_final, node2super, M, remove_diagonal=True)

        sym_Lc = (Lc + Lc.T).tocsr()
        n_comp, comp_labels = connected_components(sym_Lc, directed=False, return_labels=True)
        comp_weights = np.bincount(
            comp_labels.astype(np.int64),
            weights=super_size.astype(np.float64),
            minlength=int(n_comp),
        )
        largest_comp = int(np.argmax(comp_weights))
        keep_super = np.where(comp_labels == largest_comp)[0].astype(np.int64)
        if keep_super.size < 1:
            raise RuntimeError('Final coarsening largest component is empty.')
        _atomic_save_npy(paths['keep_super'], keep_super)

        L_sub = Lc[keep_super, :][:, keep_super].tocsr()
        P_sub = Pc[keep_super, :][:, keep_super].tocsr()
        save_npz(paths['coarse_link'], L_sub)
        save_npz(paths['coarse_tm'], P_sub)

        index_key = np.zeros((keep_super.shape[0], 3), dtype=np.int64)
        index_key[:, 0] = 0
        index_key[:, 1] = keep_super
        index_key[:, 2] = np.arange(keep_super.shape[0], dtype=np.int64)
        _atomic_save_npy(paths['coarse_index_key'], index_key)

        anchor_coords = _compute_supernode_means_chunked(
            Xpts_final,
            node2super,
            keep_super,
            out_npy_path=paths['coarse_anchor_search_coords'],
        )
        anchor_coords = np.asarray(anchor_coords, dtype=np.float64)
        if anchor_coords.shape[1] < inference_dim:
            padded = np.zeros((anchor_coords.shape[0], inference_dim), dtype=np.float64)
            padded[:, :anchor_coords.shape[1]] = anchor_coords
            anchor_coords = padded
        elif anchor_coords.shape[1] > inference_dim:
            anchor_coords = anchor_coords[:, :inference_dim]

        _atomic_save_npy(paths['coarse_anchor_search_coords'], anchor_coords)
        _atomic_save_npy(paths['coarse_anchor_supernodes'], keep_super)
        _atomic_save_npy(paths['coarse_anchor_coords'], anchor_coords)
        _atomic_save_npy(paths['coarse_anchor_coords_final_gse'], anchor_coords)
        _atomic_write_json(
            paths['coarse_anchor_search_meta'],
            _coarse_anchor_search_meta(
                _digest_arrays_hex_streaming(node2super, keep_super, super_size, labels_infomap),
                keep_super.shape[0],
                anchor_coords.shape[1],
            ),
        )
        _write_gseoutput_from_coords(paths['coarse_gse'], anchor_coords)

        coarse_input_meta_payload = {
            'layout_version': 1,
            'mode': 'final_infomap_coarsening_for_register_zf_alignment',
            'request': request,
            'n_rows': int(keep_super.shape[0]),
            'inference_dim': int(inference_dim),
            'source_final_transport_matrix': os.path.basename(tm_final_path),
            'source_final_labels': os.path.basename(paths['labels']),
            'no_fine_lift': True,
        }
        _atomic_write_json(paths['coarse_input_meta'], coarse_input_meta_payload)
        _atomic_write_json(
            paths['coarse_gse_meta'],
            {
                'layout_version': 1,
                'mode': 'final_infomap_cluster_centroids',
                'n_rows': int(keep_super.shape[0]),
                'inference_dim': int(inference_dim),
                'source': 'centroids_of_final_GSEoutput_coordinates_by_final_infomap_supernode',
                'no_coarse_full_GSE_run': True,
            },
        )
        _atomic_write_json(
            paths['meta'],
            {
                'layout_version': 1,
                'request': request,
                'root': os.path.basename(paths['root'].rstrip(os.sep)),
                'coarse_dir': os.path.relpath(paths['coarse_dir'], gdp),
                'n_fine_nodes': int(link_csr.shape[0]),
                'n_supernodes_all': int(M),
                'n_supernodes_largest_component': int(keep_super.shape[0]),
                'coarsened_graph': os.path.relpath(paths['coarse_link'], paths['root']),
                'coarsened_final_transport': os.path.relpath(paths['coarse_tm'], paths['root']),
                'no_fine_lift': True,
            },
        )
    else:
        sysOps.throw_status('Reusing final_coarsening/component0 checkpoint for register_zf alignment.')
        keep_super = np.load(paths['keep_super'], mmap_mode='r')
        anchor_coords = np.load(paths['coarse_anchor_coords_final_gse'], mmap_mode='r')

    keep_super = np.asarray(keep_super, dtype=np.int64)
    anchor_coords = np.asarray(anchor_coords, dtype=np.float64)
    coarse_n = int(keep_super.shape[0])

    register_zf_force_recompute = bool(force or (not inputs_match))
    coarse_h5ad_rebuilt = _build_coarsened_h5ad_from_parent(
        parent_group_path=gdp,
        coarse_group_path=paths['coarse_dir'],
        node2super=np.load(paths['node2super'], mmap_mode='r'),
        keep_super=keep_super,
        super_size=np.load(paths['super_size'], mmap_mode='r'),
        gse_output_name='GSEoutput.txt',
        annotation_binary_threshold=annotation_binary_threshold,
        extra_source_paths=[
            paths['node2super'],
            paths['keep_super'],
            paths['coarse_index_key'],
            paths['coarse_tm'],
        ],
    )
    register_zf_force_recompute = bool(register_zf_force_recompute or coarse_h5ad_rebuilt)

    from register_zf import get_aligned_coords_ensemble

    sysOps.throw_status(
        'Running deferred register_zf.get_aligned_coords_ensemble with ZF_FLAG=' +
        str(register_zf_flag) + ' on final coarsened aggregate h5ad ' + os.path.abspath(paths['coarse_h5ad'])
    )
    ensemble_size = int(_param_first(params, '-register_zf_ensemble_size', _REGISTER_ZF_ENSEMBLE_SIZE))
    ensemble_n_jobs, ensemble_threads_per_worker, ensemble_mp_start_method = _resolve_register_zf_ensemble_runtime(ensemble_size)
    register_zf_resolved_config = {
        'ensemble_size': int(ensemble_size),
        'ensemble_seed': int(_REGISTER_ZF_ENSEMBLE_SEED),
        'ensemble_mode': str(_REGISTER_ZF_ENSEMBLE_MODE),
        'ensemble_tie_max': int(_REGISTER_ZF_ENSEMBLE_TIE_MAX),
        'ensemble_perturb_units': int(_REGISTER_ZF_ENSEMBLE_PERTURB_UNITS),
        'ensemble_rel_tol': float(_REGISTER_ZF_ENSEMBLE_REL_TOL),
        'ensemble_abs_tol': float(_REGISTER_ZF_ENSEMBLE_ABS_TOL),
        'ensemble_n_jobs_requested': int(ensemble_n_jobs),
        'ensemble_threads_per_worker': int(ensemble_threads_per_worker),
        'ensemble_mp_start_method': ensemble_mp_start_method,
        'num_pole_pairs': int(_REGISTER_ZF_NUM_POLE_PAIRS),
        'genes_per_pole': int(_REGISTER_ZF_GENES_PER_POLE),
        'pole_pairs_json': None,
        'slice_capacity_mode': str(_REGISTER_ZF_SLICE_CAPACITY_MODE),
    }
    register_zf_payload = get_aligned_coords_ensemble(
        ZF_FLAG=str(register_zf_flag),
        agg_h5ad_path=os.path.abspath(paths['coarse_h5ad']),
        slice_h5ad_path=os.path.abspath(slice_path),
        output_dir=os.path.join(paths['coarse_dir'], f'match_result_{str(register_zf_flag)}'),
        force_recompute=register_zf_force_recompute,
        force_ensemble_recompute=register_zf_force_recompute,
        ensemble_size=ensemble_size,
        ensemble_seed=int(_REGISTER_ZF_ENSEMBLE_SEED),
        ensemble_mode=str(_REGISTER_ZF_ENSEMBLE_MODE),
        ensemble_tie_max=int(_REGISTER_ZF_ENSEMBLE_TIE_MAX),
        ensemble_perturb_units=int(_REGISTER_ZF_ENSEMBLE_PERTURB_UNITS),
        ensemble_rel_tol=float(_REGISTER_ZF_ENSEMBLE_REL_TOL),
        ensemble_abs_tol=float(_REGISTER_ZF_ENSEMBLE_ABS_TOL),
        ensemble_n_jobs=int(ensemble_n_jobs),
        ensemble_threads_per_worker=int(ensemble_threads_per_worker),
        ensemble_mp_start_method=ensemble_mp_start_method,
        num_pole_pairs=int(_REGISTER_ZF_NUM_POLE_PAIRS),
        genes_per_pole=int(_REGISTER_ZF_GENES_PER_POLE),
        match_lam_dir=float(_param_first(params, '-register_zf_match_lam_dir', _REGISTER_ZF_MATCH_LAM_DIR)),
        match_refine_iter=int(_param_first(params, '-register_zf_match_refine_iter', _REGISTER_ZF_MATCH_REFINE_ITER)),
        tree_workers=int(getattr(sysOps, 'num_workers', NTHREADS)),
        pole_pairs_json=None,
        slice_capacity_mode=str(_REGISTER_ZF_SLICE_CAPACITY_MODE),
        return_payload=True,
    )

    coarse_registered_xy_ensemble = np.asarray(register_zf_payload['coords'], dtype=np.float64)
    if coarse_registered_xy_ensemble.ndim != 3 or coarse_registered_xy_ensemble.shape[1] != int(coarse_n):
        raise ValueError(
            'register_zf ensemble shape ' + str(coarse_registered_xy_ensemble.shape) +
            ' is incompatible with final coarsened node count ' + str(coarse_n) + '.'
        )
    obs_dim = _expected_register_zf_obs_dim(inference_dim)
    if int(coarse_registered_xy_ensemble.shape[2]) != int(obs_dim):
        raise ValueError(
            'register_zf ensemble returned ' + str(coarse_registered_xy_ensemble.shape[2]) +
            ' coordinate dimension(s); final coarsen/register_zf with inference_dim=' +
            str(inference_dim) + ' expects exactly ' + str(obs_dim) + '.'
        )

    coarse_registered_xy_prior_mean = np.mean(coarse_registered_xy_ensemble, axis=0)
    anchor_prior_coords = np.asarray(anchor_coords, dtype=np.float64).copy()
    anchor_prior_coords[:, :int(obs_dim)] = coarse_registered_xy_prior_mean
    viz_meta = _save_register_zf_visualization_artifacts(
        gdp=paths['root'],
        coarse_registered_xy_ensemble=coarse_registered_xy_ensemble,
        anchor_prior_coords=anchor_prior_coords,
        anchor_gse_coords=np.asarray(anchor_coords, dtype=np.float64),
        register_zf_payload=register_zf_payload,
        register_zf_resolved_config=register_zf_resolved_config,
        inference_dim=int(inference_dim),
        obs_dim=int(obs_dim),
    )
    coarse_moments = _coarse_node_moments_from_register_ensemble(
        coarse_registered_xy_ensemble,
        np.asarray(anchor_coords, dtype=np.float64),
        inference_dim=int(inference_dim),
        cov_floor=float(_REGISTER_ZF_MOMENT_COV_FLOOR),
    )
    coarse_moments_path, coarse_moments_meta_path = _write_coarse_node_moments_file(
        out_root=paths['root'],
        moments=coarse_moments,
        register_zf_resolved_config=register_zf_resolved_config,
        register_zf_payload=register_zf_payload,
    )
    copied_aliases = _copy_register_zf_coarse_alignment_aliases(paths['root'], register_zf_payload)

    alignment_meta = {
        'layout_version': 1,
        'mode': 'final_coarsened_register_zf_alignment',
        'request': request,
        'no_fine_lift': True,
        'coarse_node_count': int(coarse_n),
        'fine_node_count': int(link_csr.shape[0]),
        'final_coarsening_root': os.path.abspath(paths['root']),
        'coarse_h5ad_path': os.path.abspath(paths['coarse_h5ad']),
        'register_zf_output_dir': register_zf_payload.get('output_dir'),
        'register_zf_ensemble_npz_path': register_zf_payload.get('ensemble_npz_path'),
        'coarse_moments_path': os.path.abspath(coarse_moments_path),
        'coarse_moments_meta_path': os.path.abspath(coarse_moments_meta_path),
        'copied_coarse_alignment_aliases': copied_aliases,
        'visualization_artifacts': viz_meta,
        'coarse_aggregated_to_slice_mapping_path': None if register_zf_payload.get('output_dir') is None else os.path.join(register_zf_payload.get('output_dir'), 'aggregated_to_slice_match_csr.npz'),
        'coarse_slice_to_aggregated_mapping_path': None if register_zf_payload.get('output_dir') is None else os.path.join(register_zf_payload.get('output_dir'), 'slice_to_aggregated_match_csr.npz'),
        'coarse_registered_xy_ensemble_shape': [int(x) for x in coarse_registered_xy_ensemble.shape],
        'coarse_anchor_supernodes_path': os.path.basename(paths['coarse_anchor_supernodes']),
        'coarse_anchor_coords_final_gse_path': os.path.basename(paths['coarse_anchor_coords_final_gse']),
        'resolved_internal_config': register_zf_resolved_config,
    }
    _atomic_write_json(paths['align_meta'], alignment_meta)
    sysOps.throw_status('Deferred final coarsen/register_zf alignment complete; coarse slice-node map left in final_coarsening form.')
    return alignment_meta


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


# -----------------------------------------------------------------------------
# Legacy coarsen/register_zf second-pass cleanup
# -----------------------------------------------------------------------------

def _purge_second_pass_solver_outputs(path, output_name=None, *, remove_transformed_matrix=True):
    """Drop artifacts derived from the second-pass operator."""
    path = _ensure_trailing_slash(str(path))
    names = [
        'evecs.npy',
        'evals.npy',
        'init_evecs.npy',
        'rank_evecs.npy',
        'new_evecs.npy',
        'preorthbasis.npy',
    ]
    if remove_transformed_matrix:
        names.insert(0, 'transformed_matrix.npz')
    for name in names:
        fpath = os.path.join(path, name)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass
    _purge_final_resume_outputs(path, output_name=output_name)


def _drop_legacy_coarsen_align_second_pass_artifacts(gdp, output_name=None):
    """Remove stale pre-patch coarsen-align operator artifacts.

    The patched route must be identical to non-coarsen mode through
    GSEoutput.txt.  Therefore a promoted aligned transformed_matrix.npz from an
    older run is unsafe and forces second-pass recomputation from the ordinary
    orig_evecs_gapnorm.npy basis.
    """
    gdp = _ensure_trailing_slash(str(gdp))
    meta_path = os.path.join(gdp, 'coarsen_align_second_pass_operator.meta.json')
    aligned_tm_path = os.path.join(gdp, 'transformed_matrix_coarsen_align.npz')
    found = os.path.exists(meta_path) or os.path.exists(aligned_tm_path)
    if not found:
        return False

    sysOps.throw_status(
        'Removing legacy coarsen-align sampled-prior second-pass artifacts so full_GSE uses the ordinary non-coarsen transformed matrix.'
    )
    for name in (
        'coarsen_align_second_pass_operator.meta.json',
        'transformed_matrix_coarsen_align.npz',
        'orig_evecs_coarsen_align.npy',
        'orig_evecs_coarsen_align.meta.json',
        'orig_evecs_gapnorm_coarsen_align.npy',
        'orig_evecs_gapnorm_coarsen_align.meta.json',
        'orig_evecs_gapnorm_coarsen_align_stacked.npy',
        'orig_evecs_gapnorm_coarsen_align_stacked.meta.json',
        'coarsen_align_fine_node_mu_cov.npz',
        'coarsen_align_fine_node_mu_cov.meta.json',
        'transformed_matrix_fine_to_coarse.npz',
        'fine_to_coarse_nearest_anchor.npy',
    ):
        fpath = os.path.join(gdp, name)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass
    _purge_second_pass_solver_outputs(gdp, output_name=output_name, remove_transformed_matrix=True)
    return True


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
                if (not os.path.exists(out_fn)) or os.path.getsize(out_fn) != os.path.getsize(final_iter_fn):
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
    """Parse and persist only the route-level public GSE parameters."""
    for el in list(params):
        if type(params[el]) != list and type(params[el]) != bool:
            params[el] = [params[el]]

    _drop_nonpublic_register_zf_params(params)

    if '-inference_eignum' in params:
        params['-inference_eignum'] = int(params['-inference_eignum'][0])
    else:
        params['-inference_eignum'] = 30

    if '-inference_dim' in params:
        params['-inference_dim'] = int(params['-inference_dim'][0])
    else:
        params['-inference_dim'] = 2

    if '-final_eignum' in params:
        params['-final_eignum'] = int(params['-final_eignum'][0])
    else:
        params['-final_eignum'] = 100

    # Required for v1-compatible non-coarsen output.  Default is the v1 default.
    if '-scales' in params:
        params['-scales'] = int(params['-scales'][0])
    else:
        params['-scales'] = 1

    if '-calc_final' in params:
        raw_calc_final = params['-calc_final'][0] if len(params['-calc_final']) else None
        calc_final = '' if raw_calc_final is None else str(raw_calc_final).strip()
        params['-calc_final'] = calc_final if calc_final else None
    else:
        params['-calc_final'] = None

    params['-coarsen_infomap'] = True if '-coarsen_infomap' in params else None
    if '-coarsen_K' in params:
        try:
            sysOps.throw_status('Ignoring legacy -coarsen_K; final coarsening uses final Infomap labels directly.')
        except Exception:
            pass
        params.pop('-coarsen_K', None)

    if '-register_zf' in params:
        raw_register_zf = params['-register_zf'][0] if len(params['-register_zf']) else None
        register_zf = '' if raw_register_zf is None else str(raw_register_zf).strip()
        params['-register_zf'] = register_zf if register_zf else None
    else:
        params['-register_zf'] = None

    if '-slice_path' in params:
        raw_slice_path = params['-slice_path'][0] if len(params['-slice_path']) else None
        slice_path = '' if raw_slice_path is None else str(raw_slice_path).strip()
        params['-slice_path'] = slice_path if slice_path else None
    else:
        params['-slice_path'] = None

    if '-register_zf_match_lam_dir' in params:
        params['-register_zf_match_lam_dir'] = float(params['-register_zf_match_lam_dir'][0])
    else:
        params['-register_zf_match_lam_dir'] = _REGISTER_ZF_MATCH_LAM_DIR

    if '-register_zf_match_refine_iter' in params:
        params['-register_zf_match_refine_iter'] = int(params['-register_zf_match_refine_iter'][0])
    else:
        params['-register_zf_match_refine_iter'] = _REGISTER_ZF_MATCH_REFINE_ITER

    if '-register_zf_ensemble_size' in params:
        params['-register_zf_ensemble_size'] = int(params['-register_zf_ensemble_size'][0])
    else:
        params['-register_zf_ensemble_size'] = _REGISTER_ZF_ENSEMBLE_SIZE

    _drop_nonpublic_register_zf_params(params)

    if '-path' in params:
        params['-path'] = str(params['-path'][0])
        if not params['-path'].endswith('/'):
            params['-path'] += '//'

    _validate_coarsen_alignment_mode(params)
    if '-h5ad_include_sequences' in params:
        params['-h5ad_include_sequences'] = True

    ordered_keys = [
        '-inference_eignum',
        '-inference_dim',
        '-scales',
        '-final_eignum',
        '-calc_final',
    ]
    if params.get('-h5ad_include_sequences') is True:
        ordered_keys.append('-h5ad_include_sequences')
    if params.get('-coarsen_infomap') is not None:
        ordered_keys.extend([
            '-coarsen_infomap',
        ])
    if params.get('-register_zf') is not None:
        ordered_keys.extend([
            '-register_zf',
            '-slice_path',
            '-register_zf_match_lam_dir',
            '-register_zf_match_refine_iter',
            '-register_zf_ensemble_size',
        ])
    ordered_keys.append('-path')

    with open(params['-path'] + 'params.txt', 'w') as paramfile:
        for el in ordered_keys:
            if el not in params:
                continue
            sysOps.throw_status(el + ' ' + str(params[el]))
            if type(params[el]) == bool and params[el]:
                paramfile.write(el + '\n')
            elif type(params[el]) != bool:
                paramfile.write(el + ' ' + str(params[el]) + '\n')


from scipy.spatial.transform import Rotation
from scipy.stats import rankdata


def _haar_random_rotation(p, rng):
    """Draw a Haar-uniform random orthogonal matrix in dimension p."""
    if p == 3:
        return Rotation.random(random_state=int(rng.integers(0, 2**31 - 1))).as_matrix()

    Z = rng.standard_normal((p, p))
    Q, R = np.linalg.qr(Z)
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    Q *= d
    return Q

_RANK_POOL = ThreadPoolExecutor(max_workers=NTHREADS)

def _rank_score_columns_exact(A, scale, center, out_dtype=np.float64):
    n, p = A.shape
    S = np.empty((n, p), dtype=out_dtype)

    def _do_col(j):
        col = np.asarray(A[:, j], dtype=np.float64)
        S[:, j] = np.asarray(scale * (rankdata(col, method='average') - center), dtype=out_dtype)

    if p >= 4:
        list(_RANK_POOL.map(_do_col, range(p)))
    else:
        for j in range(p):
            _do_col(j)

    return S


def _rank_score_columns_binned(
    A, scale, center, rng, *,
    n_bins=2048, sample_size=65536, out_dtype=np.float32,
):
    n, p = A.shape
    if n == 0:
        return np.empty((0, p), dtype=out_dtype)

    B = int(max(8, min(int(n_bins), n)))
    m = int(max(1, min(int(sample_size), n)))
    sample_idx = None if m >= n else rng.choice(n, size=m, replace=False)
    quantiles = np.linspace(0.0, 1.0, num=B + 1, dtype=np.float64)

    S = np.empty((n, p), dtype=out_dtype)

    def _do_col(j):
        col = np.asarray(A[:, j], dtype=np.float64)
        sample = col if sample_idx is None else col[sample_idx]
        finite_sample = sample[np.isfinite(sample)]
        if finite_sample.size == 0:
            S[:, j] = 0.0
            return
        edges = np.quantile(finite_sample, quantiles)
        bins = np.searchsorted(edges[1:-1], col, side='right')
        counts = np.bincount(bins, minlength=B).astype(np.float64, copy=False)
        starts = np.cumsum(counts) - counts
        midranks = starts + 0.5 * (counts + 1.0)
        S[:, j] = np.asarray(scale * (midranks[bins] - center), dtype=out_dtype)

    if p >= 4:
        list(_RANK_POOL.map(_do_col, range(p)))
    else:
        for j in range(p):
            _do_col(j)

    return S


def _rank_score_columns(
    A,
    scale,
    center,
    *,
    rank_mode='exact',
    rng=None,
    n_bins=2048,
    sample_size=65536,
    out_dtype=np.float64,
):
    """Rank-transform columns of A into centered, unit-norm scores.

    Backward compatible with the previous signature:
        _rank_score_columns(A, scale, center)

    Parameters
    ----------
    rank_mode : {'exact', 'binned', 'auto'}
        - 'exact': full per-column ranking with tie handling.
        - 'binned': sampled-quantile approximation for very large n.
        - 'auto': currently resolves to 'exact' here; rank_rotation_embedding()
          chooses between exact and binned explicitly before calling.
    """
    mode = str(rank_mode).lower()
    if mode in ('exact', 'auto'):
        return _rank_score_columns_exact(A, scale, center, out_dtype=out_dtype)
    if mode in ('binned', 'approx', 'binned_approx'):
        if rng is None:
            rng = np.random.default_rng()
        return _rank_score_columns_binned(
            A,
            scale,
            center,
            rng,
            n_bins=n_bins,
            sample_size=sample_size,
            out_dtype=out_dtype,
        )
    raise ValueError(f"Unsupported rank_mode={rank_mode!r}")


def _resolve_rank_mode(rank_mode, n, exact_rank_max_n):
    mode = str(rank_mode).lower()
    if mode == 'auto':
        return 'exact' if int(n) <= int(exact_rank_max_n) else 'binned'
    if mode in ('exact', 'binned', 'approx', 'binned_approx'):
        return 'binned' if mode != 'exact' else 'exact'
    raise ValueError("rank_mode must be one of {'auto', 'exact', 'binned'}")


def _reorth_basis(U, basis_dtype=np.float32):
    """Re-orthonormalize a thin basis while preserving column order as much as possible."""
    if U is None or U.size == 0:
        return U
    U64 = orth_preserve_order(np.asarray(U, dtype=np.float64, order='F'))
    return np.asarray(U64, dtype=basis_dtype, order='F')


def _incremental_svd_update(U, s, B, rank_keep, *, basis_dtype=np.float32, residual_tol=1e-7):
    """Incrementally update a truncated SVD with a streamed column block.

    If A ≈ U diag(s) V^T is the current approximation, this updates the top
    rank_keep left singular vectors/singular values of [A, B] without storing A.
    """
    B = np.ascontiguousarray(B, dtype=basis_dtype)
    if B.ndim != 2:
        raise ValueError("B must be 2D")

    if U is None or s is None or len(s) == 0:
        Q, R = np.linalg.qr(B, mode='reduced')
        Ucore, Sh, _ = np.linalg.svd(np.asarray(R, dtype=np.float64), full_matrices=False)
        keep = min(int(rank_keep), int(Sh.shape[0]))
        if keep == 0:
            return np.empty((B.shape[0], 0), dtype=basis_dtype, order='F'), np.empty((0,), dtype=np.float64)
        U_new = Q @ np.asarray(Ucore[:, :keep], dtype=Q.dtype)
        return np.asfortranarray(U_new[:, :keep]), np.asarray(Sh[:keep], dtype=np.float64)

    U = np.asarray(U, dtype=basis_dtype, order='F')
    s = np.asarray(s, dtype=np.float64)
    r = int(U.shape[1])

    P = U.T @ B
    residual = B - U @ P

    q = 0
    Qr = None
    Rr = None
    if residual.shape[1] > 0:
        Qr, Rr = np.linalg.qr(residual, mode='reduced')
        if Rr.size > 0:
            diag = np.abs(np.diag(Rr)).astype(np.float64, copy=False)
            tol = residual_tol * float(np.max(diag)) if diag.size > 0 else 0.0
            keep_q = diag > tol
            q = int(np.sum(keep_q))
            if q > 0:
                Qr = np.ascontiguousarray(Qr[:, :q], dtype=basis_dtype)
                Rr = np.asarray(Rr[:q, :], dtype=np.float64)

    P64 = np.asarray(P, dtype=np.float64)
    Sdiag = np.diag(s)

    if q > 0:
        upper = np.concatenate([Sdiag, P64], axis=1)
        lower = np.concatenate([np.zeros((q, r), dtype=np.float64), Rr], axis=1)
        Ksmall = np.concatenate([upper, lower], axis=0)
        Ucore, Sh, _ = np.linalg.svd(Ksmall, full_matrices=False)
        keep = min(int(rank_keep), int(Sh.shape[0]))
        basis_cat = np.concatenate([U, Qr], axis=1)
        coeffs = np.asarray(Ucore[:, :keep], dtype=basis_cat.dtype)
        U_new = basis_cat @ coeffs
    else:
        Ksmall = np.concatenate([Sdiag, P64], axis=1)
        Ucore, Sh, _ = np.linalg.svd(Ksmall, full_matrices=False)
        keep = min(int(rank_keep), int(Sh.shape[0]))
        coeffs = np.asarray(Ucore[:, :keep], dtype=U.dtype)
        U_new = U @ coeffs

    return np.asfortranarray(U_new[:, :keep]), np.asarray(Sh[:keep], dtype=np.float64)


def _sampled_subspace_change(U, sample_rows, d, prev_sample_basis=None):
    """Heuristic subspace-change estimate from a fixed row sample.

    Returns
    -------
    change : float
        1 - mean(cos^2(theta_i)) over principal angles between the previous and
        current sampled d-dimensional subspaces. Smaller is more stable.
    new_sample_basis : ndarray
        Orthonormal sampled basis to cache for the next check.
    """
    if U is None or U.size == 0:
        return np.inf, prev_sample_basis

    k = int(min(d, U.shape[1]))
    if k <= 0:
        return np.inf, prev_sample_basis

    if sample_rows is None:
        sample = np.asarray(U[:, :k], dtype=np.float64, order='F')
    else:
        sample = np.asarray(U[sample_rows, :k], dtype=np.float64, order='F')

    sample_q, _ = np.linalg.qr(sample, mode='reduced')

    if prev_sample_basis is None or prev_sample_basis.shape[1] != sample_q.shape[1]:
        return np.inf, sample_q

    sigma = np.linalg.svd(prev_sample_basis.T @ sample_q, compute_uv=False)
    sigma = np.clip(sigma[:k], 0.0, 1.0)
    overlap = float(np.mean(sigma * sigma))
    return 1.0 - overlap, sample_q


def rank_rotation_embedding(
    X,
    K=None,
    d=None,
    seed=None,
    *,
    oversample=8,
    rank_mode='auto',
    exact_rank_max_n=500_000,
    approx_rank_bins=2048,
    approx_rank_sample_size=65536,
    block_cols=None,
    basis_dtype=np.float32,
    adaptive=None,
    check_every=4,
    subspace_tol=1e-2,
    patience=2,
    min_rotations=None,
    max_rotations=None,
    reorthogonalize_every=16,
    require_convergence=True,
    max_auto_rotations=None,
):
    """Memory-lean rank-rotation embedding via streamed block incremental SVD.

    This replaces the dense-Z / dense-Gram / dense-kernel implementation with a
    one-pass low-rank update over streamed rank-score blocks. It is designed for
    very large n where materializing Z = [scores_1 ... scores_K] would cause OOM.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Input embedding. Columns are centered implicitly during projection, so a
        centered copy of X is never materialized.
    K : int or None
        If adaptive is None, an explicit K means "run exactly K rotations" and
        adaptive=False. If K is None, the historical budget max(1, int(500 / p))
        is used as the default *maximum* number of rotations and adaptive=True.
    d : int or None
        Number of components to return. Defaults to p.
    seed : int or None
        Random seed.
    oversample : int
        Extra retained singular directions inside the streaming SVD to stabilize
        the leading d-dimensional subspace.
    rank_mode : {'auto', 'exact', 'binned'}
        Exact ranks are more faithful but require a full sort per streamed
        column. Binned ranks approximate the empirical CDF using sampled quantile
        bins and are much faster for very large n.
    exact_rank_max_n : int
        Auto-switch threshold for rank_mode='auto'.
    approx_rank_bins, approx_rank_sample_size : int
        Controls for the binned rank approximation.
    block_cols : int or None
        Number of rotated columns to process at once. Lower values reduce peak
        memory. Defaults to p for small exact problems and min(p, 8) otherwise.
    basis_dtype : numpy dtype
        Storage dtype for the streamed basis and score blocks. float32 is the
        intended large-scale mode.
    adaptive : bool or None
        Whether to stop early when the top-d subspace stabilizes. When None,
        adaptive defaults to (K is None).
    check_every, subspace_tol, patience, min_rotations : control adaptive stop.
    max_rotations : int or None
        Hard cap on the number of rotations. Defaults to K when K is given, else
        to max(1, int(500 / p)).
    reorthogonalize_every : int
        Periodically re-orthonormalize the streamed basis.
    require_convergence : bool
        If True and adaptive stopping is enabled, raise an error instead of
        silently returning at max_rotations when the sampled-subspace
        convergence criterion has not been satisfied.
    max_auto_rotations : int or None
        Optional safety ceiling used only when adaptive=True and the caller did
        not explicitly supply K or max_rotations. None means "keep extending
        the budget until convergence".

    Returns
    -------
    embedding : ndarray of shape (n, d)
        Rank-rotation embedding scaled by the retained singular values. The
        function returns this single matrix, not a ``(U, s)`` tuple.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n, p = X.shape
    if n < 2:
        raise ValueError(f"X must have at least 2 rows, got {n}")
    if p < 2:
        raise ValueError(f"X must have at least 2 columns, got {p}")

    if d is None:
        d = p
    d = int(d)
    if not 1 <= d <= p:
        raise ValueError(f"d must satisfy 1 <= d <= p={p}, got d={d}")

    if K is not None:
        K = int(K)
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")

    if adaptive is None:
        adaptive = (K is None)
    adaptive = bool(adaptive)
    user_supplied_max_rotations = (max_rotations is not None)
    explicit_rotation_budget = (K is not None) or user_supplied_max_rotations
    if explicit_rotation_budget and max_auto_rotations is not None:
        raise ValueError(
            "max_auto_rotations cannot be combined with an explicit rotation "
            "budget (K or max_rotations)."
        )

 
    if max_rotations is None:
        max_rotations = K if K is not None else max(1, int(500 / p))
    max_rotations = int(max_rotations)
    if max_rotations < 1:
        raise ValueError(f"max_rotations must be >= 1, got {max_rotations}")
    initial_max_rotations = int(max_rotations)
 
    if adaptive and max_auto_rotations is None and not explicit_rotation_budget:
        max_auto_rotations = max(50, 4 * initial_max_rotations)
 
    if not adaptive and K is not None:
        max_rotations = K
 
    check_every = max(1, int(check_every))
    patience = max(1, int(patience))

    if adaptive:
        if min_rotations is None:
            min_rotations = max(2, check_every)
        min_rotations = int(max(1, min_rotations))

        first_check = int(((min_rotations + check_every - 1) // check_every) * check_every)
        min_required_rotations = int(first_check + patience * check_every)

        if max_rotations < min_required_rotations:
            if explicit_rotation_budget:
                raise ValueError(
                    "adaptive=True requires enough rotation budget to evaluate "
                    "the sampled-subspace stop criterion: "
                    f"need max_rotations >= {min_required_rotations} for "
                    f"min_rotations={min_rotations}, check_every={check_every}, "
                    f"patience={patience}; got {max_rotations}."
                )
            try:
                sysOps.throw_status(
                    "rank_rotation_embedding: increasing max_rotations from "
                    f"{max_rotations} to {min_required_rotations} so the adaptive "
                    "stop criterion can actually trigger."
                )
            except Exception:
                pass
            max_rotations = min_required_rotations

        if max_auto_rotations is not None:
            max_auto_rotations = int(max_auto_rotations)
            if max_auto_rotations < min_required_rotations:
                raise ValueError(
                    "max_auto_rotations must be >= the minimum adaptive budget "
                    f"{min_required_rotations}; got {max_auto_rotations}."
                )
            max_rotations = min(max_rotations, max_auto_rotations)
        min_rotations = int(max(1, min(min_rotations, max_rotations)))
    else:
        if min_rotations is None:
            min_rotations = max_rotations
        min_rotations = int(max(1, min(min_rotations, max_rotations)))
 

    chosen_rank_mode = _resolve_rank_mode(rank_mode, n, exact_rank_max_n)
    basis_dtype = np.dtype(basis_dtype)

    if block_cols is None:
        block_cols = min(p, 8)
    block_cols = int(max(1, min(block_cols, p)))

    target_rank = int(min(p, d + max(0, int(oversample))))

    rng = np.random.default_rng(seed)
    col_means = np.asarray(X.mean(axis=0), dtype=np.float64)
    scale = np.sqrt(12.0 / (n * (n ** 2 - 1)))
    center = (n + 1.0) / 2.0

    try:
        sysOps.throw_status(
            "rank_rotation_embedding: "
            f"rank_mode={chosen_rank_mode}, adaptive={adaptive}, "
            f"initial_max_rotations={initial_max_rotations}, "
            f"max_rotations={max_rotations}, min_rotations={min_rotations}, "
            f"check_every={check_every}, patience={patience}, "
            f"block_cols={block_cols}, basis_dtype={basis_dtype.name}, "
            f"target_rank={target_rank}."
        )
    except Exception:
        pass

    U = None
    s = np.empty((0,), dtype=np.float64)
    rotations_done = 0

    if adaptive:
        sample_size = int(min(n, max(4096, 2048 * max(1, d))))
        sample_rows = None if sample_size >= n else np.sort(rng.choice(n, size=sample_size, replace=False))
        prev_sample_basis = None
        stable_checks = 0
    else:
        sample_rows = None
        prev_sample_basis = None
        stable_checks = 0

    converged = False
    while True:
        while rotations_done < max_rotations:
            R = _haar_random_rotation(p, rng)
            for j0 in range(0, p, block_cols):
                j1 = min(p, j0 + block_cols)
                R_block = np.asarray(R[:, j0:j1], dtype=np.float64, order='F')
                proj = np.asarray(X @ R_block, dtype=np.float64)
                proj -= col_means @ R_block
                scores = _rank_score_columns(
                    proj,
                    scale,
                    center,
                    rank_mode=chosen_rank_mode,
                    rng=rng,
                    n_bins=approx_rank_bins,
                    sample_size=approx_rank_sample_size,
                    out_dtype=basis_dtype,
                )
                U, s = _incremental_svd_update(
                    U,
                    s,
                    scores,
                    rank_keep=target_rank,
                    basis_dtype=basis_dtype,
                )

            rotations_done += 1

            if reorthogonalize_every and (rotations_done % int(reorthogonalize_every) == 0):
                U = _reorth_basis(U, basis_dtype=basis_dtype)

            if adaptive and rotations_done >= min_rotations and (
                (rotations_done % check_every == 0) or (rotations_done == max_rotations)
            ):
                change, prev_sample_basis = _sampled_subspace_change(
                    U,
                    sample_rows,
                    d,
                    prev_sample_basis=prev_sample_basis,
                )
                if np.isfinite(change) and change <= float(subspace_tol):
                    stable_checks += 1
                else:
                    stable_checks = 0

                try:
                    sysOps.throw_status(
                        "rank_rotation_embedding: "
                        f"rotation={rotations_done}/{max_rotations}, sampled_subspace_change={change:.3e}, "
                        f"stable_checks={stable_checks}/{patience}."
                    )
                except Exception:
                    pass

                if stable_checks >= patience:
                    converged = True
                    try:
                        sysOps.throw_status(
                            "rank_rotation_embedding: "
                            f"early stop after {rotations_done} rotations "
                            f"(sampled_subspace_change={change:.3e})."
                        )
                    except Exception:
                        pass
                    break

        if converged or (not adaptive):
            break

        if not require_convergence:
            break

        if explicit_rotation_budget:
            break

        next_max_rotations = max(
            max_rotations + check_every,
            int(np.ceil(1.5 * max_rotations)),
        )

        if max_auto_rotations is not None:
            if max_rotations >= max_auto_rotations:
                break
            next_max_rotations = min(next_max_rotations, max_auto_rotations)

        if next_max_rotations <= max_rotations:
            break

        try:
            sysOps.throw_status(
                "rank_rotation_embedding: "
                f"extending rotation budget from {max_rotations} to {next_max_rotations} "
                "because the sampled-subspace stop criterion is not yet satisfied."
            )
        except Exception:
            pass
        max_rotations = int(next_max_rotations)
 

    if adaptive and require_convergence and not converged:
        raise RuntimeError(
            "rank_rotation_embedding stopped before satisfying the "
            "sampled-subspace stop criterion; "
            f"rotations_done={rotations_done}, max_rotations={max_rotations}, "
            f"min_rotations={min_rotations}, check_every={check_every}, "
            f"patience={patience}, subspace_tol={subspace_tol}."
        )
 
    if U is None or s.size == 0:
        raise RuntimeError("rank_rotation_embedding failed to accumulate any rank-score blocks")

    U = np.asarray(U[:, :d], dtype=basis_dtype, order='F')
    U = _reorth_basis(U, basis_dtype=basis_dtype)
    s = np.asarray(s[:d], dtype=np.float64)
    return U[:, :X.shape[1]] * s[np.newaxis, :X.shape[1]]


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
    def __init__(self, csr_op1, csr_op2):
        self.csr_op1 = csr_op1
        self.csr_op2 = csr_op2
        self.csr_op2T = csr_op2.T
        self.shape = getattr(csr_op1, "shape", None)
        self.dtype = getattr(csr_op1, "dtype", np.float64)

    def dot(self, x):
        return self.csr_op2.dot(self.csr_op1.dot(self.csr_op2T.dot(x)))
        

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
    # Primary second-pass GSE solve and final coordinate export.
    # Required route parameters include -path, -inference_dim,
    # -inference_eignum, and -final_eignum; -calc_final controls final AnnData
    # augmentation after coordinates and final cluster labels are available.

    # The objective setup below is shared by ordinary non-coarsen and
    # coarsen-and-align runs.  Coarsen-and-align no longer installs a
    # different transformed_matrix.npz before this function runs.

    if type(params['-inference_eignum']) == list:
        fill_params(params)
    _validate_coarsen_alignment_mode(params)
    params.setdefault('-scales', 1)
    inference_eignum = int(params['-inference_eignum'])
    inference_dim = int(params['-inference_dim'])
    GSE_final_eigenbasis_size = int(params['-final_eignum'])
    sysOps.num_workers = NTHREADS
    sysOps.globaldatapath = str(params['-path'])
    _remove_removed_align_kernel_artifacts(sysOps.globaldatapath, output_name=output_name)
    # Default: auto. If '-h5ad_include_sequences' is present, always include.
    # Otherwise, include sequences only when label_pt appears STAR-less and contains sequences.
    sysOps.h5ad_include_nonunique_genes = ('-h5ad_include_nonunique_genes' in params)
    sysOps.h5ad_include_sequences = True if ('-h5ad_include_sequences' in params) else None

    try:
        os.mkdir(sysOps.globaldatapath + "tmp")
    except:
        pass

    this_GSEobj = GSEobj(inference_dim, inference_eignum)

    _drop_legacy_coarsen_align_second_pass_artifacts(
        sysOps.globaldatapath,
        output_name=output_name,
    )

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
        del nn_indices, indices, indptr, data, nn_indices_csr, output0, scaled_evecs

    if not sysOps.check_file_exists('evecs.npy'):
        this_GSEobj.inference_eignum = int(GSE_final_eigenbasis_size)
        sysOps.throw_status("Generating final eigenbasis ...")
        this_GSEobj.eigen_decomp(pmax=1,rank_transform=True)
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
    - List[int]: The partition vector returned by pymetis.part_graph().
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
    if sysOps.check_file_exists("preorthbasis.npy"):
        sysOps.throw_status(sysOps.globaldatapath + "preorthbasis.npy found pre-computed.")
        return
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
    

def parallel_krylov_fill(out_block, csr_op1, diag, csr_op2, init_vector):
    # out_block shape: (N, krylov_num)
    out_block[:, 0] = init_vector
    out_block[:, 0] -= np.mean(out_block[:, 0])

    for i in range(1, out_block.shape[1]):
        prev = out_block[:, i - 1]
        if csr_op2 is None:
            out_block[:, i] = diag.dot(csr_op1.dot(prev))
        else:
            out_block[:, i] = diag.dot(
                csr_op2.dot(csr_op1.dot(csr_op2.T.dot(prev)))
            )


def get_eigs(csr_op1, k, csr_op2=None, krylov_iterations=5):
    krylov_approx = sysOps.globaldatapath + "preorthbasis.npy"
    if csr_op2 is None:
        diag = scipy.sparse.diags(np.power(np.array(csr_op1.sum(axis=1)).flatten() + 1E-10, -1))
    else:
        deg = csr_op2.dot(
            csr_op1.dot(
                csr_op2.T.dot(np.ones(csr_op1.shape[0], dtype=np.float64))
            )
        )
        diag = scipy.sparse.diags(np.power(np.array(deg).flatten() + 1E-10, -1))
        del deg

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
            innerprod = krylov_space.T.dot(diag.dot(csr_op2.dot(csr_op1.dot(csr_op2.T.dot(krylov_space)))))

        evals,evecs = LA.eig(innerprod)

        eval_order = np.argsort(-np.real(evals))[:(2*k)]
        evecs = np.real(evecs[:,eval_order])
        evals = np.real(evals[eval_order])
        seq_evecs = krylov_space.dot(evecs)
        del krylov_space, innerprod, evecs, evals

        seq_evecs -= np.mean(seq_evecs,axis=0)
        seq_evecs = seq_evecs / LA.norm(seq_evecs, axis=0, keepdims=True)

        if csr_op2 is None:
            Mv = diag.dot(csr_op1).dot(seq_evecs)
        else:
            Mv = diag.dot(csr_op2.dot(csr_op1.dot(csr_op2.T.dot(seq_evecs))))
        
        seq_evals = np.sum(seq_evecs * Mv, axis=0)
        del Mv

        order = np.argsort(-seq_evals)

        seq_evecs = seq_evecs[:,order]
        seq_evals = seq_evals[order]
        triv_eig_indices = get_triv_status(seq_evecs)
        triv_eig_indices += (seq_evals >= 1)
        seq_evecs = seq_evecs[:,~triv_eig_indices][:,:k]
        seq_evals = seq_evals[~triv_eig_indices][:k]

        krylov_iter += 1
        sysOps.throw_status('Trivial indices ' + str(np.where(triv_eig_indices)[0]) + ' removed.')

    np.save(sysOps.globaldatapath + 'init_evecs.npy',seq_evecs)
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
        self.alphas_arr = None
        self.Ls_arr = None

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
    
    
    def eigen_decomp(self,orth=False, pmax=None, rank_transform=False):
    # Assemble linear manifold from data using "local linearity" assumption
    # assumes link_data type-1- and type-2-indices at this point has non-overlapping indices
        if self.seq_evecs is not None:
            del self.seq_evecs
            self.seq_evecs = None
        if pmax == 1:
            csr_op2 = load_npz(sysOps.globaldatapath + 'transformed_matrix.npz')
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
        if not sysOps.check_file_exists('init_evecs.npy'):
            get_eigs(csr_op1=csr_op1,k=self.inference_eignum,csr_op2=csr_op2)
        del csr_op1, csr_op2
        
        self.seq_evecs = np.load(sysOps.globaldatapath + 'init_evecs.npy')
        if rank_transform: # will automatically orthogonalize
            if not sysOps.check_file_exists('rank_evecs.npy'):
                self.seq_evecs -= np.mean(self.seq_evecs,axis=0)
                self.seq_evecs = rank_rotation_embedding(self.seq_evecs.dot(np.diag(1.0/np.maximum(1E-20,np.sqrt(1-np.load(sysOps.globaldatapath + "evals.npy"))))))
                self.seq_evecs -= np.mean(self.seq_evecs,axis=0)
                self.seq_evecs /= LA.norm(self.seq_evecs,axis=0)
                np.save(sysOps.globaldatapath + 'rank_evecs.npy',self.seq_evecs)
            del self.seq_evecs
            interleave_concat(sysOps.globaldatapath + 'rank_evecs.npy',sysOps.globaldatapath + 'init_evecs.npy',sysOps.globaldatapath + 'new_evecs.npy',self.spat_dims)
            os.remove(sysOps.globaldatapath + 'init_evecs.npy')
            os.remove(sysOps.globaldatapath + 'rank_evecs.npy')
            self.seq_evecs = orth_preserve_order(np.load(sysOps.globaldatapath + 'new_evecs.npy'))
            os.remove(sysOps.globaldatapath + 'new_evecs.npy')
            np.save(sysOps.globaldatapath + 'evecs.npy',self.seq_evecs)
        elif orth:
            self.seq_evecs = orth_preserve_order(self.seq_evecs)
            np.save(sysOps.globaldatapath + 'evecs.npy',self.seq_evecs)
            os.remove(sysOps.globaldatapath + 'init_evecs.npy')
        else:
            os.rename(sysOps.globaldatapath + 'init_evecs.npy',sysOps.globaldatapath + 'evecs.npy')
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
            
                csr = dot2(csr_op1, csr_op2)
            
            vals = csr.dot(np.ones(self.Npts, dtype=np.float64))
            
            self.ampfactors = np.log(np.maximum(1E-10, csr.dot(1.0/np.maximum(1E-10, vals))))
            self.reweighted_Nlink = 0.5 * np.sum(vals)

            sysOps.throw_status('Calculating self.gl_innerprod')
            self.gl_innerprod = self.seq_evecs.dot(csr.dot(self.seq_evecs.T))
            sysOps.throw_status('Calculating self.gl_diag')
            self.gl_diag      = self.seq_evecs.dot(scipy.sparse.diags(vals).dot(self.seq_evecs.T))
            del vals, csr

            self.sub_pairing_count = 2 * (self.spat_dims + 1)
            self.hashings          = max(1, int(NTHREADS))
            Pmax_est = 3 * self.sub_pairing_count * self.Npts
            if Pmax_est == 0 and self.Npts > 0 : Pmax_est = self.Npts * 10
            if self.Npts == 0: Pmax_est = 1 # Avoid zero-size array for w_buff if Npts is 0

            self.w_buff      = np.zeros((Pmax_est, self.spat_dims), np.float64)
            self.dXpts_buff  = np.zeros((self.Npts, self.spat_dims, self.hashings), np.float64)
            self.hessp_buff  = np.zeros_like(self.dXpts_buff)
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
                self.w_buff = np.zeros((P_actual, self.spat_dims), np.float64)
            elif P_actual == 0: # Ensure w_buff is at least 1 for Numba if P_actual is 0
                 self.w_buff = np.zeros((1, self.spat_dims), np.float64) # Or handle P_actual=0 in Numba
            else:
                self.w_buff = self.w_buff[:P_actual, :]
            self._pair_sorted   = None
            self._wt_sorted     = None
            self._pair_offsets  = None
        elif self.Npts == 0 : # No points, no pairings
             self.subsample_pairings = np.empty((0,2), dtype=np.int32)
             self.subsample_pairing_weights = np.empty((0,), dtype=np.float64)
             self.w_buff = np.zeros((1, self.spat_dims), np.float64)
             self._pair_sorted   = None
             self._wt_sorted     = None
             self._pair_offsets  = None


        self.Xpts[:] = current_Xpts
        P_actual = self.subsample_pairings.shape[0]

        if P_actual == 0 and self.Npts > 0 : # if pairings became empty unexpectedly
             # Fallback for w_buff if P_actual is 0 but we might proceed
             self.w_buff = np.zeros((1, self.spat_dims), np.float64)

        # --- Bucketing: ensure we have plane-partitioned pairs/weights ---
        if P_actual > 0 and (self._pair_sorted is None or
                             self._pair_sorted.shape[0] != P_actual or
                             self._pair_offsets is None or
                             self._pair_offsets.size != (self.hashings + 1)):
            self._ensure_pair_buckets()

        if do_grad:
            if self.w_buff.shape[0] < P_actual and P_actual > 0:
                 self.w_buff = np.zeros((P_actual, self.spat_dims), np.float64)
            elif P_actual == 0: # Ensure w_buff is at least size 1 for Numba if P_actual is 0
                 if self.w_buff.shape[0] < 1: self.w_buff = np.zeros((1, self.spat_dims), np.float64)


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
                    inp_pts, self.sumw,
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
        Uses modulo hashing so every pairing is assigned to exactly one plane.
        """
        pair = np.ascontiguousarray(self.subsample_pairings, dtype=np.int32)
        wt = np.ascontiguousarray(self.subsample_pairing_weights, dtype=np.float64)
        H = int(self.hashings)

        planes = ((pair[:, 0] + pair[:, 1]) % H).astype(np.int64)

        counts = np.bincount(planes, minlength=H).astype(np.int64)
        offsets = np.empty(H + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        pair_sorted = np.empty_like(pair)
        wt_sorted = np.empty_like(wt)
        _bucket_pairs_stable(pair, wt, planes, offsets, pair_sorted, wt_sorted)
        self._pair_sorted = pair_sorted
        self._wt_sorted = wt_sorted
        self._pair_offsets = offsets

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
    "float64[:](float64[:], float64[:], float64[:], float64[:])",
    fastmath=True, parallel=True, cache=True)
def _kernel_vec_multi(r: np.ndarray,
                      shell: np.ndarray,
                      alphas: np.ndarray,
                      Ls:     np.ndarray) -> np.ndarray:
    """
    Radial kernel

        w(r) = r^(d-1) * sum_k alpha_k * exp(-r^2 / L_k^2)

    Parameters
    ----------
    r, shell : 1-D float64 arrays with the same length
    alphas   : 1-D float64 array of mixture weights
    Ls       : 1-D float64 array of positive length scales

    Returns
    -------
    out : 1-D float64 array with the same length as r
    """
    n = r.shape[0]
    K = alphas.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        r2 = r[i] * r[i]
        acc = 0.0
        for k in range(K):
            L = Ls[k]
            denom = L * L
            if denom <= 0.0:
                denom = 1.0e-300
            acc += alphas[k] * math.exp(-r2 / denom)
        out[i] = shell[i] * acc
    return out

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
    shell_obs = np.ones_like(r_obs) if d == 1 else r_obs ** (d - 1)
    N_tot = N_ij.sum()

    # ---------- 2. Monte‑Carlo panel for Σ w ------------------------
    total_pairs = N * (N - 1) // 2
    flat_idx    = rng.choice(total_pairs, size=sample_pairs, replace=False)
    i_samp, j_samp = _idx_to_pair(flat_idx, N)

    r_samp     = np.linalg.norm(coords[i_samp] - coords[j_samp], axis=1).clip(min=dist_eps)
    shell_samp = np.ones_like(r_samp) if d == 1 else r_samp ** (d - 1)
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
    "float64[:,:], float64[:,:], float64, "
    "int64, int64, int64, int64, float64[:], float64[:], float64[:])",
    fastmath=True, parallel=True, cache=True)
def get_hessp_bucketed(out, pair_sorted, wt_sorted, offsets, wbuf,
                       grad_sum_dxpts, v_pts, sumw_total,
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
            gamma_sum = 0.0
            beta_sum  = 0.0
            for m_idx in range(alphas.size):
                w_component = alphas[m_idx] * math.exp(-diff_sq_val * invL2[m_idx])
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



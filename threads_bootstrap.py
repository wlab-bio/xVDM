from __future__ import annotations

import multiprocessing
import os
import re
import warnings

# ---------- decide how many threads this process should use ----------
# Slurm launches one Python parent process for optimOps.py.  That parent should
# keep the requested thread budget (for example 20 threads).  Registration
# ensemble child processes can set REGZF_THREADS_PER_PROCESS=1 before importing
# this module so the children become single-threaded without changing the Slurm
# preamble for the parent job.
_to_int = lambda v: int(v) if v and re.fullmatch(r"\d+", str(v)) else None


def _affinity_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return None


n_slurm = _to_int(os.getenv("SLURM_CPUS_PER_TASK"))
n_affinity = _affinity_count()
n_hw = max(1, multiprocessing.cpu_count())

# Prefer the scheduler allocation, then the process CPU affinity mask, then the
# host CPU count.  This prevents an inherited OMP_NUM_THREADS from exceeding the
# actual allocation.
n_alloc = n_slurm or n_affinity or n_hw or 1

# REGZF_THREADS_PER_PROCESS is a per-process override used by register_zf
# ensemble workers.  Otherwise existing exported OMP/NUMBA values remain the
# user-requested per-process budget, clamped to the allocation.
n_user = _to_int(os.getenv("REGZF_THREADS_PER_PROCESS"))
n_pre = (
    n_user
    or _to_int(os.getenv("OMP_NUM_THREADS"))
    or _to_int(os.getenv("NUMBA_NUM_THREADS"))
)

NTHREADS = max(1, min(n_pre or n_alloc, n_alloc))

if n_pre and n_pre > n_alloc:
    warnings.warn(
        f"Requested {n_pre} threads per process, but the visible allocation "
        f"appears to be {n_alloc}; clamping to {NTHREADS}.",
        RuntimeWarning,
    )

# ---------- freeze all thread-pool variables BEFORE numba import ------
for k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "TBB_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[k] = str(NTHREADS)

os.environ.update({
    "MKL_DYNAMIC": "FALSE",
    "OMP_DYNAMIC": "FALSE",
    "KMP_BLOCKTIME": "0",
})

# ---------- now it is SAFE to load Numba -----------------------------
try:
    import numba

    # NUMBA_NUM_THREADS is an upper bound fixed at import.  In normal parent
    # operation this is NTHREADS.  In ensemble child processes register_zf sets
    # REGZF_THREADS_PER_PROCESS/NUMBA_NUM_THREADS before spawn, so this is 1.
    max_numba_threads = int(getattr(numba.config, "NUMBA_NUM_THREADS", NTHREADS))
    numba.set_num_threads(max(1, min(NTHREADS, max_numba_threads)))
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Could not configure numba threads: {exc}", RuntimeWarning)

print(f"[threads_bootstrap] Using {NTHREADS} threads everywhere (allocation cap {n_alloc}).")

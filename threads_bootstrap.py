import os, multiprocessing, re

# ---------- decide how many threads we really have -----------------
_to_int = lambda v: int(v) if v and re.fullmatch(r"\d+", str(v)) else None

n_pre   = _to_int(os.getenv("OMP_NUM_THREADS")) \
       or _to_int(os.getenv("NUMBA_NUM_THREADS"))          # hard override
n_slurm = _to_int(os.getenv("SLURM_CPUS_PER_TASK"))        # slurm allocation
n_auto  = max(1, multiprocessing.cpu_count() // 2)         # fallback

NTHREADS = n_pre or n_slurm or n_auto

# ---------- freeze all thread‑pool variables BEFORE numba import ---
for k in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "TBB_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"
):
    os.environ[k] = str(NTHREADS)

os.environ.update({
    "MKL_DYNAMIC": "FALSE",
    "OMP_DYNAMIC": "FALSE",
    "KMP_BLOCKTIME": "0",
})

# ---------- now it is SAFE to load Numba ---------------------------
import numba
numba.set_num_threads(NTHREADS)      # lock the pool

print(f"[threads_bootstrap] Using {NTHREADS} threads everywhere.")

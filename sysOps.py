import csv
import sys
import time
import os
import os.path
import shutil
import subprocess
from joblib import Parallel, delayed


statuslogfilename = 'statuslog.csv'
globaldatapath = ''
pipeline_command = ''
num_workers = -1


def sh(cmd_str):
    # throw_status("RUNNING: " + cmd_str)
    return subprocess.run(
        cmd_str,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    ).stdout


def exitProgram():
    # add_nodes_running(-1,0,True)
    throw_status("PROGRAM ENDED.")
    sys.exit()

def prune_dir_except(keep: list[str], path: str | None = None) -> None:
    """
    Delete all files/directories in *path* (default: globaldatapath) except those
    whose basenames appear in *keep*.
    """
    if path is None:
        path = globaldatapath

    keep_set = set(keep)

    try:
        entries = os.listdir(path)
    except FileNotFoundError:
        return

    for name in entries:
        if name in keep_set:
            continue
        full_path = os.path.join(path, name)
        try:
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            # Best-effort: pruning is optional and should not crash analysis
            throw_status(f"Warning: failed to remove {full_path}: {e}")

 
# ---------------------------------------------------------------------
def check_file_exists(filename: str, path: str | None = None) -> bool:
    """
    Return True if *filename* exists.

    • If *filename* is already an absolute path, the *path* argument (or
      globaldatapath) is ignored.
    • If *filename* is a path that already includes *globaldatapath* as a prefix,
      it is treated as a fully-qualified path and checked as-is.
    • Otherwise the function behaves like the legacy implementation.
    """
    if os.path.isabs(filename) or (path is None and filename.startswith(globaldatapath)):
        return os.path.isfile(filename)

    if path is None:
        path = globaldatapath

    return os.path.isfile(os.path.join(path, filename))
# ---------------------------------------------------------------------


def _resolve_n_jobs(default: int = -1) -> int:
    """Resolve joblib worker count for optional parallel sorting."""
    global num_workers
    if isinstance(num_workers, int) and num_workers > 0:
        return num_workers

    env = os.getenv("DNAMIC_NUM_WORKERS") or os.getenv("NUM_WORKERS") or os.getenv("JOBLIB_NUM_WORKERS")
    if env is not None and str(env).isdigit():
        return int(env)

    slurm = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm is not None and str(slurm).isdigit():
        return int(slurm)

    return default


# ---------------------------------------------------------------------
def big_sort(
    param_str: str,
    infilename: str,
    outfilename: str,
    path: str | None = None,
    splitfile_size: int = 500_000,
    parallel: bool = False,
):
    """
    External GNU-sort wrapper.

    Surgical fixes:
      • Supports *absolute* input and output filenames cleanly.
      • Allows limiting joblib parallelism via sysOps.num_workers or env vars.

    Behaviour is unchanged for the common case of relative filenames in the
    current run directory.
    """

    # ---------- determine working directory ---------------------------
    # This is where splitfiles and temporary sort files will live.
    if path is None:
        if os.path.isabs(infilename):
            work_dir = os.path.dirname(infilename) + os.sep
        elif os.path.isabs(outfilename):
            work_dir = os.path.dirname(outfilename) + os.sep
        else:
            # Legacy convenience: callers sometimes pass globaldatapath+filename.
            if infilename.startswith(globaldatapath):
                infilename = infilename[len(globaldatapath):]
            if outfilename.startswith(globaldatapath):
                outfilename = outfilename[len(globaldatapath):]
            work_dir = globaldatapath
    else:
        work_dir = path

    if work_dir is None:
        work_dir = ""

    if work_dir != "" and not work_dir.endswith(os.sep):
        work_dir += os.sep

    # ---------- resolve full paths -----------------------------------
    infile_full = infilename if os.path.isabs(infilename) else os.path.join(work_dir, infilename)
    outfile_full = outfilename if os.path.isabs(outfilename) else os.path.join(work_dir, outfilename)

    # ---------- sanity checks ----------------------------------------
    if not check_file_exists(infile_full):
        throw_status(f"Sort failed. Could not find {infile_full}.")
        exitProgram()

    # ---------- empty file: nothing to sort --------------------------
    if os.path.getsize(infile_full) == 0:
        throw_status(f"Sort input {infile_full} is empty. Writing empty output.")
        open(outfile_full, "w").close()
        return

    throw_status(f"Beginning sort {infile_full}  -->  {outfile_full}.")

    # ---------- temporary workspace ----------------------------------
    tmp_root = os.path.join(work_dir, "tmp")
    os.makedirs(tmp_root, exist_ok=True)

    # ---------- split large file -------------------------------------
    split_prefix = os.path.join(work_dir, "splitfile-")
    sh(f"split -l {splitfile_size} {infile_full} {split_prefix}")

    _, filenames = get_directory_and_file_list(work_dir)
    filenames = [fn for fn in filenames if fn.startswith("splitfile-")]

    # ---------- per-chunk sort ---------------------------------------
    if parallel:
        n_jobs = _resolve_n_jobs(-1)
        tmpdirs = [f"{tmp_root}_{fn}//" for fn in filenames]
        sh("mkdir " + " ".join(tmpdirs))
        Parallel(n_jobs=n_jobs)(
            delayed(sh)(
                f"sort -T {tdir} {param_str} {os.path.join(work_dir, fn)} "
                f"> {os.path.join(work_dir, 'sorted_' + fn)} && rm {os.path.join(work_dir, fn)} && rm -r {tdir}"
            )
            for fn, tdir in zip(filenames, tmpdirs)
        )
    else:
        for fn in filenames:
            sh(
                f"sort -T {tmp_root} {param_str} {os.path.join(work_dir, fn)} "
                f"> {os.path.join(work_dir, 'sorted_' + fn)} && rm {os.path.join(work_dir, fn)}"
            )

    # ---------- final merge ------------------------------------------
    sh(
        f"sort -T {tmp_root} -m {param_str} {os.path.join(work_dir, 'sorted_splitfile-*')} > {outfile_full}"
    )
    sh(f"rm {os.path.join(work_dir, 'sorted_splitfile-*')}") 

    if not check_file_exists(outfile_full):
        throw_status("Sort failed. Exiting.")
        exitProgram()
# ---------------------------------------------------------------------


def throw_exception(this_input, path=None):
    # throws exception this_input[0] to file-name this_input[1], if this_input[1] exists, or errorlog.csv otherwise

    if path is None:
        path = globaldatapath

    if type(this_input) == list and len(this_input) == 2:
        statusphrase = this_input[0]
        statuslog_filename = this_input[1]
    else:
        if type(this_input) == list:
            statusphrase = this_input[0]
        else:
            statusphrase = this_input
        statuslog_filename = path + "errorlog.csv"

    my_datetime = time.strftime("%Y/%m/%d %H:%M:%S")
    with open(statuslog_filename, "a+") as csvfile:
        csvfile.write(my_datetime + "|" + statusphrase + "\n")


def throw_status(this_input, path=None):
    # throws status this_input[0] to file-name this_input[1], if this_input[1] exists, or statuslog.csv otherwise
    # if this_input[1] is global variable statuslogfilename, globaldatapath will already be incorporated to beginning of string, and therefore it is not included in call to file-open function

    if path is None:
        path = globaldatapath

    if type(this_input) == list and len(this_input) == 2:
        statusphrase = this_input[0]
        statuslog_filename = this_input[1]
    else:
        if type(this_input) == list:
            statusphrase = this_input[0]
        else:
            statusphrase = this_input
        statuslog_filename = path + "statuslog.csv"

    my_datetime = time.strftime("%Y/%m/%d %H:%M:%S")
    with open(statuslog_filename, "a+") as csvfile:
        csvfile.write(my_datetime + "|" + statusphrase + "\n")

    print(my_datetime + "|" + statusphrase)


def get_directory_and_file_list(path=None):
    if path is None:
        path = globaldatapath + "."
    elif path == "":
        path = "."
    else:
        path = path + "."

    while True:
        try:
            for dirname, dirnames, filenames in os.walk(path):
                return [dirnames, filenames]  # first level of directory hierarchy only
            return [list(), list()]
        except:
            print("Error during file/directory-readout. Re-trying.")


def initiate_statusfilename(prefix='', make_file=False):
    # globaldatarunpath added directly to statuslogfilename
    global statuslogfilename
    fullprefix = prefix + 'statuslog'
    max_statuslog_index = 0
    [dirnames, filenames] = get_directory_and_file_list()
    for filename in filenames:
        if filename.startswith(fullprefix) and filename.endswith('.csv'):
            try:
                max_statuslog_index = max(max_statuslog_index, int(filename[len(fullprefix):(len(filename) - 4)]))
            except:  # no integer-form index substring
                pass

    statuslogfilename = globaldatapath + fullprefix + str(max_statuslog_index + 1) + ".csv"

    if make_file:
        status_outfile = open(statuslogfilename, 'w')
        status_outfile.close()
    return


def initiate_runpath(mydatapath, autoinitialize_statusfilename=True):
    global globaldatapath
    globaldatapath = mydatapath
    print('init=' + mydatapath)

    if autoinitialize_statusfilename:
        initiate_statusfilename()

    try:
        os.mkdir(globaldatapath + "tmp")
    except:
        pass
    return


def delay_with_alertfile(alertfile):
    # delays until alertfile is removed from directory, at which point the alertfile is replaced and process continues
    while True:
        try:
            alertfile_handle = open(globaldatapath + alertfile, 'rU')
            alertfile_handle.close()
            time.sleep(1)
        except:
            with open(globaldatapath + alertfile, 'w') as alertfile_handle:
                alertfile_handle.write('1')
            break
    return


def remove_alertfile(alertfile):
    os.remove(globaldatapath + alertfile)
    return
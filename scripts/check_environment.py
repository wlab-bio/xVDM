#!/usr/bin/env python3
"""Check Python modules and external executables used by the xVDM baseline."""

from __future__ import annotations

import importlib.util
import shutil
import sys


PYTHON_MODULES = {
    "annoy": "Annoy nearest-neighbor index",
    "anndata": "AnnData writers/readers",
    "faiss": "GSE nearest-neighbor search",
    "joblib": "parallel workers",
    "numba": "JIT kernels",
    "numpy": "array operations",
    "pandas": "tables and metadata",
    "pymetis": "graph partitioning",
    "scanpy": "registration AnnData compatibility",
    "scipy": "sparse matrices and numerical routines",
    "sklearn": "nearest neighbors / optional HDBSCAN",
    "threadpoolctl": "native thread limiting",
}

OPTIONAL_PYTHON_MODULES = {
    "hdbscan": "legacy/optional HDBSCAN clustering route",
    "igraph": "optional Leiden clustering route",
    "leidenalg": "optional Leiden clustering route",
    "ortools": "zebrafish registration min-cost flow",
}

EXECUTABLES = {
    "awk": "AWK helper scripts",
    "sort": "large file sorting",
    "gzip": "compressed intermediate files",
}

OPTIONAL_EXECUTABLES = {
    "bioawk": "FASTQ processing on some library routes",
    "STAR": "genome alignment when requested by lib.settings",
    "infomap": "Infomap clustering executable",
}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def print_check(label: str, name: str, detail: str, ok: bool) -> None:
    status = "ok" if ok else "missing"
    print(f"{label:9} {status:7} {name:14} {detail}")


def main() -> int:
    missing_required = []

    print("Python modules:")
    for name, detail in PYTHON_MODULES.items():
        ok = module_available(name)
        print_check("required", name, detail, ok)
        if not ok:
            missing_required.append(f"python:{name}")

    for name, detail in OPTIONAL_PYTHON_MODULES.items():
        print_check("optional", name, detail, module_available(name))

    print("\nExecutables:")
    for name, detail in EXECUTABLES.items():
        ok = shutil.which(name) is not None
        print_check("required", name, detail, ok)
        if not ok:
            missing_required.append(f"exec:{name}")

    infomap_ok = False
    for name, detail in OPTIONAL_EXECUTABLES.items():
        ok = shutil.which(name) is not None
        if name == "infomap":
            ok = ok or shutil.which("Infomap") is not None
            infomap_ok = ok
        print_check("optional", name, detail, ok)

    if not infomap_ok:
        print("note      missing infomap      GSE routes that call Infomap will fail until installed")

    if missing_required:
        print("\nMissing required dependencies:")
        for item in missing_required:
            print(f"- {item}")
        return 1

    print("\nBaseline environment checks passed for required dependencies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

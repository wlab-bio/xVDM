#!/usr/bin/env python3
"""
Thin benchmark adapter for register_zf.py.

The holdout benchmark imports a registration module rather than launching the
backend script.  This adapter only resolves the backend, re-exports its public
surface, and injects a deterministic slice-capacity default for the matcher.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any


_BACKEND_ENV_NAMES = (
    "REGZF_BACKEND_SCRIPT",
    "REGISTER_ZF_BACKEND",
    "REGISTER_ZF_SCRIPT",
    "REGZF_SCRIPT",
)


def _resolve_backend_script() -> Path:
    explicit = next((os.getenv(k) for k in _BACKEND_ENV_NAMES if os.getenv(k)), None)
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"register_zf backend not found: {path}")
        return path

    here = Path(__file__).resolve().parent
    candidates = [here / "register_zf.py"] + [
        p for p in sorted(here.glob("register_zf*.py"))
        if "adapter" not in p.name.lower() and "benchmark" not in p.name.lower()
    ]
    for path in candidates:
        if path.exists() and path.is_file() and path.resolve() != Path(__file__).resolve():
            return path.resolve()
    raise FileNotFoundError("Could not find register_zf.py; set REGZF_BACKEND_SCRIPT=/path/to/register_zf.py")


def _load_backend(path: Path):
    path = path.resolve()
    if path.name == "register_zf.py":
        sys.path.insert(0, str(path.parent))
        mod = importlib.import_module("register_zf")
        if Path(getattr(mod, "__file__", "")).resolve() == path:
            return mod

    module_name = f"_register_zf_backend_{abs(hash(str(path))) & 0xffffffff:x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import register_zf backend from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_BACKEND_SCRIPT = _resolve_backend_script()
_BACKEND = _load_backend(_BACKEND_SCRIPT)

for _name in dir(_BACKEND):
    if not _name.startswith("__") and _name not in globals():
        globals()[_name] = getattr(_BACKEND, _name)


def __getattr__(name: str) -> Any:
    return getattr(_BACKEND, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_BACKEND)))


def _accepts_kwargs(func: Any) -> bool:
    try:
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in inspect.signature(func).parameters.values())
    except (TypeError, ValueError):
        return True


def _call_supported(func: Any, /, **kwargs: Any) -> Any:
    if _accepts_kwargs(func):
        return func(**kwargs)
    sig = inspect.signature(func)
    return func(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def run_sparse_graph_matching_on_ratio_vectors(
    agg_h5ad_path: str,
    XA_features_01,
    YB_features_01,
    YB_coords,
    source_node_mass,
    output_dir: str,
    k0: int = 16,
    k_max: int = 256,
    lam_dir: float | None = None,
    refine_iter: int = 1,
    tree_workers: int | None = None,
    slice_capacity_mode: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Delegate matching to register_zf.py with benchmark-safe capacity defaults."""
    mode = str(
        slice_capacity_mode
        or os.getenv("REGZF_ADAPTER_SLICE_CAPACITY_MODE")
        or os.getenv("REGZF_SLICE_CAPACITY_MODE")
        or "mass_exact"
    ).strip().lower()
    result = _call_supported(
        _BACKEND.run_sparse_graph_matching_on_ratio_vectors,
        agg_h5ad_path=agg_h5ad_path,
        XA_features_01=XA_features_01,
        YB_features_01=YB_features_01,
        YB_coords=YB_coords,
        source_node_mass=source_node_mass,
        output_dir=output_dir,
        k0=k0,
        k_max=k_max,
        lam_dir=lam_dir,
        refine_iter=refine_iter,
        tree_workers=tree_workers,
        slice_capacity_mode=mode,
        **kwargs,
    )
    out = dict(result)
    out.setdefault("adapter_backend_script", str(_BACKEND_SCRIPT))
    out.setdefault("adapter_slice_capacity_mode", mode)
    return out


globals()["run_sparse_graph_matching_on_ratio_vectors"] = run_sparse_graph_matching_on_ratio_vectors

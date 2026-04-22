from __future__ import annotations

import ctypes
import os
import sys


_APPLIED_ENV = "TRAFFIC_ANOMALY_VLM_CUDA_COMPAT_APPLIED"


def ensure_cuda_runtime_compat() -> None:
    """Best-effort preload for conda CUDA libs to avoid symbol resolution issues."""
    if os.environ.get(_APPLIED_ENV) == "1":
        return

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        os.environ[_APPLIED_ENV] = "1"
        return

    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvjitlink = os.path.join(
        conda_prefix,
        "lib",
        py_ver,
        "site-packages",
        "nvidia",
        "nvjitlink",
        "lib",
    )
    cusparse = os.path.join(
        conda_prefix,
        "lib",
        py_ver,
        "site-packages",
        "nvidia",
        "cusparse",
        "lib",
    )

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    for p in (nvjitlink, cusparse):
        if os.path.isdir(p) and p not in ld_path:
            ld_path = f"{p}:{ld_path}" if ld_path else p
    os.environ["LD_LIBRARY_PATH"] = ld_path

    nvjitlink_so = os.path.join(nvjitlink, "libnvJitLink.so.12")
    cusparse_so = os.path.join(cusparse, "libcusparse.so.12")
    try:
        if os.path.exists(nvjitlink_so):
            ctypes.CDLL(nvjitlink_so, mode=ctypes.RTLD_GLOBAL)
        if os.path.exists(cusparse_so):
            ctypes.CDLL(cusparse_so, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        # Keep import behavior unchanged; downstream imports will still raise if needed.
        pass

    os.environ[_APPLIED_ENV] = "1"

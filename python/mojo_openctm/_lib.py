"""ctypes bindings for the Mojo OpenCTM kernels."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = Path(os.environ.get("MOJO_OPENCTM_LIB", ROOT / "dist" / "libmojo-openctm.so"))
I = ctypes.c_int64
F = ctypes.c_float

_SIGNATURES = {
    "ctm_rearrange_triangles": ([I, I], None),
    "ctm_make_index_deltas": ([I, I], None),
    "ctm_restore_indices": ([I, I], None),
    "ctm_interleave_ints": ([I, I, I, I, I], None),
    "ctm_deinterleave_ints": ([I, I, I, I, I], None),
    "ctm_interleave_floats": ([I, I, I, I], None),
    "ctm_deinterleave_floats": ([I, I, I, I], None),
    "ctm_setup_grid": ([I, I, I, I, I, I], None),
    "ctm_point_grid_indices": ([I, I, I, I, I, I], None),
    "ctm_make_vertex_deltas": ([I, I, I, I, I, I, I, F, I], None),
    "ctm_restore_vertices": ([I, I, I, I, I, I, F, I], None),
    "ctm_smooth_normals": ([I, I, I, I, I], None),
    "ctm_make_normal_deltas": ([I, I, I, I, F, I], None),
    "ctm_restore_normals": ([I, I, I, F, I], None),
    "ctm_make_map_deltas": ([I, I, I, I, F, I], None),
    "ctm_restore_map": ([I, I, I, F, I], None),
    "ctm_delta_encode_u32": ([I, I], None),
    "ctm_delta_decode_u32": ([I, I], None),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not LIBRARY.exists():
            raise RuntimeError("Mojo library is not built; run `pixi run build`")
        _library = ctypes.CDLL(str(LIBRARY))
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    """Return an address only for buffers matching the Mojo ABI contract."""
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.size == 0:
        raise ValueError("empty arrays must not cross the Mojo FFI boundary")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.uint32), np.dtype(np.uint8)):
        raise TypeError("FFI buffers must use float32, int32, uint32, or uint8")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffers must have a non-null address")
    return address

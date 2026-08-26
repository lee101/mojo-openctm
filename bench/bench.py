"""Benchmarks OpenCTM Mojo kernels against ports written from the same C source."""

from __future__ import annotations

import os
import platform
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "python"))

from mojo_openctm._lib import addr, lib  # noqa: E402


def timed(function, repetitions=5):
    function()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def numpy_index_deltas(source, destination):
    destination[:] = source
    same = destination[1:, 0] == destination[:-1, 0]
    with np.errstate(over="ignore"):
        destination[1:, 1] -= np.where(same, destination[:-1, 1], destination[1:, 0])
        destination[0, 1] -= destination[0, 0]
        destination[:, 2] -= destination[:, 0]
        destination[1:, 0] -= destination[:-1, 0]


def numpy_interleave(values, destination):
    raw = values.view(np.uint8).reshape(len(values), 3, 4)
    destination[:] = raw[:, :, ::-1].transpose(2, 1, 0).reshape(-1)


def numpy_smooth_normals(vertices, triangles, destination):
    destination.fill(0)
    first = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    second = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    normals = np.cross(first, second)
    lengths = np.linalg.norm(normals, axis=1)
    inverse = np.ones_like(lengths)
    np.divide(1, lengths, out=inverse, where=lengths > 1e-10)
    normals *= inverse[:, None]
    for corner in range(3):
        np.add.at(destination, triangles[:, corner], normals)
    lengths = np.linalg.norm(destination, axis=1)
    inverse = np.ones_like(lengths)
    np.divide(1, lengths, out=inverse, where=lengths > 1e-10)
    destination *= inverse[:, None]


def machine_name():
    cpu = platform.processor()
    if not cpu or cpu in {"x86_64", "AMD64", "aarch64"}:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as source:
                cpu = next(
                    line.split(":", 1)[1].strip()
                    for line in source
                    if line.startswith("model name")
                )
        except (OSError, StopIteration):
            cpu = "unknown CPU"
    return f"{cpu}; {platform.system()} {platform.machine()}"


def main():
    rng = np.random.default_rng(123)
    rows = []

    triangle_count = 1_000_000
    base = np.sort(rng.integers(0, 2_000_000, size=(triangle_count, 3), dtype=np.uint32), axis=1)
    base = base[np.lexsort((base[:, 1], base[:, 0]))]
    mojo_indices = np.empty_like(base)
    numpy_indices = np.empty_like(base)

    def mojo_delta():
        lib().ctm_make_index_deltas_to(addr(base), addr(mojo_indices), triangle_count)

    mojo_time = timed(mojo_delta)
    numpy_time = timed(lambda: numpy_index_deltas(base, numpy_indices))
    assert np.array_equal(mojo_indices, numpy_indices)
    rows.append(("MG1/MG2 index prediction, 1M tris", mojo_time, numpy_time))

    integer_count = 2_000_000
    integers = rng.integers(-(2**30), 2**30, size=(integer_count, 3), dtype=np.int32)
    mojo_bytes = np.empty(integers.size * 4, dtype=np.uint8)
    numpy_bytes = np.empty_like(mojo_bytes)
    mojo_time = timed(
        lambda: lib().ctm_interleave_ints(
            addr(integers), addr(mojo_bytes), integer_count, 3, 0
        )
    )
    numpy_time = timed(lambda: numpy_interleave(integers, numpy_bytes))
    assert np.array_equal(mojo_bytes, numpy_bytes)
    rows.append(("LZMA integer interleave, 6M ints", mojo_time, numpy_time))

    nx, ny = 700, 572
    x, y = np.meshgrid(
        np.linspace(0, 10, nx, dtype=np.float32),
        np.linspace(0, 8, ny, dtype=np.float32),
    )
    vertices = np.column_stack((x.ravel(), y.ravel(), (0.1 * np.sin(x) * np.cos(y)).ravel()))
    vertex_count = len(vertices)
    cell = np.arange((ny - 1) * (nx - 1), dtype=np.uint32)
    lower = (cell // (nx - 1)) * nx + cell % (nx - 1)
    triangles = np.vstack(
        (
            np.column_stack((lower, lower + 1, lower + nx)),
            np.column_stack((lower + 1, lower + nx + 1, lower + nx)),
        )
    ).astype(np.uint32)
    mojo_normals = np.empty_like(vertices)
    numpy_normals = np.empty_like(vertices)
    mojo_time = timed(
        lambda: lib().ctm_smooth_normals(
            addr(vertices), vertex_count, addr(triangles), len(triangles), addr(mojo_normals)
        ),
        repetitions=3,
    )
    numpy_time = timed(
        lambda: numpy_smooth_normals(vertices, triangles, numpy_normals),
        repetitions=3,
    )
    assert np.allclose(mojo_normals, numpy_normals, atol=2e-5)
    rows.append((f"MG2 smooth-normal prediction, {len(triangles) // 1000}K tris", mojo_time, numpy_time))

    print(f"Machine: {machine_name()}")
    print()
    print("| Kernel | Mojo | NumPy reference | Speedup |")
    print("|---|---:|---:|---:|")
    for name, mojo_time, numpy_time in rows:
        print(
            f"| {name} | {mojo_time * 1000:.2f} ms | "
            f"{numpy_time * 1000:.2f} ms | {numpy_time / mojo_time:.2f}x |"
        )


if __name__ == "__main__":
    main()

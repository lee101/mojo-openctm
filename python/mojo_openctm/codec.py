"""OpenCTM v5 MG1/MG2 reader and writer backed by Mojo transforms."""

from __future__ import annotations

import io
import lzma
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._lib import addr, lib

F32 = NDArray[np.float32]
U32 = NDArray[np.uint32]
I32 = NDArray[np.int32]
MAX_DECODED_BYTES = 1 << 30


@dataclass(slots=True)
class UVMap:
    values: F32
    name: str = ""
    file_name: str = ""
    precision: float = 1.0 / 4096.0


@dataclass(slots=True)
class AttributeMap:
    values: F32
    name: str = ""
    precision: float = 1.0 / 256.0


@dataclass(slots=True)
class Mesh:
    vertices: F32
    triangles: U32
    normals: F32 | None = None
    uv_maps: list[UVMap] = field(default_factory=list)
    attributes: list[AttributeMap] = field(default_factory=list)
    comment: str = ""
    method: str = "MG2"


class CTMError(ValueError):
    pass


def _positive_f32(value: float, name: str) -> float:
    try:
        narrowed = float(np.float32(value))
    except (TypeError, ValueError, OverflowError) as error:
        raise CTMError(f"{name} must be a finite positive float32 value") from error
    if not np.isfinite(narrowed) or narrowed <= 0:
        raise CTMError(f"{name} must be a finite positive float32 value")
    return narrowed


def _f32(values: ArrayLike, shape: tuple[int, int], name: str) -> F32:
    source = np.asarray(values)
    if (
        source.size
        and source.dtype.kind == "f"
        and source.dtype.itemsize > np.dtype(np.float32).itemsize
        and np.isfinite(source).all()
    ):
        raise CTMError(f"{name} uses {source.dtype}; explicitly convert to float32")
    result = np.ascontiguousarray(values, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != shape[1]:
        raise CTMError(f"{name} must have shape (n, {shape[1]})")
    return result


def _mesh(
    vertices: ArrayLike,
    triangles: ArrayLike,
    normals: ArrayLike | None,
    uv_maps: Mapping[str, ArrayLike] | list[UVMap] | None,
    attributes: Mapping[str, ArrayLike] | list[AttributeMap] | None,
    comment: str,
    method: str,
) -> Mesh:
    verts = _f32(vertices, (-1, 3), "vertices")
    triangle_source = np.asarray(triangles)
    if triangle_source.dtype.kind not in "iu":
        raise CTMError("triangles must contain integer indices")
    if triangle_source.size and (
        np.any(triangle_source < 0) or np.any(triangle_source > np.iinfo(np.uint32).max)
    ):
        raise CTMError("triangle indices must fit in uint32")
    tris = np.ascontiguousarray(triangle_source, dtype=np.uint32)
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise CTMError("triangles must have shape (m, 3)")
    if len(verts) == 0 or len(tris) == 0:
        raise CTMError("OpenCTM requires at least one vertex and one triangle")
    if not np.isfinite(verts).all():
        raise CTMError("vertices must be finite")
    if np.any(tris >= len(verts)):
        raise CTMError("triangle index is outside the vertex array")
    norm = None if normals is None else _f32(normals, (-1, 3), "normals")
    if norm is not None and (len(norm) != len(verts) or not np.isfinite(norm).all()):
        raise CTMError("normals must be finite and have one row per vertex")

    maps: list[UVMap] = []
    if isinstance(uv_maps, Mapping):
        maps = [UVMap(_f32(value, (-1, 2), "UV map"), name) for name, value in uv_maps.items()]
    elif uv_maps:
        maps = [
            UVMap(_f32(value.values, (-1, 2), "UV map"), value.name, value.file_name, value.precision)
            for value in uv_maps
        ]
    attrs: list[AttributeMap] = []
    if isinstance(attributes, Mapping):
        attrs = [
            AttributeMap(_f32(value, (-1, 4), "attribute map"), name)
            for name, value in attributes.items()
        ]
    elif attributes:
        attrs = [
            AttributeMap(_f32(value.values, (-1, 4), "attribute map"), value.name, value.precision)
            for value in attributes
        ]
    for value in [*maps, *attrs]:
        if len(value.values) != len(verts) or not np.isfinite(value.values).all():
            raise CTMError("maps must be finite and have one row per vertex")
        value.precision = _positive_f32(value.precision, "map precision")
    method = method.upper()
    if method not in {"MG1", "MG2"}:
        raise CTMError("method must be 'MG1' or 'MG2'")
    return Mesh(verts, tris, norm, maps, attrs, comment, method)


class _Writer:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()

    def raw(self, value: bytes) -> None:
        self.buffer.write(value)

    def u32(self, value: int) -> None:
        self.raw(struct.pack("<I", value))

    def f32(self, value: float) -> None:
        self.raw(struct.pack("<f", value))

    def string(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.u32(len(encoded))
        self.raw(encoded)


class _Reader:
    def __init__(self, data: bytes | bytearray | memoryview) -> None:
        self.buffer = io.BytesIO(bytes(data))

    def raw(self, count: int) -> bytes:
        if count < 0 or count > self.remaining():
            raise CTMError("truncated CTM stream")
        value = self.buffer.read(count)
        if len(value) != count:
            raise CTMError("truncated CTM stream")
        return value

    def remaining(self) -> int:
        return len(self.buffer.getbuffer()) - self.buffer.tell()

    def expect(self, value: bytes) -> None:
        if self.raw(len(value)) != value:
            raise CTMError(f"expected CTM section {value!r}")

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.raw(4))[0]

    def string(self) -> str:
        try:
            return self.raw(self.u32()).decode("utf-8")
        except UnicodeDecodeError as error:
            raise CTMError("invalid UTF-8 string in CTM stream") from error


def _lzma_filter(level: int) -> dict[str, int]:
    if not 0 <= level <= 9:
        raise CTMError("compression_level must be between 0 and 9")
    dictionary = 1 << (level * 2 + 14) if level <= 5 else (1 << 25 if level == 6 else 1 << 26)
    return {
        "id": lzma.FILTER_LZMA1,
        "dict_size": dictionary,
        "lc": 3,
        "lp": 0,
        "pb": 2,
        "mode": lzma.MODE_FAST if level == 0 else lzma.MODE_NORMAL,
        "nice_len": 32 if level < 7 else 64,
        "mf": lzma.MF_HC4 if level == 0 else lzma.MF_BT4,
    }


def _write_packed(writer: _Writer, interleaved: NDArray[np.uint8], level: int) -> None:
    filter_spec = _lzma_filter(level)
    packed = lzma.compress(interleaved, format=lzma.FORMAT_RAW, filters=[filter_spec])
    properties = lzma._encode_filter_properties(filter_spec)
    writer.u32(len(packed))
    writer.raw(properties)
    writer.raw(packed)


def _read_packed(reader: _Reader, byte_count: int) -> NDArray[np.uint8]:
    if byte_count < 0 or byte_count > MAX_DECODED_BYTES:
        raise CTMError("packed section is too large")
    packed_size = reader.u32()
    properties = reader.raw(5)
    packed = reader.raw(packed_size)
    try:
        filter_spec = lzma._decode_filter_properties(lzma.FILTER_LZMA1, properties)
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=[filter_spec])
        # The LZMA SDK used by OpenCTM omits an end marker; the format supplies
        # the exact output length, just as LzmaUncompress receives it upstream.
        unpacked = decoder.decompress(packed, max_length=byte_count + 1)
    except lzma.LZMAError as error:
        raise CTMError("invalid LZMA stream") from error
    if len(unpacked) != byte_count:
        raise CTMError("LZMA stream has the wrong uncompressed size")
    return np.frombuffer(unpacked, dtype=np.uint8)


def _pack_ints(writer: _Writer, values: I32, count: int, size: int, signed: bool, level: int) -> None:
    values = np.ascontiguousarray(values, dtype=np.int32)
    interleaved = np.empty(count * size * 4, dtype=np.uint8)
    if interleaved.size:
        lib().ctm_interleave_ints(addr(values), addr(interleaved), count, size, signed)
    _write_packed(writer, interleaved, level)


def _unpack_ints(reader: _Reader, count: int, size: int, signed: bool) -> I32:
    interleaved = _read_packed(reader, count * size * 4)
    values = np.empty((count, size), dtype=np.int32)
    if values.size:
        lib().ctm_deinterleave_ints(addr(interleaved), addr(values), count, size, signed)
    return values


def _pack_floats(writer: _Writer, values: F32, count: int, size: int, level: int) -> None:
    values = np.ascontiguousarray(values, dtype=np.float32)
    interleaved = np.empty(count * size * 4, dtype=np.uint8)
    lib().ctm_interleave_floats(addr(values), addr(interleaved), count, size)
    _write_packed(writer, interleaved, level)


def _unpack_floats(reader: _Reader, count: int, size: int) -> F32:
    interleaved = _read_packed(reader, count * size * 4)
    values = np.empty((count, size), dtype=np.float32)
    lib().ctm_deinterleave_floats(addr(interleaved), addr(values), count, size)
    return values


def _sorted_triangles(triangles: U32) -> U32:
    result = np.ascontiguousarray(triangles, dtype=np.uint32).copy()
    lib().ctm_rearrange_triangles(addr(result), len(result))
    return np.ascontiguousarray(result[np.lexsort((result[:, 1], result[:, 0]))])


def _index_deltas(triangles: U32) -> I32:
    result = np.empty_like(triangles)
    lib().ctm_make_index_deltas_to(addr(triangles), addr(result), len(result))
    return result.view(np.int32)


def _write_header(writer: _Writer, mesh: Mesh) -> None:
    writer.raw(b"OCTM")
    writer.u32(5)
    writer.raw(mesh.method.encode("ascii") + b"\0")
    writer.u32(len(mesh.vertices))
    writer.u32(len(mesh.triangles))
    writer.u32(len(mesh.uv_maps))
    writer.u32(len(mesh.attributes))
    writer.u32(1 if mesh.normals is not None else 0)
    writer.string(mesh.comment)


def _encode_mg1(writer: _Writer, mesh: Mesh, level: int) -> None:
    triangles = _sorted_triangles(mesh.triangles)
    writer.raw(b"INDX")
    _pack_ints(writer, _index_deltas(triangles), len(triangles), 3, False, level)
    writer.raw(b"VERT")
    _pack_floats(writer, mesh.vertices.reshape(-1, 1), mesh.vertices.size, 1, level)
    if mesh.normals is not None:
        writer.raw(b"NORM")
        _pack_floats(writer, mesh.normals, len(mesh.vertices), 3, level)
    for uv_map in mesh.uv_maps:
        writer.raw(b"TEXC")
        writer.string(uv_map.name)
        writer.string(uv_map.file_name)
        _pack_floats(writer, uv_map.values, len(mesh.vertices), 2, level)
    for attribute in mesh.attributes:
        writer.raw(b"ATTR")
        writer.string(attribute.name)
        _pack_floats(writer, attribute.values, len(mesh.vertices), 4, level)


def _grid(vertices: F32) -> tuple[F32, F32, U32, F32, U32]:
    minimum = np.empty(3, dtype=np.float32)
    maximum = np.empty(3, dtype=np.float32)
    division = np.empty(3, dtype=np.uint32)
    size = np.empty(3, dtype=np.float32)
    indices = np.empty(len(vertices), dtype=np.uint32)
    lib().ctm_setup_grid(addr(vertices), len(vertices), addr(minimum), addr(maximum), addr(division), addr(size))
    lib().ctm_point_grid_indices(addr(vertices), len(vertices), addr(minimum), addr(size), addr(division), addr(indices))
    return minimum, maximum, division, size, indices


def _restore_vertices(integers: I32, grid_indices: U32, minimum: F32, size: F32, division: U32, precision: float) -> F32:
    vertices = np.empty((len(integers), 3), dtype=np.float32)
    lib().ctm_restore_vertices(
        addr(integers), addr(grid_indices), len(integers), addr(minimum), addr(size),
        addr(division), precision, addr(vertices),
    )
    return vertices


def _smooth(vertices: F32, triangles: U32) -> F32:
    result = np.empty_like(vertices)
    lib().ctm_smooth_normals(addr(vertices), len(vertices), addr(triangles), len(triangles), addr(result))
    return result


def _encode_mg2(
    writer: _Writer,
    mesh: Mesh,
    level: int,
    vertex_precision: float,
    normal_precision: float,
) -> None:
    vertex_precision = _positive_f32(vertex_precision, "vertex precision")
    normal_precision = _positive_f32(normal_precision, "normal precision")
    minimum, maximum, division, size, unsorted_grid = _grid(mesh.vertices)
    writer.raw(b"MG2H")
    writer.f32(vertex_precision)
    writer.f32(normal_precision)
    for value in minimum:
        writer.f32(float(value))
    for value in maximum:
        writer.f32(float(value))
    for value in division:
        writer.u32(int(value))

    order = np.ascontiguousarray(
        np.lexsort((mesh.vertices[:, 0], unsorted_grid)), dtype=np.uint32
    )
    grid_indices = np.ascontiguousarray(unsorted_grid[order])
    integer_vertices = np.empty((len(mesh.vertices), 3), dtype=np.int32)
    lib().ctm_make_vertex_deltas(
        addr(mesh.vertices), addr(order), addr(grid_indices), len(mesh.vertices),
        addr(minimum), addr(size), addr(division), vertex_precision, addr(integer_vertices),
    )
    writer.raw(b"VERT")
    _pack_ints(writer, integer_vertices, len(mesh.vertices), 3, False, level)

    grid_deltas = grid_indices.copy()
    lib().ctm_delta_encode_u32(addr(grid_deltas), len(grid_deltas))
    writer.raw(b"GIDX")
    _pack_ints(writer, grid_deltas.view(np.int32), len(mesh.vertices), 1, False, level)

    restored = _restore_vertices(integer_vertices, grid_indices, minimum, size, division, vertex_precision)
    lookup = np.empty(len(mesh.vertices), dtype=np.uint32)
    lookup[order] = np.arange(len(mesh.vertices), dtype=np.uint32)
    triangles = _sorted_triangles(lookup[mesh.triangles])
    writer.raw(b"INDX")
    _pack_ints(writer, _index_deltas(triangles), len(triangles), 3, False, level)

    if mesh.normals is not None:
        smooth = _smooth(restored, triangles)
        integer_normals = np.empty((len(mesh.vertices), 3), dtype=np.int32)
        lib().ctm_make_normal_deltas(
            addr(mesh.normals), addr(order), addr(smooth), len(mesh.vertices),
            normal_precision, addr(integer_normals),
        )
        writer.raw(b"NORM")
        _pack_ints(writer, integer_normals, len(mesh.vertices), 3, False, level)

    for uv_map in mesh.uv_maps:
        integers = np.empty((len(mesh.vertices), 2), dtype=np.int32)
        lib().ctm_make_map_deltas(
            addr(uv_map.values), addr(order), len(mesh.vertices), 2,
            uv_map.precision, addr(integers),
        )
        writer.raw(b"TEXC")
        writer.string(uv_map.name)
        writer.string(uv_map.file_name)
        writer.f32(uv_map.precision)
        _pack_ints(writer, integers, len(mesh.vertices), 2, True, level)
    for attribute in mesh.attributes:
        integers = np.empty((len(mesh.vertices), 4), dtype=np.int32)
        lib().ctm_make_map_deltas(
            addr(attribute.values), addr(order), len(mesh.vertices), 4,
            attribute.precision, addr(integers),
        )
        writer.raw(b"ATTR")
        writer.string(attribute.name)
        writer.f32(attribute.precision)
        _pack_ints(writer, integers, len(mesh.vertices), 4, True, level)


def dumps(
    vertices: ArrayLike,
    triangles: ArrayLike,
    *,
    normals: ArrayLike | None = None,
    uv_maps: Mapping[str, ArrayLike] | list[UVMap] | None = None,
    attributes: Mapping[str, ArrayLike] | list[AttributeMap] | None = None,
    method: str = "MG2",
    compression_level: int = 6,
    vertex_precision: float = 1.0 / 1024.0,
    normal_precision: float = 1.0 / 256.0,
    comment: str = "",
) -> bytes:
    """Encode a triangle mesh as an OpenCTM v5 byte string."""
    mesh = _mesh(vertices, triangles, normals, uv_maps, attributes, comment, method)
    writer = _Writer()
    _write_header(writer, mesh)
    if mesh.method == "MG1":
        _encode_mg1(writer, mesh, compression_level)
    else:
        _encode_mg2(writer, mesh, compression_level, vertex_precision, normal_precision)
    return writer.buffer.getvalue()


def _decode_mg1(reader: _Reader, vertex_count: int, triangle_count: int, flags: int, uv_count: int, attribute_count: int, comment: str) -> Mesh:
    reader.expect(b"INDX")
    triangles = _unpack_ints(reader, triangle_count, 3, False).view(np.uint32)
    lib().ctm_restore_indices(addr(triangles), triangle_count)
    reader.expect(b"VERT")
    vertices = _unpack_floats(reader, vertex_count * 3, 1).reshape(vertex_count, 3)
    normals = None
    if flags & 1:
        reader.expect(b"NORM")
        normals = _unpack_floats(reader, vertex_count, 3)
    uv_maps = []
    for _ in range(uv_count):
        reader.expect(b"TEXC")
        name, file_name = reader.string(), reader.string()
        uv_maps.append(UVMap(_unpack_floats(reader, vertex_count, 2), name, file_name))
    attributes = []
    for _ in range(attribute_count):
        reader.expect(b"ATTR")
        name = reader.string()
        attributes.append(AttributeMap(_unpack_floats(reader, vertex_count, 4), name))
    return Mesh(vertices, triangles, normals, uv_maps, attributes, comment, "MG1")


def _decode_mg2(reader: _Reader, vertex_count: int, triangle_count: int, flags: int, uv_count: int, attribute_count: int, comment: str) -> Mesh:
    reader.expect(b"MG2H")
    vertex_precision, normal_precision = reader.f32(), reader.f32()
    minimum = np.array([reader.f32() for _ in range(3)], dtype=np.float32)
    maximum = np.array([reader.f32() for _ in range(3)], dtype=np.float32)
    division = np.array([reader.u32() for _ in range(3)], dtype=np.uint32)
    if (
        not np.isfinite(vertex_precision)
        or not np.isfinite(normal_precision)
        or vertex_precision <= 0
        or normal_precision <= 0
        or not np.isfinite(minimum).all()
        or not np.isfinite(maximum).all()
        or np.any(maximum < minimum)
        or np.any(division < 1)
    ):
        raise CTMError("invalid MG2 header")
    size = (maximum - minimum) / division.astype(np.float32)
    reader.expect(b"VERT")
    integer_vertices = _unpack_ints(reader, vertex_count, 3, False)
    reader.expect(b"GIDX")
    grid_indices = _unpack_ints(reader, vertex_count, 1, False).reshape(-1).view(np.uint32)
    lib().ctm_delta_decode_u32(addr(grid_indices), vertex_count)
    vertices = _restore_vertices(integer_vertices, grid_indices, minimum, size, division, vertex_precision)
    reader.expect(b"INDX")
    triangles = _unpack_ints(reader, triangle_count, 3, False).view(np.uint32)
    lib().ctm_restore_indices(addr(triangles), triangle_count)
    if np.any(triangles >= vertex_count):
        raise CTMError("decoded triangle index is outside the vertex array")
    normals = None
    if flags & 1:
        reader.expect(b"NORM")
        integer_normals = _unpack_ints(reader, vertex_count, 3, False)
        smooth = _smooth(vertices, triangles)
        normals = np.empty_like(vertices)
        lib().ctm_restore_normals(
            addr(integer_normals), addr(smooth), vertex_count, normal_precision, addr(normals)
        )
    uv_maps = []
    for _ in range(uv_count):
        reader.expect(b"TEXC")
        name, file_name, precision = reader.string(), reader.string(), reader.f32()
        if not np.isfinite(precision) or precision <= 0:
            raise CTMError("invalid UV precision")
        integers = _unpack_ints(reader, vertex_count, 2, True)
        values = np.empty((vertex_count, 2), dtype=np.float32)
        lib().ctm_restore_map(addr(integers), vertex_count, 2, precision, addr(values))
        uv_maps.append(UVMap(values, name, file_name, precision))
    attributes = []
    for _ in range(attribute_count):
        reader.expect(b"ATTR")
        name, precision = reader.string(), reader.f32()
        if not np.isfinite(precision) or precision <= 0:
            raise CTMError("invalid attribute precision")
        integers = _unpack_ints(reader, vertex_count, 4, True)
        values = np.empty((vertex_count, 4), dtype=np.float32)
        lib().ctm_restore_map(addr(integers), vertex_count, 4, precision, addr(values))
        attributes.append(AttributeMap(values, name, precision))
    return Mesh(vertices, triangles, normals, uv_maps, attributes, comment, "MG2")


def loads(data: bytes | bytearray | memoryview) -> Mesh:
    """Decode an OpenCTM v5 MG1 or MG2 stream."""
    reader = _Reader(data)
    reader.expect(b"OCTM")
    if reader.u32() != 5:
        raise CTMError("unsupported OpenCTM version")
    method_raw = reader.raw(4)
    if method_raw not in {b"MG1\0", b"MG2\0"}:
        raise CTMError("only MG1 and MG2 streams are supported")
    vertex_count, triangle_count = reader.u32(), reader.u32()
    uv_count, attribute_count, flags = reader.u32(), reader.u32(), reader.u32()
    comment = reader.string()
    if vertex_count == 0 or triangle_count == 0:
        raise CTMError("invalid empty OpenCTM mesh")
    if flags & ~1:
        raise CTMError("unsupported OpenCTM flags")
    if uv_count > MAX_DECODED_BYTES // 8 or attribute_count > MAX_DECODED_BYTES // 16:
        raise CTMError("map count is too large")
    largest_section = max(vertex_count * 16, triangle_count * 12)
    if largest_section > MAX_DECODED_BYTES:
        raise CTMError("mesh is too large")
    if method_raw == b"MG1\0":
        mesh = _decode_mg1(reader, vertex_count, triangle_count, flags, uv_count, attribute_count, comment)
    else:
        mesh = _decode_mg2(reader, vertex_count, triangle_count, flags, uv_count, attribute_count, comment)
    if reader.remaining():
        raise CTMError("trailing data after OpenCTM stream")
    return mesh


def save(path: str | Path, vertices: ArrayLike, triangles: ArrayLike, **kwargs: object) -> None:
    Path(path).write_bytes(dumps(vertices, triangles, **kwargs))


def load(path: str | Path) -> Mesh:
    return loads(Path(path).read_bytes())

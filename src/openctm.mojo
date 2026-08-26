"""Compute kernels for OpenCTM's MG1/MG2 codecs.

This is an altered, structure-of-arrays-free port of OpenCTM's C kernels.
Callers own every buffer; the C ABI passes them as integer addresses.
"""

from std.math import acos, atan2, ceil, cos, floor, pow, sin, sqrt
from std.sys.info import simd_width_of as simdwidthof

comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


def fp(address: Int) -> F32Ptr:
    return F32Ptr(unsafe_from_address=address)


def ip(address: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=address)


def up(address: Int) -> U32Ptr:
    return U32Ptr(unsafe_from_address=address)


def bp(address: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=address)


# OpenCTM: lib/compressMG1.c _ctmReArrangeTriangles
@export("ctm_rearrange_triangles")
def rearrange_triangles(indices_address: Int, triangle_count: Int) abi("C"):
    var indices = up(indices_address)
    for i in range(triangle_count):
        var base = i * 3
        var a = indices[base]
        var b = indices[base + 1]
        var c = indices[base + 2]
        if b < a and b < c:
            indices[base] = b
            indices[base + 1] = c
            indices[base + 2] = a
        elif c < a and c < b:
            indices[base] = c
            indices[base + 1] = a
            indices[base + 2] = b


@always_inline
def make_index_delta_range(indices: U32Ptr, triangle_count: Int):
    var i = triangle_count - 1
    comptime W = simdwidthof[DType.float64]()
    while i - W + 1 >= 1:
        var start = i - W + 1
        comptime
        if W == 4:
            var packed0 = indices.load[width=W](start * 3)
            var packed1 = indices.load[width=W](start * 3 + W)
            var packed2 = indices.load[width=W](start * 3 + 2 * W)
            var a01 = packed0.shuffle[0, 3, 0, 0]()
            var a23 = packed1.shuffle[2, 5, 0, 0](packed2)
            var a = a01.shuffle[0, 1, 4, 5](a23)
            var b01 = packed0.shuffle[1, 4, 0, 0](packed1)
            var b23 = packed1.shuffle[3, 6, 0, 0](packed2)
            var b = b01.shuffle[0, 1, 4, 5](b23)
            var c01 = packed0.shuffle[2, 5, 0, 0](packed1)
            var c23 = packed2.shuffle[0, 3, 0, 0]()
            var c = c01.shuffle[0, 1, 4, 5](c23)
            var previous_a = SIMD[DType.uint32, W](
                indices[start * 3 - 3], a[0], a[1], a[2]
            )
            var previous_b = SIMD[DType.uint32, W](
                indices[start * 3 - 2], b[0], b[1], b[2]
            )
            var delta_a = a - previous_a
            var delta_b = b - a.eq(previous_a).select(previous_b, a)
            var delta_c = c - a
            var result0 = SIMD[DType.uint32, W](
                delta_a[0], delta_b[0], delta_c[0], delta_a[1]
            )
            var result1 = SIMD[DType.uint32, W](
                delta_b[1], delta_c[1], delta_a[2], delta_b[2]
            )
            var result2 = SIMD[DType.uint32, W](
                delta_c[2], delta_a[3], delta_b[3], delta_c[3]
            )
            indices.store(start * 3, result0)
            indices.store(start * 3 + W, result1)
            indices.store(start * 3 + 2 * W, result2)
        else:
            var a = (indices + start * 3).strided_load[width=W](3)
            var b = (indices + start * 3 + 1).strided_load[width=W](3)
            var c = (indices + start * 3 + 2).strided_load[width=W](3)
            var previous_a = (indices + (start - 1) * 3).strided_load[width=W](3)
            var previous_b = (
                indices + (start - 1) * 3 + 1
            ).strided_load[width=W](3)
            (indices + start * 3 + 1).strided_store[width=W](
                b - a.eq(previous_a).select(previous_b, a), 3
            )
            (indices + start * 3 + 2).strided_store[width=W](c - a, 3)
            (indices + start * 3).strided_store[width=W](a - previous_a, 3)
        i -= W
    while i >= 1:
        var base = i * 3
        if indices[base] == indices[base - 3]:
            indices[base + 1] -= indices[base - 2]
        else:
            indices[base + 1] -= indices[base]
        indices[base + 2] -= indices[base]
        indices[base] -= indices[base - 3]
        i -= 1
    indices[1] -= indices[0]
    indices[2] -= indices[0]


# OpenCTM: lib/compressMG1.c _ctmMakeIndexDeltas
@export("ctm_make_index_deltas")
def make_index_deltas(indices_address: Int, triangle_count: Int) abi("C"):
    if triangle_count > 0:
        make_index_delta_range(up(indices_address), triangle_count)


def interleave_int_component(
    data: I32Ptr,
    bytes: U8Ptr,
    count: Int,
    size: Int,
    signed_ints: Int,
    k: Int,
):
    comptime W = simdwidthof[DType.float64]()
    var plane = count * size
    var i = 0
    while i + W <= count:
        var value = (data + i * size + k).strided_load[width=W](size)
        if signed_ints != 0:
            value = value.lt(0).select(-1 - (value << 1), value << 1)
        var x = value.cast[DType.uint32]()
        var offset = i + k * count
        bytes.store(offset + 3 * plane, (x & 0xff).cast[DType.uint8]())
        bytes.store(offset + 2 * plane, ((x >> 8) & 0xff).cast[DType.uint8]())
        bytes.store(offset + plane, ((x >> 16) & 0xff).cast[DType.uint8]())
        bytes.store(offset, (x >> 24).cast[DType.uint8]())
        i += W
    while i < count:
        var value = data[i * size + k]
        if signed_ints != 0:
            value = -1 - (value << 1) if value < 0 else value << 1
        var x = UInt32(value)
        var offset = i + k * count
        bytes[offset + 3 * plane] = UInt8(x & 0xff)
        bytes[offset + 2 * plane] = UInt8((x >> 8) & 0xff)
        bytes[offset + plane] = UInt8((x >> 16) & 0xff)
        bytes[offset] = UInt8((x >> 24) & 0xff)
        i += 1


# OpenCTM: lib/compressMG1.c _ctmRestoreIndices
@export("ctm_restore_indices")
def restore_indices(indices_address: Int, triangle_count: Int) abi("C"):
    var indices = up(indices_address)
    for i in range(triangle_count):
        var base = i * 3
        if i >= 1:
            indices[base] += indices[base - 3]
        indices[base + 2] += indices[base]
        if i >= 1 and indices[base] == indices[base - 3]:
            indices[base + 1] += indices[base - 2]
        else:
            indices[base + 1] += indices[base]


# OpenCTM: lib/stream.c _ctmStreamWritePackedInts
@export("ctm_interleave_ints")
def interleave_ints(
    data_address: Int,
    bytes_address: Int,
    count: Int,
    size: Int,
    signed_ints: Int,
) abi("C"):
    var data = ip(data_address)
    var bytes = bp(bytes_address)

    for k in range(size):
        interleave_int_component(data, bytes, count, size, signed_ints, k)


# OpenCTM: lib/stream.c _ctmStreamReadPackedInts
@export("ctm_deinterleave_ints")
def deinterleave_ints(
    bytes_address: Int,
    data_address: Int,
    count: Int,
    size: Int,
    signed_ints: Int,
) abi("C"):
    var bytes = bp(bytes_address)
    var data = ip(data_address)
    var plane = count * size
    for i in range(count):
        for k in range(size):
            var offset = i + k * count
            var x = (
                UInt32(bytes[offset + 3 * plane])
                | (UInt32(bytes[offset + 2 * plane]) << 8)
                | (UInt32(bytes[offset + plane]) << 16)
                | (UInt32(bytes[offset]) << 24)
            )
            if signed_ints != 0:
                data[i * size + k] = (
                    -Int32((x + 1) >> 1) if (x & 1) != 0 else Int32(x >> 1)
                )
            else:
                data[i * size + k] = Int32(x)


# OpenCTM: lib/stream.c _ctmStreamWritePackedFloats
@export("ctm_interleave_floats")
def interleave_floats(
    data_address: Int, bytes_address: Int, count: Int, size: Int
) abi("C"):
    var source = bp(data_address)
    var bytes = bp(bytes_address)
    var plane = count * size
    for i in range(count):
        for k in range(size):
            var scalar = i * size + k
            var offset = i + k * count
            bytes[offset + 3 * plane] = source[scalar * 4]
            bytes[offset + 2 * plane] = source[scalar * 4 + 1]
            bytes[offset + plane] = source[scalar * 4 + 2]
            bytes[offset] = source[scalar * 4 + 3]


# OpenCTM: lib/stream.c _ctmStreamReadPackedFloats
@export("ctm_deinterleave_floats")
def deinterleave_floats(
    bytes_address: Int, data_address: Int, count: Int, size: Int
) abi("C"):
    var bytes = bp(bytes_address)
    var destination = bp(data_address)
    var plane = count * size
    for i in range(count):
        for k in range(size):
            var scalar = i * size + k
            var offset = i + k * count
            destination[scalar * 4] = bytes[offset + 3 * plane]
            destination[scalar * 4 + 1] = bytes[offset + 2 * plane]
            destination[scalar * 4 + 2] = bytes[offset + plane]
            destination[scalar * 4 + 3] = bytes[offset]


# OpenCTM: lib/compressMG2.c _ctmSetupGrid
@export("ctm_setup_grid")
def setup_grid(
    vertices_address: Int,
    vertex_count: Int,
    minimum_address: Int,
    maximum_address: Int,
    division_address: Int,
    size_address: Int,
) abi("C"):
    var vertices = fp(vertices_address)
    var minimum = fp(minimum_address)
    var maximum = fp(maximum_address)
    var division = up(division_address)
    var grid_size = fp(size_address)
    for axis in range(3):
        minimum[axis] = vertices[axis]
        maximum[axis] = vertices[axis]
    for i in range(1, vertex_count):
        for axis in range(3):
            var value = vertices[i * 3 + axis]
            if value < minimum[axis]:
                minimum[axis] = value
            elif value > maximum[axis]:
                maximum[axis] = value
    var fx = maximum[0] - minimum[0]
    var fy = maximum[1] - minimum[1]
    var fz = maximum[2] - minimum[2]
    var total = fx + fy + fz
    if total > Float32(1.0e-30):
        var inverse = Float32(1.0) / total
        var wanted = pow(
            Float32(100.0) * Float32(vertex_count),
            Float32(1.0) / Float32(3.0),
        )
        division[0] = max(UInt32(1), UInt32(ceil(wanted * fx * inverse)))
        division[1] = max(UInt32(1), UInt32(ceil(wanted * fy * inverse)))
        division[2] = max(UInt32(1), UInt32(ceil(wanted * fz * inverse)))
    else:
        division[0] = 4
        division[1] = 4
        division[2] = 4
    for axis in range(3):
        grid_size[axis] = (
            (maximum[axis] - minimum[axis]) / Float32(division[axis])
        )


# OpenCTM: lib/compressMG2.c _ctmPointToGridIdx
@export("ctm_point_grid_indices")
def point_grid_indices(
    vertices_address: Int,
    vertex_count: Int,
    minimum_address: Int,
    size_address: Int,
    division_address: Int,
    indices_address: Int,
) abi("C"):
    var vertices = fp(vertices_address)
    var minimum = fp(minimum_address)
    var grid_size = fp(size_address)
    var division = up(division_address)
    var indices = up(indices_address)
    for i in range(vertex_count):
        var ix = UInt32(0)
        var iy = UInt32(0)
        var iz = UInt32(0)
        if grid_size[0] > 0.0:
            ix = UInt32(floor((vertices[i * 3] - minimum[0]) / grid_size[0]))
            if ix >= division[0]:
                ix = division[0] - 1
        if grid_size[1] > 0.0:
            iy = UInt32(floor((vertices[i * 3 + 1] - minimum[1]) / grid_size[1]))
            if iy >= division[1]:
                iy = division[1] - 1
        if grid_size[2] > 0.0:
            iz = UInt32(floor((vertices[i * 3 + 2] - minimum[2]) / grid_size[2]))
            if iz >= division[2]:
                iz = division[2] - 1
        indices[i] = ix + division[0] * (iy + division[1] * iz)


# OpenCTM: lib/compressMG2.c _ctmGridIdxToPoint
@always_inline
def grid_origin(
    grid_index: UInt32,
    minimum: F32Ptr,
    grid_size: F32Ptr,
    division: U32Ptr,
    axis: Int,
) -> Float32:
    var zdiv = division[0] * division[1]
    var iz = grid_index // zdiv
    var remainder = grid_index - iz * zdiv
    var iy = remainder // division[0]
    var ix = remainder - iy * division[0]
    if axis == 0:
        return Float32(ix) * grid_size[0] + minimum[0]
    if axis == 1:
        return Float32(iy) * grid_size[1] + minimum[1]
    return Float32(iz) * grid_size[2] + minimum[2]


# OpenCTM: lib/compressMG2.c _ctmMakeVertexDeltas
@export("ctm_make_vertex_deltas")
def make_vertex_deltas(
    vertices_address: Int,
    order_address: Int,
    grid_indices_address: Int,
    vertex_count: Int,
    minimum_address: Int,
    size_address: Int,
    division_address: Int,
    precision: Float32,
    integers_address: Int,
) abi("C"):
    var vertices = fp(vertices_address)
    var order = up(order_address)
    var grid_indices = up(grid_indices_address)
    var minimum = fp(minimum_address)
    var grid_size = fp(size_address)
    var division = up(division_address)
    var integers = ip(integers_address)
    var scale = Float32(1.0) / precision
    var previous_grid = UInt32(0x7fffffff)
    var previous_x = Int32(0)
    for i in range(vertex_count):
        var grid_index = grid_indices[i]
        var old_index = Int(order[i])
        var dx = Int32(
            floor(
                scale
                * (vertices[old_index * 3] - grid_origin(
                    grid_index, minimum, grid_size, division, 0
                ))
                + Float32(0.5)
            )
        )
        integers[i * 3] = dx - previous_x if grid_index == previous_grid else dx
        integers[i * 3 + 1] = Int32(
            floor(
                scale
                * (vertices[old_index * 3 + 1] - grid_origin(
                    grid_index, minimum, grid_size, division, 1
                ))
                + Float32(0.5)
            )
        )
        integers[i * 3 + 2] = Int32(
            floor(
                scale
                * (vertices[old_index * 3 + 2] - grid_origin(
                    grid_index, minimum, grid_size, division, 2
                ))
                + Float32(0.5)
            )
        )
        previous_grid = grid_index
        previous_x = dx


# OpenCTM: lib/compressMG2.c _ctmRestoreVertices
@export("ctm_restore_vertices")
def restore_vertices(
    integers_address: Int,
    grid_indices_address: Int,
    vertex_count: Int,
    minimum_address: Int,
    size_address: Int,
    division_address: Int,
    precision: Float32,
    vertices_address: Int,
) abi("C"):
    var integers = ip(integers_address)
    var grid_indices = up(grid_indices_address)
    var minimum = fp(minimum_address)
    var grid_size = fp(size_address)
    var division = up(division_address)
    var vertices = fp(vertices_address)
    var previous_grid = UInt32(0x7fffffff)
    var previous_x = Int32(0)
    for i in range(vertex_count):
        var grid_index = grid_indices[i]
        var dx = integers[i * 3]
        if grid_index == previous_grid:
            dx += previous_x
        vertices[i * 3] = (
            precision * Float32(dx)
            + grid_origin(grid_index, minimum, grid_size, division, 0)
        )
        vertices[i * 3 + 1] = (
            precision * Float32(integers[i * 3 + 1])
            + grid_origin(grid_index, minimum, grid_size, division, 1)
        )
        vertices[i * 3 + 2] = (
            precision * Float32(integers[i * 3 + 2])
            + grid_origin(grid_index, minimum, grid_size, division, 2)
        )
        previous_grid = grid_index
        previous_x = dx


# OpenCTM: lib/compressMG2.c _ctmCalcSmoothNormals
@export("ctm_smooth_normals")
def smooth_normals(
    vertices_address: Int,
    vertex_count: Int,
    indices_address: Int,
    triangle_count: Int,
    normals_address: Int,
) abi("C"):
    var vertices = fp(vertices_address)
    var indices = up(indices_address)
    var normals = fp(normals_address)
    for i in range(vertex_count * 3):
        normals[i] = 0.0
    for i in range(triangle_count):
        var a = Int(indices[i * 3])
        var b = Int(indices[i * 3 + 1])
        var c = Int(indices[i * 3 + 2])
        var e1x = vertices[b * 3] - vertices[a * 3]
        var e1y = vertices[b * 3 + 1] - vertices[a * 3 + 1]
        var e1z = vertices[b * 3 + 2] - vertices[a * 3 + 2]
        var e2x = vertices[c * 3] - vertices[a * 3]
        var e2y = vertices[c * 3 + 1] - vertices[a * 3 + 1]
        var e2z = vertices[c * 3 + 2] - vertices[a * 3 + 2]
        var nx = e1y * e2z - e1z * e2y
        var ny = e1z * e2x - e1x * e2z
        var nz = e1x * e2y - e1y * e2x
        var length = sqrt(nx * nx + ny * ny + nz * nz)
        var inverse = Float32(1.0) / length if length > 1.0e-10 else Float32(1.0)
        nx *= inverse
        ny *= inverse
        nz *= inverse
        normals[a * 3] += nx
        normals[a * 3 + 1] += ny
        normals[a * 3 + 2] += nz
        normals[b * 3] += nx
        normals[b * 3 + 1] += ny
        normals[b * 3 + 2] += nz
        normals[c * 3] += nx
        normals[c * 3 + 1] += ny
        normals[c * 3 + 2] += nz
    for i in range(vertex_count):
        var nx = normals[i * 3]
        var ny = normals[i * 3 + 1]
        var nz = normals[i * 3 + 2]
        var length = sqrt(nx * nx + ny * ny + nz * nz)
        var inverse = Float32(1.0) / length if length > 1.0e-10 else Float32(1.0)
        normals[i * 3] *= inverse
        normals[i * 3 + 1] *= inverse
        normals[i * 3 + 2] *= inverse


# OpenCTM: lib/compressMG2.c _ctmMakeNormalCoordSys
@always_inline
def normal_basis_component(nx: Float32, ny: Float32, nz: Float32, item: Int) -> Float32:
    var xx = -ny
    var xy = nx - nz
    var xz = ny
    var length = sqrt(Float32(2.0) * xx * xx + xy * xy)
    if length > 1.0e-20:
        var inverse = Float32(1.0) / length
        xx *= inverse
        xy *= inverse
        xz *= inverse
    var yx = ny * xz - nz * xy
    var yy = nz * xx - nx * xz
    var yz = nx * xy - ny * xx
    if item == 0:
        return xx
    if item == 1:
        return xy
    if item == 2:
        return xz
    if item == 3:
        return yx
    if item == 4:
        return yy
    if item == 5:
        return yz
    if item == 6:
        return nx
    if item == 7:
        return ny
    return nz


# OpenCTM: lib/compressMG2.c _ctmMakeNormalDeltas
@export("ctm_make_normal_deltas")
def make_normal_deltas(
    normals_address: Int,
    order_address: Int,
    smooth_address: Int,
    vertex_count: Int,
    precision: Float32,
    integers_address: Int,
) abi("C"):
    var normals = fp(normals_address)
    var order = up(order_address)
    var smooth = fp(smooth_address)
    var integers = ip(integers_address)
    var scale = Float32(1.0) / precision
    var pi = Float32(3.141592653589793238462643)
    for i in range(vertex_count):
        var old_index = Int(order[i])
        var ox = normals[old_index * 3]
        var oy = normals[old_index * 3 + 1]
        var oz = normals[old_index * 3 + 2]
        var magnitude = sqrt(ox * ox + oy * oy + oz * oz)
        if magnitude < 1.0e-10:
            magnitude = 1.0
        var sx = smooth[i * 3]
        var sy = smooth[i * 3 + 1]
        var sz = smooth[i * 3 + 2]
        if sx * ox + sy * oy + sz * oz < 0.0:
            magnitude = -magnitude
        integers[i * 3] = Int32(floor(scale * magnitude + Float32(0.5)))
        var inverse = Float32(1.0) / magnitude
        var nx = ox * inverse
        var ny = oy * inverse
        var nz = oz * inverse
        var rx = (
            normal_basis_component(sx, sy, sz, 0) * nx
            + normal_basis_component(sx, sy, sz, 1) * ny
            + normal_basis_component(sx, sy, sz, 2) * nz
        )
        var ry = (
            normal_basis_component(sx, sy, sz, 3) * nx
            + normal_basis_component(sx, sy, sz, 4) * ny
            + normal_basis_component(sx, sy, sz, 5) * nz
        )
        var rz = (
            normal_basis_component(sx, sy, sz, 6) * nx
            + normal_basis_component(sx, sy, sz, 7) * ny
            + normal_basis_component(sx, sy, sz, 8) * nz
        )
        var phi = Float32(0.0) if rz >= 1.0 else acos(rz)
        var theta = atan2(ry, rx)
        var int_phi = Int32(
            floor(phi * (scale / (Float32(0.5) * pi)) + Float32(0.5))
        )
        var theta_scale = Float32(0.0)
        if int_phi != 0:
            theta_scale = (
                Float32(2.0) / pi
                if int_phi <= 4
                else Float32(int_phi) / (Float32(2.0) * pi)
            )
        integers[i * 3 + 1] = int_phi
        integers[i * 3 + 2] = Int32(
            floor((theta + pi) * theta_scale + Float32(0.5))
        )


# OpenCTM: lib/compressMG2.c _ctmRestoreNormals
@export("ctm_restore_normals")
def restore_normals(
    integers_address: Int,
    smooth_address: Int,
    vertex_count: Int,
    precision: Float32,
    normals_address: Int,
) abi("C"):
    var integers = ip(integers_address)
    var smooth = fp(smooth_address)
    var normals = fp(normals_address)
    var pi = Float32(3.141592653589793238462643)
    for i in range(vertex_count):
        var magnitude = Float32(integers[i * 3]) * precision
        var int_phi = integers[i * 3 + 1]
        var phi = Float32(int_phi) * Float32(0.5) * pi * precision
        var theta_scale = Float32(0.0)
        if int_phi != 0:
            theta_scale = (
                pi / Float32(2.0)
                if int_phi <= 4
                else Float32(2.0) * pi / Float32(int_phi)
            )
        var theta = Float32(integers[i * 3 + 2]) * theta_scale - pi
        var rx = sin(phi) * cos(theta)
        var ry = sin(phi) * sin(theta)
        var rz = cos(phi)
        var sx = smooth[i * 3]
        var sy = smooth[i * 3 + 1]
        var sz = smooth[i * 3 + 2]
        normals[i * 3] = magnitude * (
            normal_basis_component(sx, sy, sz, 0) * rx
            + normal_basis_component(sx, sy, sz, 3) * ry
            + normal_basis_component(sx, sy, sz, 6) * rz
        )
        normals[i * 3 + 1] = magnitude * (
            normal_basis_component(sx, sy, sz, 1) * rx
            + normal_basis_component(sx, sy, sz, 4) * ry
            + normal_basis_component(sx, sy, sz, 7) * rz
        )
        normals[i * 3 + 2] = magnitude * (
            normal_basis_component(sx, sy, sz, 2) * rx
            + normal_basis_component(sx, sy, sz, 5) * ry
            + normal_basis_component(sx, sy, sz, 8) * rz
        )


# OpenCTM: lib/compressMG2.c _ctmMakeUVCoordDeltas and _ctmMakeAttribDeltas
@export("ctm_make_map_deltas")
def make_map_deltas(
    values_address: Int,
    order_address: Int,
    vertex_count: Int,
    channels: Int,
    precision: Float32,
    integers_address: Int,
) abi("C"):
    var values = fp(values_address)
    var order = up(order_address)
    var integers = ip(integers_address)
    var scale = Float32(1.0) / precision
    for channel in range(channels):
        var previous = Int32(0)
        for i in range(vertex_count):
            var old_index = Int(order[i])
            var value = Int32(
                floor(
                    scale * values[old_index * channels + channel]
                    + Float32(0.5)
                )
            )
            integers[i * channels + channel] = value - previous
            previous = value


# OpenCTM: lib/compressMG2.c _ctmRestoreUVCoords and _ctmRestoreAttribs
@export("ctm_restore_map")
def restore_map(
    integers_address: Int,
    vertex_count: Int,
    channels: Int,
    precision: Float32,
    values_address: Int,
) abi("C"):
    var integers = ip(integers_address)
    var values = fp(values_address)
    for channel in range(channels):
        var previous = Int32(0)
        for i in range(vertex_count):
            var value = integers[i * channels + channel] + previous
            values[i * channels + channel] = Float32(value) * precision
            previous = value


# OpenCTM: lib/compressMG2.c grid-index delta loops
@export("ctm_delta_encode_u32")
def delta_encode_u32(values_address: Int, count: Int) abi("C"):
    var values = up(values_address)
    var i = count - 1
    while i >= 1:
        values[i] -= values[i - 1]
        i -= 1


# OpenCTM: lib/compressMG2.c grid-index restoration loop
@export("ctm_delta_decode_u32")
def delta_decode_u32(values_address: Int, count: Int) abi("C"):
    var values = up(values_address)
    for i in range(1, count):
        values[i] += values[i - 1]

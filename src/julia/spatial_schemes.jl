## spatial schemes: the averages and differences of the C-grid

"""
    SpatialScheme

Discretisation of the averages and differences on the C-grid, given by `operators(scheme)`. A scheme supplies
matching pairs (avg_x with dif_X, …): energy consistency relies on their summation by parts.
"""
abstract type SpatialScheme end

"""
    Centred2()

Second-order centred scheme: two-point averages and differences (see `avg_x`, `dif_x`).
"""
struct Centred2 <: SpatialScheme end

"""
    ops = operators(scheme::SpatialScheme)

Return the averages and differences of `scheme` as a named tuple: `avg_x, avg_z, avg_X, avg_Z, avg_xz, avg_xZ,
avg_Xz, avg_XZ, dif_x, dif_z, dif_X, dif_Z` (lower case: onto cells, upper case: onto faces). Unpack them in a
kernel, `(; avg_X, dif_X) = operators(model.spatial_scheme)`, at no runtime cost.
"""
operators(::Centred2) = (; avg_x, avg_z, avg_X, avg_Z, avg_xz, avg_xZ, avg_Xz, avg_XZ, dif_x, dif_z, dif_X, dif_Z)

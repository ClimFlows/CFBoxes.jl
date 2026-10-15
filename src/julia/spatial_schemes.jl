## spatial schemes: the interpolations and differences of the C-grid

"""
    SpatialScheme

Discretisation of the interpolation and difference on the C-grid, given by `operators(scheme)`. A scheme supplies a
matching pair: energy consistency relies on the summation by parts of `δ` onto faces with `ℑ` onto cells.
"""
abstract type SpatialScheme end

"""
    Centred2()

Second-order centred scheme: two-point interpolation and difference (see `ℑ`, `δ`).
"""
struct Centred2 <: SpatialScheme end

"""
    ops = operators(scheme::SpatialScheme)

Return the interpolation `ℑ` and difference `δ` of `scheme` as a named tuple. Unpack them in a kernel,
`(; ℑ, δ) = operators(spatial_scheme)`, at no runtime cost.
"""
operators(::Centred2) = (; ℑ, δ)

"""
    halo_width(scheme::SpatialScheme)

Return the number of halo cells the stencils of `scheme` reach beyond the interior.
"""
halo_width(::Centred2) = 1

"""
    check_halo(domain, scheme::SpatialScheme)

Check that the halo of `domain` is wide enough for the stencils of `scheme`.
"""
check_halo(domain, scheme::SpatialScheme) =
    all(≥(halo_width(scheme)), halo_size(domain)) || error("$scheme needs a halo of $(halo_width(scheme)) cells, got $(halo_size(domain)).")

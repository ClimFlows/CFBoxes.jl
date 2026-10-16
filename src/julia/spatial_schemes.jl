# Spatial schemes: the interpolation and difference operators of the C-grid

"""
    SpatialScheme

Discretization of the interpolation `ℑ` and difference `δ` on the C-grid, given by `operators(scheme)`. The pair
must sum by parts (δ onto the faces against ℑ onto the cells), on which energy consistency relies.
"""
abstract type SpatialScheme end

"""
    Centred2()

Second-order centered scheme: two-point interpolation and difference (see `ℑ`, `δ`).
"""
struct Centred2 <: SpatialScheme end

"""
    ops = operators(scheme::SpatialScheme)

The interpolation `ℑ` and difference `δ` of `scheme`, as a named tuple to unpack in a kernel at no runtime cost:
`(; ℑ, δ) = operators(spatial_scheme)`.
"""
operators(::Centred2) = (; ℑ, δ)

"""
    halo_width(scheme::SpatialScheme)

Number of halo cells that the stencils of `scheme` reach beyond the interior.
"""
halo_width(::Centred2) = 1

"""
    check_halo(domain, scheme::SpatialScheme)

Check that the halo of `domain` is at least `halo_width(scheme)` cells wide along every dimension.
"""
check_halo(domain, scheme::SpatialScheme) =
    all(≥(halo_width(scheme)), halo_size(domain)) || error("$scheme needs a halo of $(halo_width(scheme)) cells, got $(halo_size(domain)).")

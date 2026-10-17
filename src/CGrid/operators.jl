#=
Interpolation and difference operators of the second-order centered scheme on a C-grid (see `Centred2`).

ℑ(a, dims, to, I...) interpolates `a` half a cell along each direction in the tuple `dims` (dimension numbers, 1 = x,
2 = z in 2D), onto the staggering in the tuple `to`, one per direction:
  Face(): onto the faces along that direction (reads i-1 and i);
  Cell(): onto the cell centers along that direction (reads i and i+1).
δ(a, (d,), (to,), I...) differences along one direction in the same way. Examples:
  ℑ(m, (1,), (Face(),), I, j)                 cell → u-face
  ℑ(ωᴵᴶ, (2,), (Cell(),), I, j)               vertex → u-face
  ℑ(ρ, (1, 2), (Face(), Face()), I, J)        cell → vertex
  ℑ(u, (1, 2), (Cell(), Face()), i, J)        u-face → w-face
The staggering is given per direction because the same place is reached from different sources in opposite
directions (a u-face is Face() along 1 from a cell, Cell() along 2 from a vertex). Arrays carry no grid location, so
asking for the staggering an array already has along a direction (e.g. Cell() along 1 of a cell array) is a usage
error, not a no-op.

Shorthand in comments: upper case onto the faces, lower case onto the cells, ℑX(a) = ℑ(a, (1,), (Face(),)),
ℑz(a) = ℑ(a, (2,), (Cell(),)) in 2D, ℑXZ(a) = ℑ(a, (1, 2), (Face(), Face())), and likewise δ. Scratch names carry
the interpolations applied as superscripts: ρᴵ = ℑX(ρ) on the u-faces, ρᴵᴶ = ℑXZ(ρ) on the vertices.

`a` is an array, a vertical profile (a vector, e.g. ρ_R(z), read along z), or a tuple of factors multiplied point
by point, e.g. ℑ((ρ_R, u), (2,), (Face(),), I, J) interpolates ρ_R u. Indices stay separate arguments, so that `@vec`
can pass a SIMD range as i. The direction tuples are literals at the call sites and the operators are inlined, so
each direction is known at compile time.
=#

# the only dimension-specific code of the operators
"""
    lo, hi = stencil_points(s, Val(d), I...)

The two points (lower, upper) read by a stencil onto the staggering `s` (`Face()` or `Cell()`) along dimension `d`,
at index `I`: a 2D or 3D index, or a single index along z for a vertical profile.
"""
@inline stencil_points(::Cell, ::Val{1}, i, j) = ((i, j), (i+1, j))
@inline stencil_points(::Face, ::Val{1}, i, j) = ((i-1, j), (i, j))
@inline stencil_points(::Cell, ::Val{2}, i, j) = ((i, j), (i, j+1))
@inline stencil_points(::Face, ::Val{2}, i, j) = ((i, j-1), (i, j))

@inline stencil_points(::Cell, ::Val{1}, i, j, k) = ((i, j, k), (i+1, j, k))
@inline stencil_points(::Face, ::Val{1}, i, j, k) = ((i-1, j, k), (i, j, k))
@inline stencil_points(::Cell, ::Val{2}, i, j, k) = ((i, j, k), (i, j+1, k))
@inline stencil_points(::Face, ::Val{2}, i, j, k) = ((i, j-1, k), (i, j, k))
@inline stencil_points(::Cell, ::Val{3}, i, j, k) = ((i, j, k), (i, j, k+1))
@inline stencil_points(::Face, ::Val{3}, i, j, k) = ((i, j, k-1), (i, j, k))

# vertical profile (one index, along z)
@inline stencil_points(::Cell, ::Val, j) = ((j,), (j+1,))
@inline stencil_points(::Face, ::Val, j) = ((j-1,), (j,))

"""
    point(a, I...)

Value of `a` at the grid point `I`: `a[i, j]` for an array, `a[j]` (`a[k]` in 3D) for a vertical profile, and the
product of the factors for a tuple.
"""
Base.@propagate_inbounds point(a::AbstractArray, I...) = a[I...]
Base.@propagate_inbounds point(v::AbstractVector, i, j) = v[j]
Base.@propagate_inbounds point(v::AbstractVector, i, j, k) = v[k]
Base.@propagate_inbounds point((a, b)::Tuple, I...) = point(a, I...) * point(b, I...)

"""
    ℑ(a, dims::Tuple, to::Tuple, I...)

Interpolation of `a` at index `I`, half a cell along each direction in `dims` (dimension numbers), onto the staggering
`to` (one `Face()` or `Cell()` per direction): the mean of two points for one direction, four for two. `a` is an
array, a vertical profile (a vector, read along z), or a tuple of factors multiplied point by point.
"""
Base.@propagate_inbounds ℑ(a, ::Tuple{}, ::Tuple{}, I...) = point(a, I...)
Base.@propagate_inbounds function ℑ(a, (d, dims...)::Tuple, (s, to...)::Tuple, I...)
    lo, hi = stencil_points(s, Val(d), I...)
    return 0.5 * (ℑ(a, dims, to, lo...) + ℑ(a, dims, to, hi...))
end

"""
    δ(a, (d,), (to,), I...)

Two-point difference of `a` at index `I` along dimension `d`, onto the staggering `to` (`Face()` or `Cell()`): the
upper point minus the lower one, to divide by the grid spacing for a derivative. `a` is as for `ℑ`.
"""
Base.@propagate_inbounds function δ(a, (d,)::Tuple{Any}, (s,)::Tuple{Any}, I...)
    lo, hi = stencil_points(s, Val(d), I...)
    return point(a, hi...) - point(a, lo...)
end

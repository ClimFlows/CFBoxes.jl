#=
Boxes: M interior cells and H halo cells on each side along every dimension, numbered (x, z) in 2D and (x, y, z)
in 3D. Indices run over the whole array, halo included: interior cell i is at index H + i.
=#

"""
    AbstractBox{Mx, My, Mz, Bx, By, Bz}

A rectangular box of `Mx × My × Mz` interior cells (`My = 0` in 2D), with boundary topology `Bx`, `By`, `Bz`
(`Periodic` or `Bounded`; `By = Nothing` in 2D).
"""
abstract type AbstractBox{Mx, My, Mz, Bx, By, Bz} end

"""
    Box2D(; Mx, Mz, Hx = 1, Hz = 1, boundary = (Periodic, Bounded))

2D box of `Mx × Mz` interior cells with `Hx`, `Hz` halo cells on each side, and the boundary topology `(Bx, Bz)`.
"""
struct Box2D{Mx, Mz, Hx, Hz, Bx, Bz} <: AbstractBox{Mx, 0, Mz, Bx, Nothing, Bz} end
Box2D(; Mx, Mz, Hx = 1, Hz = 1, boundary = (Periodic, Bounded)) = Box2D{Mx, Mz, Hx, Hz, boundary[1], boundary[2]}()

"""
    Box3D(; Mx, My, Mz, Hx = 1, Hy = 1, Hz = 1, boundary = (Periodic, Periodic, Bounded))

3D box of `Mx × My × Mz` interior cells with `Hx`, `Hy`, `Hz` halo cells on each side, and the boundary topology
`(Bx, By, Bz)`.
"""
struct Box3D{Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz} <: AbstractBox{Mx, My, Mz, Bx, By, Bz} end
Box3D(; Mx, My, Mz, Hx = 1, Hy = 1, Hz = 1, boundary = (Periodic, Periodic, Bounded)) =
    Box3D{Mx, My, Mz, Hx, Hy, Hz, boundary[1], boundary[2], boundary[3]}()

"""
    dims(box) -> NTuple

Number of interior cells along each dimension.
"""
@inline dims(::Box2D{Mx, Mz}) where {Mx, Mz} = (Mx, Mz)
@inline dims(::Box3D{Mx, My, Mz}) where {Mx, My, Mz} = (Mx, My, Mz)

"""
    halo_size(box) -> NTuple

Number of halo cells on each side, along each dimension.
"""
@inline halo_size(::Box2D{Mx, Mz, Hx, Hz}) where {Mx, Mz, Hx, Hz} = (Hx, Hz)
@inline halo_size(::Box3D{Mx, My, Mz, Hx, Hy, Hz}) where {Mx, My, Mz, Hx, Hy, Hz} = (Hx, Hy, Hz)

"""
    boundary_topology(box) -> NTuple

Boundary topology (`Periodic()` or `Bounded()`) along each dimension.
"""
@inline boundary_topology(::Box2D{Mx, Mz, Hx, Hz, Bx, Bz}) where {Mx, Mz, Hx, Hz, Bx, Bz} = (Bx(), Bz())
@inline boundary_topology(::Box3D{Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz}) where {Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz} =
    (Bx(), By(), Bz())

# Grid along one dimension d: a bounded dimension has one more face than cells (both walls), a periodic one as many

"""
    Face()

Staggering on the faces along one dimension (see `Cell`). A tuple of `Face()`/`Cell()`, one per dimension, is a grid
location, e.g. `(Face(), Cell())` for the u-faces of a 2D box (see `alloc`). The operators `ℑ` and `δ` take one per
direction for the staggering of their result.
"""
struct Face end

"""
    Cell()

Staggering at the cell centers along one dimension (see `Face`).
"""
struct Cell end

"""
    extra_face(topology) -> Int

Number of faces beyond the cell count along a dimension: 1 if `Bounded` (both walls are faces), 0 if `Periodic`.
"""
@inline extra_face(::Periodic) = 0
@inline extra_face(::Bounded) = 1

"""
    ncells(box, d) -> Int

Number of cells along dimension `d`, halo included.
"""
@inline ncells(box::AbstractBox, d) = dims(box)[d] + 2halo_size(box)[d]

"""
    nfaces(box, d) -> Int

Number of faces along dimension `d`, halo included: one more than `ncells` if `d` is bounded.
"""
@inline nfaces(box::AbstractBox, d) = ncells(box, d) + extra_face(boundary_topology(box)[d])

"""
    cells(box, d) -> UnitRange

Indices along dimension `d`, halo included, at which a stencil onto the cells (`ℑ`, `δ` with `Cell()` along `d`)
stays inside the array: the loop range of the kernels that write at the cells.
"""
@inline cells(box::AbstractBox, d) = 1:nfaces(box, d)-1

"""
    faces(box, d) -> UnitRange

Indices along dimension `d`, halo included, at which a stencil onto the faces (`ℑ`, `δ` with `Face()` along `d`)
stays inside the array: the loop range of the kernels that write on the faces.
"""
@inline faces(box::AbstractBox, d) = 2:ncells(box, d)

"""
    interior_cells(box, d) -> UnitRange

Indices of the interior cells along dimension `d`.
"""
@inline interior_cells(box::AbstractBox, d) = halo_size(box)[d]+1 : halo_size(box)[d]+dims(box)[d]

"""
    interior_faces(box, d) -> UnitRange

Indices of the interior faces along dimension `d`. When `d` is bounded, the first and last are the walls.
"""
@inline interior_faces(box::AbstractBox, d) =
    halo_size(box)[d]+1 : halo_size(box)[d]+dims(box)[d]+extra_face(boundary_topology(box)[d])

"""
    cellpoint(box, Δ, d, i)

Coordinate of the center of cell `i` along dimension `d`, with grid spacing `Δ`; the first interior face is at 0.
"""
@inline cellpoint(box::AbstractBox, Δ, d, i) = (i - halo_size(box)[d] - 0.5) * Δ

"""
    facepoint(box, Δ, d, i)

Coordinate of face `i` along dimension `d`, with grid spacing `Δ`; the first interior face (the first wall, if
bounded) is at 0.
"""
@inline facepoint(box::AbstractBox, Δ, d, i) = (i - halo_size(box)[d] - 1) * Δ

"""
    cellgrid(box, Δ, d) -> Vector

Coordinates of all the cell centers along dimension `d`, halo included (see `cellpoint`).
"""
cellgrid(box::AbstractBox, Δ, d) = [cellpoint(box, Δ, d, i) for i in 1:ncells(box, d)]

"""
    facegrid(box, Δ, d) -> Vector

Coordinates of all the faces along dimension `d`, halo included (see `facepoint`).
"""
facegrid(box::AbstractBox, Δ, d) = [facepoint(box, Δ, d, i) for i in 1:nfaces(box, d)]

"""
    npoints(box, Face(), d) -> Int
    npoints(box, Cell(), d) -> Int

Number of points along dimension `d` of an array on the faces or at the cell centers, halo included.
"""
@inline npoints(box::AbstractBox, ::Face, d) = nfaces(box, d)
@inline npoints(box::AbstractBox, ::Cell, d) = ncells(box, d)

# Arrays on a grid location

"""
    interior(box, a) -> SubArray

View of the interior of the array `a`, on any grid location: its halo removed (on the faces of a bounded dimension,
both walls are kept).
"""
interior(::Box2D{Mx, Mz, Hx, Hz}, a::AbstractArray) where {Mx, Mz, Hx, Hz} = @view a[1+Hx:end-Hx, 1+Hz:end-Hz]
interior(::Box3D{Mx, My, Mz, Hx, Hy, Hz}, a::AbstractArray) where {Mx, My, Mz, Hx, Hy, Hz} =
    @view a[1+Hx:end-Hx, 1+Hy:end-Hy, 1+Hz:end-Hz]

"""
    dimensions(location::Tuple) -> NTuple

The dimension numbers `(1, 2, …)` of a grid location.
"""
@inline dimensions(location::Tuple) = ntuple(identity, Val(length(location)))

"""
    coordinates(box, spacing, location::Tuple) -> Tuple

Coordinates of the interior points of the grid `location`, one vector per dimension, matching `interior(box, a)` for
an array `a` on that location. `spacing` holds the grid spacing along each dimension, e.g. `(dx, dz)` in 2D.
"""
coordinates(box::AbstractBox, spacing, location::Tuple) =
    map((s, d) -> interior_grid(box, spacing[d], s, d), location, dimensions(location))

"""
    interior_grid(box, Δ, s, d) -> Vector

Coordinates of the interior points along dimension `d` of an array with staggering `s` (`Face()` or `Cell()`).
"""
interior_grid(box, Δ, ::Face, d) = facegrid(box, Δ, d)[interior_faces(box, d)]
interior_grid(box, Δ, ::Cell, d) = cellgrid(box, Δ, d)[interior_cells(box, d)]

"""
    alloc(F::Type, mgr, box, location::Tuple, dims...) -> Array

Zero array of element type `F` on the grid `location`: one `Face()` or `Cell()` per dimension, e.g. `(Cell(), Cell())`
for the cells, `(Face(), Cell())` the u-faces, `(Cell(), Face())` the w-faces and `(Face(), Face())` the vertices of a
2D box. Trailing `dims` add extra dimensions (e.g. tracers). Starting from zero defines the halo points that no kernel
writes (e.g. beyond walls).

The loop manager `mgr` writes the zeros (`nothing`: serially), its threads sharing out the last box dimension as in
the kernels, so that on a multi-socket node each thread's memory pages are placed on its own socket.
"""
function alloc(F::Type, mgr, box::AbstractBox, location::Tuple, dims...)
    a = Array{F}(undef, map((s, d) -> npoints(box, s, d), location, dimensions(location))..., dims...)
    return zero!(a, mgr, box)
end

"""
    alloc!(a, F::Type, mgr, box, location::Tuple, dims...) -> Array

Return `a` if it is an array, else a new array from `alloc` (e.g. for `a = void` from MutatingOrNot): how kernels get
their scratch space, as `similar!` does.
"""
alloc!(a::AbstractArray, F::Type, mgr, box::AbstractBox, location::Tuple, dims...) = a
alloc!(_, F::Type, mgr, box::AbstractBox, location::Tuple, dims...) = alloc(F, mgr, box, location, dims...)

"""
    zero!(a, mgr, box) -> a

Set `a` to zero with the threads of `mgr`, each on the levels (last box dimension) it handles in the kernels; trailing
extra dimensions are done one at a time.
"""
function zero!(a, mgr, ::Box2D)
    # plain loops: @vec cannot store a value independent of i
    levels = reshape(a, size(a, 1), size(a, 2), :)                         # (x, z, extra)
    for n in axes(levels, 3)
        @with mgr, let (irange, jrange) = (axes(levels, 1), axes(levels, 2))
            @inbounds for j in jrange, i in irange
                levels[i, j, n] = 0
            end
        end
    end
    return a
end
function zero!(a, mgr, ::Box3D)
    levels = reshape(a, size(a, 1), size(a, 2), size(a, 3), :)             # (x, y, z, extra)
    for n in axes(levels, 4)
        @with mgr, let (irange, jrange, krange) = (axes(levels, 1), axes(levels, 2), axes(levels, 3))
            @inbounds for k in krange, j in jrange, i in irange
                levels[i, j, k, n] = 0
            end
        end
    end
    return a
end

# Indices across one dimension d, known at compile time

"""
    slice(a, ::Val{d}) -> CartesianIndices

The indices of `a` with dimension `d` removed: one per line of points along `d` (e.g. per position along a wall
normal to `d`). See `across`.
"""
@inline slice(a::AbstractArray{T, N}, ::Val{d}) where {T, N, d} =
    CartesianIndices((ntuple(e -> axes(a, e), Val(d-1))..., ntuple(e -> axes(a, d+e), Val(N-d))...))

"""
    across(I, ::Val{d}, k) -> CartesianIndex

The index of `a` that puts `k` at dimension `d` of the `slice` index `I`.
"""
@inline across(I::CartesianIndex{N}, ::Val{d}, k) where {N, d} =
    CartesianIndex(ntuple(e -> I[e], Val(d-1))..., k, ntuple(e -> I[d-1+e], Val(N-d+1))...)

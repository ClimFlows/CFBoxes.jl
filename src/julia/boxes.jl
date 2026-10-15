# Definition of box domains with halo cells for finite difference methods
# - the following code assumes a centred finite difference scheme on a C-grid

"""
    AbstractBox{Mx, My, Mz, Bx, By, Bz} <: FDDomain

Abstract type representing a finite difference domain box.

# Type Parameters
- `Mx`: Number of cells in x-direction
- `My`: Number of cells in y-direction
- `Mz`: Number of cells in z-direction
- `Bx`: Boundary topology in x-direction
- `By`: Boundary topology in y-direction
- `Bz`: Boundary topology in z-direction
"""
abstract type AbstractBox{Mx, My, Mz, Bx, By, Bz} end

"""
    AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz} <: AbstractBox{Mx, My, Mz, Bx, By, Bz}

Abstract type representing a box with halo (ghost) cells for at least two dimensions.

# Type Parameters
- `Mx`: Number of cells in x-direction
- `My`: Number of cells in y-direction
- `Mz`: Number of cells in z-direction
- `Hx`: Number of halo cells in x-direction
- `Hy`: Number of halo cells in y-direction
- `Hz`: Number of halo cells in z-direction
- `Bx`: Boundary topology in x-direction
- `By`: Boundary topology in y-direction
- `Bz`: Boundary topology in z-direction
"""
abstract type AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz} <: AbstractBox{Mx, My, Mz, Bx, By, Bz} end

"""
    Box2D{Mx, Mz, Hx, Hz, Bx, Bz}

A 2D rectangular box with lateral halo cells.

# Type Parameters
- `Mx`: Number of cells in x-direction
- `Mz`: Number of cells in z-direction
- `Hx`: Number of halo cells in x-direction
- `Hz`: Number of halo cells in z-direction
- `Bx`: Boundary topology in x-direction
- `Bz`: Boundary topology in z-direction

# Constructor
    Box2D(; Mx, Mz, Hx=1, Hz=1, boundary=(Periodic, Bounded))

Create a 2D box with specified dimensions and boundary conditions.
"""
struct Box2D{Mx, Mz, Hx, Hz, Bx, Bz} <: AbstractBoxHalo{Mx, 0, Mz, Hx, 0, Hz, Bx, Nothing, Bz} end
Box2D(; Mx, Mz, Hx=1, Hz=1, boundary=(Periodic, Bounded)) = Box2D{Mx, Mz, Hx, Hz, boundary[1], boundary[2]}()

"""
    Box3D{Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz}

A 3D rectangular box with halo cells on all sides.

# Type Parameters
- `Mx`: Number of cells in x-direction
- `My`: Number of cells in y-direction
- `Mz`: Number of cells in z-direction
- `Hx`: Number of halo cells in x-direction
- `Hy`: Number of halo cells in y-direction
- `Hz`: Number of halo cells in z-direction
- `Bx`: Boundary topology in x-direction
- `By`: Boundary topology in y-direction
- `Bz`: Boundary topology in z-direction

# Constructor
    Box3D(; Mx, My, Mz, Hx=1, Hy=1, Hz=1, boundary=(Periodic, Periodic, Bounded))

Create a 3D box with specified dimensions and boundary conditions.
"""
struct Box3D{Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz} <: AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz} end
Box3D(; Mx, My, Mz, Hx=1, Hy=1, Hz=1, boundary=(Periodic, Periodic, Bounded)) = Box3D{Mx, My, Mz, Hx, Hy, Hz, boundary[1], boundary[2], boundary[3]}()

"""
    dims(box::AbstractBox) -> NTuple

Get the interior dimensions (number of cells) of the box excluding halo cells.
"""
@inline dims(::Box2D{Mx, Mz}) where {Mx, Mz} = (Mx, Mz)
@inline dims(::Box3D{Mx, My, Mz}) where {Mx, My, Mz} = (Mx, My, Mz)

"""
    halo_size(box::AbstractBox) -> NTuple

Get the size of halo cells in each direction.
"""
@inline halo_size(::Box2D{Mx, Mz, Hx, Hz}) where {Mx, Mz, Hx, Hz} = (Hx, Hz)
@inline halo_size(::Box3D{Mx, My, Mz, Hx, Hy, Hz}) where {Mx, My, Mz, Hx, Hy, Hz} = (Hx, Hy, Hz)

"""
    boundary_topology(box::AbstractBox) -> NTuple

Get the boundary topology of the box in each direction.
"""
@inline boundary_topology(::Box2D{Mx, Mz, Hx, Hz, Bx, Bz}) where {Mx, Mz, Hx, Hz, Bx, Bz} = (Bx(), Bz())
@inline boundary_topology(::Box3D{Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz}) where {Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz} = (Bx(), By(), Bz())

#####
##### Grid along one dimension d (1 = x, 2 = z in 2D; 1, 2, 3 = x, y, z in 3D): M interior cells, H halo cells on
##### each side; a bounded dimension has one more face than cells (both wall faces), a periodic one as many
#####

"""
    Face(), Cell()

Staggering along one dimension: on the faces, or on the cell centres. A tuple of them in dimension order is a grid
location, e.g. `(Face(), Cell())` for the u-faces of a 2D box (see `alloc`); the interpolation `ℑ` and difference `δ`
take one per direction for the location of their result.
"""
struct Face end
struct Cell end

# faces beyond the cell count: both wall faces of a bounded dimension
@inline extra_face(::Periodic) = 0
@inline extra_face(::Bounded) = 1

"""
    ncells(box, d) -> Int
    nfaces(box, d) -> Int

Number of cells or faces along dimension `d`, halo included.
"""
@inline ncells(box::AbstractBox, d) = dims(box)[d] + 2halo_size(box)[d]
@inline nfaces(box::AbstractBox, d) = ncells(box, d) + extra_face(boundary_topology(box)[d])

"""
    cells(box, d) -> UnitRange
    faces(box, d) -> UnitRange

Indices along dimension `d`, halo included, where the stencils onto cells (`ℑ`, `δ` with `Cell()` along d) or onto faces
(with `Face()`) can be evaluated: the loop ranges of the kernels.
"""
@inline cells(box::AbstractBox, d) = 1:nfaces(box, d)-1
@inline faces(box::AbstractBox, d) = 2:ncells(box, d)

"""
    interior_cells(box, d) -> UnitRange
    interior_faces(box, d) -> UnitRange

Indices of the interior cells or faces along dimension `d`. When `d` is bounded, the first and last interior faces
are the walls.
"""
@inline interior_cells(box::AbstractBox, d) = halo_size(box)[d]+1 : halo_size(box)[d]+dims(box)[d]
@inline interior_faces(box::AbstractBox, d) = halo_size(box)[d]+1 : halo_size(box)[d]+dims(box)[d]+extra_face(boundary_topology(box)[d])

"""
    cellpoint(box, Δ, d, i) -> Float64
    facepoint(box, Δ, d, i) -> Float64

Coordinate of cell centre `i` or face `i` along dimension `d`, with grid spacing `Δ`; the first wall (or first
interior face) is at 0.
"""
@inline cellpoint(box::AbstractBox, Δ, d, i) = (i - halo_size(box)[d] - 0.5) * Δ
@inline facepoint(box::AbstractBox, Δ, d, i) = (i - halo_size(box)[d] - 1) * Δ

"""
    cellgrid(box, Δ, d) -> Vector
    facegrid(box, Δ, d) -> Vector

Coordinates of all cell centres or faces along dimension `d`, halo included.
"""
cellgrid(box::AbstractBox, Δ, d) = [cellpoint(box, Δ, d, i) for i in 1:ncells(box, d)]
facegrid(box::AbstractBox, Δ, d) = [facepoint(box, Δ, d, i) for i in 1:nfaces(box, d)]

"""
    interior(box::AbstractBox, data::AbstractArray) -> SubArray

View of the interior region of a data array, without its halo cells, on any grid location (e.g. on the faces of a
bounded direction, the interior includes both wall faces).
"""
interior(::Box2D{Mx, Mz, Hx, Hz}, data::AbstractArray) where {Mx, Mz, Hx, Hz} = @view data[1+Hx:end-Hx, 1+Hz:end-Hz]
interior(::Box3D{Mx, My, Mz, Hx, Hy, Hz}, data::AbstractArray) where {Mx, My, Mz, Hx, Hy, Hz} = @view data[1+Hx:end-Hx, 1+Hy:end-Hy, 1+Hz:end-Hz]

"""
    add_halo(box::AbstractBox, data::Array) -> Array

Add a halo around a data array.
"""
function add_halo(::Box2D{Mx, Mz, Hx, Hz}, data::Array{T, 2}) where {Mx, Mz, Hx, Hz, T}
    Marr, Narr, = size(data)
    data_halo = Array{T}(undef, Marr+2Hx, Narr+2Hz)
    data_halo[1+Hx:Marr+Hx, 1+Hz:Narr+Hz] = data
    return data_halo
end
function add_halo(::Box3D{Mx, My, Mz, Hx, Hy, Hz}, data::Array{T, 3}) where {Mx, My, Mz, Hx, Hy, Hz, T}
    Marr, Narr, Parr = size(data)
    data_halo = Array{T}(undef, Marr+2Hx, Narr+2Hy, Parr+2Hz)
    data_halo[1+Hx:Marr+Hx, 1+Hy:Narr+Hy, 1+Hz:Parr+Hz] = data
    return data_halo
end

"""
    npoints(box, Face(), d) -> Int
    npoints(box, Cell(), d) -> Int

Number of points along dimension `d` of an array on the faces or on the cell centres, halo included.
"""
@inline npoints(box::AbstractBox, ::Face, d) = nfaces(box, d)
@inline npoints(box::AbstractBox, ::Cell, d) = ncells(box, d)

# the dimension numbers 1, 2, … of a location tuple
@inline dimensions(location::Tuple) = ntuple(identity, Val(length(location)))

"""
    coordinates(box::AbstractBox, spacing, location::Tuple) -> Tuple

Coordinates of the interior points of the grid `location` (one `Face()`/`Cell()` per dimension, see `alloc`), one
vector per dimension, matching `interior(box, data)` for data on that location. `spacing` holds the grid spacing in
each dimension, e.g. `(dx, dz)` in 2D.
"""
coordinates(box::AbstractBox, spacing, location::Tuple) =
    map((s, d) -> interior_grid(box, spacing[d], s, d), location, dimensions(location))

# interior coordinates along dimension d of an array on the faces or on the cell centres
interior_grid(box, Δ, ::Face, d) = facegrid(box, Δ, d)[interior_faces(box, d)]
interior_grid(box, Δ, ::Cell, d) = cellgrid(box, Δ, d)[interior_cells(box, d)]

"""
    alloc(F::Type, box::AbstractBox, location::Tuple, dims...) -> Array

Allocate a zero array on the grid `location` of the box: one `Face()` or `Cell()` per dimension, in dimension order,
e.g. `(Cell(), Cell())` cells, `(Face(), Cell())` u-faces, `(Cell(), Face())` w-faces, `(Face(), Face())` vertices in
2D. Trailing `dims` are extra dimensions. Arrays start at zero, so that halo points no kernel writes (e.g. beyond
walls) are defined.
"""
alloc(F::Type, box::AbstractBox, location::Tuple, dims...) =
    zeros(F, map((s, d) -> npoints(box, s, d), location, dimensions(location))..., dims...)

"""
    a = alloc!(a, F::Type, box::AbstractBox, location::Tuple, dims...)
    a = alloc!(void, F::Type, box::AbstractBox, location::Tuple, dims...)

Return the array `a` unchanged, or a new zero array on the grid `location` (see `alloc`) when `a` is not an array
(e.g. MutatingOrNot `void`), as `similar!` does for scratch space.
"""
alloc!(a::AbstractArray, F::Type, box::AbstractBox, location::Tuple, dims...) = a
alloc!(_, F::Type, box::AbstractBox, location::Tuple, dims...) = alloc(F, box, location, dims...)

#####
##### Indices across one dimension (dimension d known at compile time)
#####

"""
    slice(a, ::Val{d}) -> CartesianIndices
    across(i_slice, ::Val{d}, k) -> CartesianIndex

`slice` gives the indices of `a` with dimension `d` removed (one `i_slice` per line of points along `d`, e.g. per
position along a wall normal to `d`); `across` puts the index `k` back at dimension `d`.
"""
@inline slice(a::AbstractArray{T, N}, ::Val{d}) where {T, N, d} =
    CartesianIndices((ntuple(e -> axes(a, e), Val(d-1))..., ntuple(e -> axes(a, d+e), Val(N-d))...))
@inline across(i_slice::CartesianIndex{N}, ::Val{d}, k) where {N, d} =
    CartesianIndex(ntuple(e -> i_slice[e], Val(d-1))..., k, ntuple(e -> i_slice[d-1+e], Val(N-d+1))...)

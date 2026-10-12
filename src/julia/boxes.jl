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
    xdim(box::AbstractBox) -> NTuple
    ydim(box::AbstractBox) -> NTuple
    zdim(box::AbstractBox) -> NTuple

Get the dimensions (number of cells) of the primal grid of the box including halo cells.
"""
@inline xdim(::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz}) where {Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz} = 2Hx+Mx
@inline ydim(::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz}) where {Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz} = 2Hy+My
@inline zdim(::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz}) where {Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz} = 2Hz+Mz

"""
    Xdim(box::AbstractBox) -> NTuple
    Ydim(box::AbstractBox) -> NTuple
    Zdim(box::AbstractBox) -> NTuple

Get the dimensions (number of cells) of the dual grid of the box including halo cells.
"""
@inline Xdim(::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz, Periodic, By, Bz}) where {Mx, My, Mz, Hx, Hy, Hz, By, Bz} = 2Hx+Mx
@inline Xdim(::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz, Bounded, By, Bz}) where {Mx, My, Mz, Hx, Hy, Hz, By, Bz} = 2Hx+Mx+1
@inline Ydim(::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz, Bx, Periodic, Bz}) where {Mx, My, Mz, Hx, Hy, Hz, Bx, Bz} = 2Hy+My
@inline Ydim(::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz, Bx, Bounded, Bz}) where {Mx, My, Mz, Hx, Hy, Hz, Bx, Bz} = 2Hy+My+1
@inline Zdim(::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz, Bx, By, Periodic}) where {Mx, My, Mz, Hx, Hy, Hz, Bx, By} = 2Hz+Mz
@inline Zdim(::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bounded}) where {Mx, My, Mz, Hx, Hy, Hz, Bx, By} = 2Hz+Mz+1

"""
    xrange(box::AbstractBoxHalo) -> UnitRange
    yrange(box::AbstractBoxHalo) -> UnitRange
    zrange(box::AbstractBoxHalo) -> UnitRange

Get the index range to use with centred finite difference operators onto the primal grid in the x/y/z-direction
"""
@inline xrange(box::AbstractBoxHalo) = 1:Xdim(box)-1
@inline yrange(box::AbstractBoxHalo) = 1:Ydim(box)-1
@inline zrange(box::AbstractBoxHalo) = 1:Zdim(box)-1

"""
    Xrange(box::AbstractBoxHalo) -> UnitRange
    Yrange(box::AbstractBoxHalo) -> UnitRange
    Zrange(box::AbstractBoxHalo) -> UnitRange

Get the index range to use with centred finite difference operators onto the dual grid in the x/y/z-direction
"""
@inline Xrange(box::AbstractBoxHalo) = 2:xdim(box)
@inline Yrange(box::AbstractBoxHalo) = 2:ydim(box)
@inline Zrange(box::AbstractBoxHalo) = 2:zdim(box)

"""
    xrange_interior(box::AbstractBoxHalo) -> UnitRange
    yrange_interior(box::AbstractBoxHalo) -> UnitRange
    zrange_interior(box::AbstractBoxHalo) -> UnitRange

Get the index range of the grid interior (excl. halo) in the x/y/z-direction
"""
@inline xrange_interior(box::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz}) where {Mx, My, Mz, Hx, Hy, Hz} = (Hx+1):(xdim(box)-Hx)
@inline yrange_interior(box::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz}) where {Mx, My, Mz, Hx, Hy, Hz} = (Hy+1):(ydim(box)-Hy)
@inline zrange_interior(box::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz}) where {Mx, My, Mz, Hx, Hy, Hz} = (Hz+1):(zdim(box)-Hz)

"""
    Xrange_interior(box::AbstractBoxHalo) -> UnitRange
    Yrange_interior(box::AbstractBoxHalo) -> UnitRange
    Zrange_interior(box::AbstractBoxHalo) -> UnitRange

Get the index range of the dual grid interior (excl. halo) in the x/y/z-direction
"""
@inline Xrange_interior(box::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz}) where {Mx, My, Mz, Hx, Hy, Hz} = (Hx+1):(Xdim(box)-Hx)
@inline Yrange_interior(box::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz}) where {Mx, My, Mz, Hx, Hy, Hz} = (Hy+1):(Ydim(box)-Hy)
@inline Zrange_interior(box::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz}) where {Mx, My, Mz, Hx, Hy, Hz} = (Hz+1):(Zdim(box)-Hz)

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
    interior(box::AbstractBox, data::AbstractArray) -> SubArray

View of the interior region of a data array, without its halo cells, on any grid location (e.g. on the dual grid of a
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
    boundary_topology(box::AbstractBox) -> NTuple

Get the boundary topology of the box in each direction.
"""
@inline boundary_topology(::Box2D{Mx, Mz, Hx, Hz, Bx, Bz}) where {Mx, Mz, Hx, Hz, Bx, Bz} = (Bx(), Bz())
@inline boundary_topology(::Box3D{Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz}) where {Mx, My, Mz, Hx, Hy, Hz, Bx, By, Bz} = (Bx(), By(), Bz())

"""
    xpoint(box::AbstractBoxHalo, dx, i::Int) -> Float64
    ypoint(box::AbstractBoxHalo, dy, j::Int) -> Float64
    zpoint(box::AbstractBoxHalo, dz, k::Int) -> Float64

Get the coordinate of a point on the primal grid in the x/y/z-direction.
"""
@inline xpoint(::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz}, dx, i::Int) where {Mx, My, Mz, Hx, Hy, Hz} = (i - Hx - 0.5) * dx
@inline ypoint(::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz}, dy, j::Int) where {Mx, My, Mz, Hx, Hy, Hz} = (j - Hy - 0.5) * dy
@inline zpoint(::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz}, dz, k::Int) where {Mx, My, Mz, Hx, Hy, Hz} = (k - Hz - 0.5) * dz

"""
    Xpoint(box::AbstractBoxHalo, dx, I::Int) -> Float64
    Ypoint(box::AbstractBoxHalo, dy, J::Int) -> Float64
    Zpoint(box::AbstractBoxHalo, dz, K::Int) -> Float64

Get the coordinate of a point on the dual grid in the x/y/z-direction.
"""
@inline Xpoint(::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz}, dx, I::Int) where {Mx, My, Mz, Hx, Hy, Hz} = (I - Hx - 1.) * dx
@inline Ypoint(::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz}, dy, J::Int) where {Mx, My, Mz, Hx, Hy, Hz} = (J - Hy - 1.) * dy
@inline Zpoint(::AbstractBoxHalo{Mx, My, Mz, Hx, Hy, Hz}, dz, K::Int) where {Mx, My, Mz, Hx, Hy, Hz} = (K - Hz - 1.) * dz

"""
    xgrid(box::AbstractBoxHalo, dx) -> Vector
    ygrid(box::AbstractBoxHalo, dy) -> Vector
    zgrid(box::AbstractBoxHalo, dz) -> Vector

Get the primal grid vectors of the box (incl. halo cells) in the x/y/z-direction.
"""
@inline xgrid(box::AbstractBoxHalo, dx) = [xpoint(box, dx, i) for i in 1:xdim(box)]
@inline ygrid(box::AbstractBoxHalo, dy) = [ypoint(box, dy, j) for j in 1:ydim(box)]
@inline zgrid(box::AbstractBoxHalo, dz) = [zpoint(box, dz, k) for k in 1:zdim(box)]

"""
    Xgrid(box::AbstractBoxHalo, dx) -> Vector
    Ygrid(box::AbstractBoxHalo, dy) -> Vector
    Zgrid(box::AbstractBoxHalo, dz) -> Vector

Get the dual grid vectors of the box (incl. halo cells) in the x/y/z-direction.
"""
@inline Xgrid(box::AbstractBoxHalo, dx) = [Xpoint(box, dx, I) for I in 1:Xdim(box)]
@inline Ygrid(box::AbstractBoxHalo, dy) = [Ypoint(box, dy, J) for J in 1:Ydim(box)]
@inline Zgrid(box::AbstractBoxHalo, dz) = [Zpoint(box, dz, K) for K in 1:Zdim(box)]

"""
    xgrid_interior(box::AbstractBox, dx) -> Vector
    ygrid_interior(box::AbstractBox, dy) -> Vector
    zgrid_interior(box::AbstractBox, dz) -> Vector

Get the primal grid vectors of the box interior (excl. halo) in the x/y/z-direction.
"""
@inline xgrid_interior(box, dx) = xgrid(box, dx)[xrange_interior(box)]
@inline ygrid_interior(box, dy) = ygrid(box, dy)[yrange_interior(box)]
@inline zgrid_interior(box, dz) = zgrid(box, dz)[zrange_interior(box)]

"""
    Xgrid_interior(box::AbstractBox, dx) -> Vector
    Ygrid_interior(box::AbstractBox, dy) -> Vector
    Zgrid_interior(box::AbstractBox, dz) -> Vector

Get the dual grid vectors of the box interior (excl. halo) in the x/y/z-direction.
"""
@inline Xgrid_interior(box, dx) = Xgrid(box, dx)[Xrange_interior(box)]
@inline Ygrid_interior(box, dy) = Ygrid(box, dy)[Yrange_interior(box)]
@inline Zgrid_interior(box, dz) = Zgrid(box, dz)[Zrange_interior(box)]

"""
    coordinates(box::AbstractBox, spacing, location::Symbol) -> Tuple

Coordinates of the interior points of the grid `location` (e.g. `:xz`, `:Xz`; see `alloc`), one vector per dimension,
matching `interior(box, data)` for data on that location. `spacing` holds the grid spacing in each dimension, e.g.
`(dx, dz)` in 2D.
"""
coordinates(box::AbstractBox, spacing, location::Symbol) =
    map((letter, d) -> grid_interior(box, d, letter), Tuple(String(location)), Tuple(spacing))

grid_interior(box, d, letter::Char) =
    letter == 'x' ? xgrid_interior(box, d) : letter == 'X' ? Xgrid_interior(box, d) :
    letter == 'y' ? ygrid_interior(box, d) : letter == 'Y' ? Ygrid_interior(box, d) :
    letter == 'z' ? zgrid_interior(box, d) : letter == 'Z' ? Zgrid_interior(box, d) :
    throw(ArgumentError("unknown grid letter $letter"))

"""
    alloc_xz(F::Type, box::Box2D, dims...) -> Array
    alloc_Xz(F::Type, box::Box2D, dims...) -> Array
    alloc_xZ(F::Type, box::Box2D, dims...) -> Array
    alloc_XZ(F::Type, box::Box2D, dims...) -> Array

- `alloc_xz` - allocate primal grid in x and z
- `alloc_Xz` - allocate dual grid in x, primal grid in z
- `alloc_xZ` - allocate primal grid in x, dual grid in z
- `alloc_XZ` - allocate dual grid in x and z
"""
@inline alloc_xz(F::Type, box::Box2D, dims...) = Array{F}(undef, xdim(box), zdim(box), dims...)
@inline alloc_Xz(F::Type, box::Box2D, dims...) = Array{F}(undef, Xdim(box), zdim(box), dims...)
@inline alloc_xZ(F::Type, box::Box2D, dims...) = Array{F}(undef, xdim(box), Zdim(box), dims...)
@inline alloc_XZ(F::Type, box::Box2D, dims...) = Array{F}(undef, Xdim(box), Zdim(box), dims...)

"""
    alloc_xyz(F::Type, box::Box2D, dims...) -> Array
    alloc_Xyz(F::Type, box::Box2D, dims...) -> Array
    alloc_xYz(F::Type, box::Box2D, dims...) -> Array
    alloc_xyZ(F::Type, box::Box2D, dims...) -> Array
    alloc_XYz(F::Type, box::Box2D, dims...) -> Array
    alloc_xYZ(F::Type, box::Box2D, dims...) -> Array
    alloc_XyZ(F::Type, box::Box2D, dims...) -> Array
    alloc_xYZ(F::Type, box::Box2D, dims...) -> Array
    alloc_XYZ(F::Type, box::Box2D, dims...) -> Array

- `alloc_xyz` - allocate primal grid in x, y, z
- `alloc_Xyz` - allocate dual grid in x, primal grid in y, z
- `alloc_xYz` - allocate primal grid in x, z, dual grid in y
- `alloc_xyZ` - allocate primal grid in x, y, dual grid in z
- `alloc_XYz` - allocate dual grid in x, y, primal grid in z
- `alloc_xYZ` - allocate primal grid in x, dual grid in y, z
- `alloc_XyZ` - allocate primal grid in y, dual grid in x, z
- `alloc_xYZ` - allocate primal grid in x, dual grid in y and z
- `alloc_XYZ` - allocate dual grid in x, y, z
"""
@inline alloc_xyz(F::Type, box::Box3D, dims...) = Array{F}(undef, xdim(box), ydim(box), zdim(box), dims...)
@inline alloc_Xyz(F::Type, box::Box3D, dims...) = Array{F}(undef, Xdim(box), ydim(box), zdim(box), dims...)
@inline alloc_xYz(F::Type, box::Box3D, dims...) = Array{F}(undef, xdim(box), Ydim(box), zdim(box), dims...)
@inline alloc_xyZ(F::Type, box::Box3D, dims...) = Array{F}(undef, xdim(box), ydim(box), Zdim(box), dims...)
@inline alloc_XYz(F::Type, box::Box3D, dims...) = Array{F}(undef, Xdim(box), Ydim(box), zdim(box), dims...)
@inline alloc_XyZ(F::Type, box::Box3D, dims...) = Array{F}(undef, Xdim(box), ydim(box), Zdim(box), dims...)
@inline alloc_xYZ(F::Type, box::Box3D, dims...) = Array{F}(undef, xdim(box), Ydim(box), Zdim(box), dims...)
@inline alloc_XYZ(F::Type, box::Box3D, dims...) = Array{F}(undef, Xdim(box), Ydim(box), Zdim(box), dims...)

"""
    alloc(F::Type, box::AbstractBox, location::Symbol, dims...) -> Array

Allocate an array on the grid `location` of the box: one letter per dimension, lower case for the primal grid and
upper case for the dual grid, e.g. `:xz`, `:Xz`, `:xZ`, `:XZ` in 2D and `:xyz` … `:XYZ` in 3D (see `alloc_xz` etc.).
"""
alloc(F::Type, box::AbstractBox, location::Symbol, dims...) = alloc(F, box, Val(location), dims...)
alloc(F::Type, box::Box2D, ::Val{:xz}, dims...) = alloc_xz(F, box, dims...)
alloc(F::Type, box::Box2D, ::Val{:Xz}, dims...) = alloc_Xz(F, box, dims...)
alloc(F::Type, box::Box2D, ::Val{:xZ}, dims...) = alloc_xZ(F, box, dims...)
alloc(F::Type, box::Box2D, ::Val{:XZ}, dims...) = alloc_XZ(F, box, dims...)
alloc(F::Type, box::Box3D, ::Val{:xyz}, dims...) = alloc_xyz(F, box, dims...)
alloc(F::Type, box::Box3D, ::Val{:Xyz}, dims...) = alloc_Xyz(F, box, dims...)
alloc(F::Type, box::Box3D, ::Val{:xYz}, dims...) = alloc_xYz(F, box, dims...)
alloc(F::Type, box::Box3D, ::Val{:xyZ}, dims...) = alloc_xyZ(F, box, dims...)
alloc(F::Type, box::Box3D, ::Val{:XYz}, dims...) = alloc_XYz(F, box, dims...)
alloc(F::Type, box::Box3D, ::Val{:XyZ}, dims...) = alloc_XyZ(F, box, dims...)
alloc(F::Type, box::Box3D, ::Val{:xYZ}, dims...) = alloc_xYZ(F, box, dims...)
alloc(F::Type, box::Box3D, ::Val{:XYZ}, dims...) = alloc_XYZ(F, box, dims...)
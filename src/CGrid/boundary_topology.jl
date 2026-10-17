"""
    BoundaryTopology

Topology of a box along one dimension: `Periodic` or `Bounded` (by two walls).
"""
abstract type BoundaryTopology end

"""
    Periodic()

The box wraps around along this dimension: the halo copies the interior from the opposite side.
"""
struct Periodic <: BoundaryTopology end

"""
    Bounded()

The box ends at a wall on each side along this dimension; the grid has one more face than cells.
"""
struct Bounded <: BoundaryTopology end

"""
    CFBoxes

Rectangular 2D and 3D boxes for finite-volume models on a C-grid: grids with halos, the interpolation and difference
operators, wall boundary conditions, and schemes for the vortex force ω×u.
"""
module CFBoxes

using ManagedLoops: @vec, @with

# boundary topology
export Periodic, Bounded

# boxes and grids
export AbstractBox, Box2D, Box3D
export dims, halo_size, boundary_topology
export ncells, nfaces, cells, faces, interior_cells, interior_faces
export cellpoint, facepoint, cellgrid, facegrid, coordinates
export Face, Cell, npoints, alloc, alloc!, interior

# wall boundary conditions and halos
export FreeSlip, NoStress, NoSlip, TangentialVelocity, Stress, Flux, HeatFlux, Value, Wall
export BoxBoundaryConditions, discretize, check_tracers, bc_value, wall_sides
export fill_velocity_halo!, fill_halo!, ZeroGradient, periodize!, zero_halo!, zero_walls!

# interpolation and difference
export ℑ, δ

# schemes
export SpatialScheme, Centred2, operators, halo_width, check_halo
export VortexScheme, VortexEnergyCons, VortexEnstrophyCons, VortexEnergyEnstrophyCons, vortex_force!

include("julia/boundary_topology.jl")
include("julia/boxes.jl")
include("julia/walls.jl")
include("julia/operators.jl")
include("julia/spatial_schemes.jl")
include("julia/vortex_schemes.jl")
include("julia/vortex_force.jl")

end

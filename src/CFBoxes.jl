module CFBoxes

using ManagedLoops: @vec, @with

# boundary topology exports
export Periodic, Bounded

# halo exports
export periodize!

# wall boundary condition exports
export FreeSlip, NoStress, NoSlip, TangentialVelocity, Stress, Flux, HeatFlux, Value, Wall, isheat
export BoxBoundaryConditions, discretize, check_tracers, bc_value, fill_velocity_halo!, fill_halo!, ZeroGradient
export zero_halo!, zero_walls!, wall_sides

# box exports
export AbstractBox, AbstractBoxHalo, Box2D, Box3D
export dims, halo_size, boundary_topology
export ncells, nfaces, cells, faces, interior_cells, interior_faces
export cellpoint, facepoint, cellgrid, facegrid, coordinates
export Face, Cell, npoints, alloc, alloc!, interior, add_halo

# interpolation and difference exports
export ℑ, δ

# scheme exports
export SpatialScheme, Centred2, operators, halo_width, check_halo
export VortexScheme, VortexEnergyCons, VortexEnstrophyCons, vortex_force!

include("julia/boundary_topology.jl")
include("julia/boxes.jl")
include("julia/walls.jl")
include("julia/operators.jl")
include("julia/spatial_schemes.jl")
include("julia/vortex_schemes.jl")
include("julia/vortex_force.jl")

end

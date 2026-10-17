"""
    CFBoxes

Rectangular boxes for finite-volume models. Each discretization is a submodule, whose exported names `CFBoxes`
re-exports:
  - [`CGrid`](@ref): 2D and 3D boxes on a C-grid, with halos, operators, wall boundary conditions and vortex-force
    schemes.
"""
module CFBoxes

include("CGrid/CGrid.jl")
using .CGrid

#= unexported names that downstream packages reach as `CFBoxes.name` =#
using .CGrid: slice, across, pick, foreach_dim, face_densities!

for name in names(CGrid)
    name === :CGrid || @eval export $name
end

end

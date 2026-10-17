# Tests of CFBoxes.CGrid, on its own names

using CFBoxes.CGrid
using MutatingOrNot: void
using LoopManagers: PlainCPU, VectorizedCPU, MultiThread

# bytes allocated by a call, after a first call that compiles it
allocations(f, args...) = (f(args...); @allocated f(args...))

@testset "CGrid" begin
    include("boxes.jl")
    include("operators.jl")
    include("walls.jl")
    include("vortex_force.jl")
end

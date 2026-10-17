using CFBoxes
using Test

@testset "CFBoxes.jl" begin
    # CFBoxes exports every exported name of each submodule, bound to the same object
    @testset "re-exports" begin
        for submodule in (CFBoxes.CGrid,), name in names(submodule)
            name === nameof(submodule) && continue
            @test Base.isexported(CFBoxes, name) && getfield(CFBoxes, name) === getfield(submodule, name)
        end
    end

    include("CGrid/runtests.jl")
end

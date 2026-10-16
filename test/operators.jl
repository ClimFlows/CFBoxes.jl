#=
Interpolations and differences: exact on linear fields, every destination in 2D and 3D, profiles and products,
summation by parts, no allocations, inside @vec loops; and the SpatialScheme interface
=#

using ManagedLoops: @vec, @with

# bytes allocated by a few operators, products included; typed, as the generic `allocations` would box the Float64 result
some_ops(a, b, v, i, j) = ℑ(a, (1, 2), (Face(), Face()), i, j) + δ((a, b), (1,), (Face(),), i, j) + ℑ((v, a), (2,), (Face(),), i, j)
some_ops_bytes(a, b, v, i, j) = (some_ops(a, b, v, i, j); @allocated some_ops(a, b, v, i, j))

@testset "operators" begin
    # linear field a = 1 + 2i + 3j (+ 5k): interpolations land on the midpoint, differences give the slope
    lin2(i, j) = 1.0 + 2i + 3j
    lin3(i, j, k) = 1.0 + 2i + 3j + 5k
    a2 = [lin2(i, j) for i in 1:6, j in 1:6]
    a3 = [lin3(i, j, k) for i in 1:6, j in 1:6, k in 1:6]
    i, j, k = 3, 3, 3

    @test ℑ(a2, (1,), (Cell(),), i, j) == lin2(i + 0.5, j) && ℑ(a2, (1,), (Face(),), i, j) == lin2(i - 0.5, j)
    @test ℑ(a2, (2,), (Cell(),), i, j) == lin2(i, j + 0.5) && ℑ(a2, (2,), (Face(),), i, j) == lin2(i, j - 0.5)
    @test ℑ(a2, (1, 2), (Cell(), Cell()), i, j) == lin2(i + 0.5, j + 0.5) && ℑ(a2, (1, 2), (Cell(), Face()), i, j) == lin2(i + 0.5, j - 0.5)
    @test ℑ(a2, (1, 2), (Face(), Cell()), i, j) == lin2(i - 0.5, j + 0.5) && ℑ(a2, (1, 2), (Face(), Face()), i, j) == lin2(i - 0.5, j - 0.5)
    @test δ(a2, (1,), (Cell(),), i, j) == δ(a2, (1,), (Face(),), i, j) == 2 && δ(a2, (2,), (Cell(),), i, j) == δ(a2, (2,), (Face(),), i, j) == 3

    #=
    every direction and every pair of directions in 3D, onto either staggering: Face() moves half a cell down, Cell()
    half a cell up
    =#
    step(::Cell) = 0.5
    step(::Face) = -0.5
    shift(d, s) = ntuple(e -> step(s) * (e == d), 3)
    for d in 1:3, s in (Cell(), Face())
        @test ℑ(a3, (d,), (s,), i, j, k) == lin3(((i, j, k) .+ shift(d, s))...)
    end
    for (d, e) in ((1, 2), (1, 3), (2, 3), (2, 1), (3, 1), (3, 2)), s in (Cell(), Face()), t in (Cell(), Face())
        @test ℑ(a3, (d, e), (s, t), i, j, k) == lin3(((i, j, k) .+ shift(d, s) .+ shift(e, t))...)
    end
    @test δ(a3, (1,), (Cell(),), i, j, k) == δ(a3, (1,), (Face(),), i, j, k) == 2
    @test δ(a3, (2,), (Cell(),), i, j, k) == δ(a3, (2,), (Face(),), i, j, k) == 3
    @test δ(a3, (3,), (Cell(),), i, j, k) == δ(a3, (3,), (Face(),), i, j, k) == 5

    # vertical profile: alone (1D) or as a factor read along z; products of factors
    v = [1.0 + 3j for j in 1:6]
    b2 = rand(6, 6)
    @test ℑ(v, (2,), (Cell(),), j) == ℑ(v, (2,), (Cell(),), i, j) == 1 + 3(j + 0.5) && ℑ(v, (2,), (Face(),), j) == 1 + 3(j - 0.5) && δ(v, (2,), (Cell(),), j) == δ(v, (2,), (Face(),), j) == 3
    @test ℑ((v, b2), (2,), (Face(),), i, j) == 0.5 * (v[j] * b2[i, j] + v[j-1] * b2[i, j-1])
    @test ℑ((b2, b2), (1,), (Cell(),), i, j) == 0.5 * (b2[i+1, j]^2 + b2[i, j]^2)
    @test δ((a2, b2), (1,), (Face(),), i, j) == a2[i, j] * b2[i, j] - a2[i-1, j] * b2[i-1, j]

    # summation by parts on a periodic row: Σ_I c δ(f, (1,), (Face(),)) = -Σ_i δ(c, (1,), (Cell(),)) f, with c on faces and f on cells
    n = 8
    c, f = rand(n + 1, 1), rand(n + 1, 1)
    c[1] = c[n+1]; f[n+1] = f[1]                     # one periodic copy each
    @test sum(c[I] * δ(f, (1,), (Face(),), I, 1) for I in 2:n+1) ≈ -sum(δ(c, (1,), (Cell(),), i, 1) * f[i] for i in 1:n)

    # no allocations, products included
    @test some_ops_bytes(a2, b2, v, i, j) == 0

    # inside @vec loops, with a SIMD range as the first index: the same values as plain loops
    box = Box2D(Mx = 16, Mz = 4, boundary = (Periodic, Bounded))
    a, b = rand(ncells(box, 1), ncells(box, 2)), rand(ncells(box, 1), ncells(box, 2))
    out_vec, out_ref = alloc(Float64, nothing, box, (Face(), Cell())), alloc(Float64, nothing, box, (Face(), Cell()))
    function kernel!(out, mgr, box, a, b, v)
        @with mgr, let (irange, jrange) = (faces(box, 1), cells(box, 2))
            @inbounds @vec for I in irange, j in jrange
                out[I, j] = ℑ(a, (1,), (Face(),), I, j) + δ((a, b), (1,), (Face(),), I, j) + ℑ((v, a), (1,), (Face(),), I, j)
            end
        end
        return out
    end
    kernel!(out_vec, VectorizedCPU(), box, a, b, v[1:ncells(box, 2)])
    for I in faces(box, 1), j in cells(box, 2)
        out_ref[I, j] = ℑ(a, (1,), (Face(),), I, j) + δ((a, b), (1,), (Face(),), I, j) + ℑ((v, a), (1,), (Face(),), I, j)
    end
    @test out_vec == out_ref
end

@testset "spatial schemes" begin
    @test operators(Centred2()) == (; ℑ, δ)
    @test halo_width(Centred2()) == 1
    @test check_halo(Box2D(Mx = 4, Mz = 4), Centred2())
    @test_throws ErrorException check_halo(Box2D(Mx = 4, Mz = 4, Hz = 0), Centred2())     # no halo along z
end

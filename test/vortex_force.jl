# Vortex force ω×u: no work done by the energy-conserving scheme (2D and 3D), 3D reduces to 2D, uniform density
# reduces to the density-free formulas, no allocations once the scratch exists

using MutatingOrNot: void
using LoopManagers: PlainCPU

vortex_bytes(out, model, scheme, velocity, ρ) =
    (vortex_force!(out, model, scheme, velocity, ρ); @allocated vortex_force!(out, model, scheme, velocity, ρ))

@testset "vortex force" begin
    dx, dy, dz = 0.3, 0.2, 0.1
    box2 = Box2D(Mx = 8, Mz = 6, boundary = (Periodic, Periodic))
    box3 = Box3D(Mx = 8, My = 4, Mz = 6, boundary = (Periodic, Periodic, Periodic))
    model2 = (mgr = PlainCPU(), domain = box2, spatial_scheme = Centred2(), inv_dx = 1 / dx, inv_dz = 1 / dz)
    model3 = (mgr = PlainCPU(), domain = box3, spatial_scheme = Centred2(), inv_dx = 1 / dx, inv_dy = 1 / dy, inv_dz = 1 / dz)
    random(box, location) = (a = rand(size(alloc(Float64, box, location))...); a .-= 0.5; a)

    # periodic 2D fields, halos filled
    u, w = random(box2, (Face(), Cell())), random(box2, (Cell(), Face()))
    ρ = 1 .+ rand(size(alloc(Float64, box2, (Cell(), Cell())))...)
    periodize!(model2, (u, w, ρ))
    Iu, Ju = interior_faces(box2, 1), interior_cells(box2, 2)
    Iw, Jw = interior_cells(box2, 1), interior_faces(box2, 2)

    @testset "2D, $scheme" for scheme in (VortexEnergyCons(), VortexEnstrophyCons())
        ωw, ωu, scratch = vortex_force!((void, void, void), model2, scheme, (u, w), ρ)
        ρᴵ, ρᴶ = scratch[1], scratch[2]
        # kinetic energy tendency of ∂ₜu = -ωw, ∂ₜw = +ωu, weighted by the face densities
        work = -sum(ρᴵ[Iu, Ju] .* u[Iu, Ju] .* ωw[Iu, Ju]) + sum(ρᴶ[Iw, Jw] .* w[Iw, Jw] .* ωu[Iw, Jw])
        scale = sum(abs, ρᴵ[Iu, Ju] .* u[Iu, Ju] .* ωw[Iu, Ju])
        scheme isa VortexEnergyCons && @test abs(work) < 1e-13 * scale

        # no allocations once the scratch exists
        @test vortex_bytes((ωw, ωu, scratch), model2, scheme, (u, w), ρ) == 0
    end

    # a vertical profile gives the same force as the cell array it stands for (box bounded in z, as for a reference
    # profile)
    @testset "2D, density profile, $scheme" for scheme in (VortexEnergyCons(), VortexEnstrophyCons())
        box = Box2D(Mx = 8, Mz = 6, boundary = (Periodic, Bounded))
        model = (; model2..., domain = box)
        ub, wb = random(box, (Face(), Cell())), random(box, (Cell(), Face()))
        periodize!(model, (ub, wb))
        ρ_z = 1 .+ rand(ncells(box, 2))
        ρ_xz = [ρ_z[j] for i in 1:ncells(box, 1), j in 1:ncells(box, 2)]
        ωw, ωu, scratch = vortex_force!((void, void, void), model, scheme, (ub, wb), ρ_z)
        ωw_xz, ωu_xz, _ = vortex_force!((void, void, void), model, scheme, (ub, wb), ρ_xz)
        x, X, z, Z = interior_cells(box, 1), interior_faces(box, 1), interior_cells(box, 2), interior_faces(box, 2)
        @test ωw[X, z] == ωw_xz[X, z] && ωu[x, Z] == ωu_xz[x, Z]
        @test vortex_bytes((ωw, ωu, scratch), model, scheme, (ub, wb), ρ_z) == 0
    end

    @testset "2D, uniform density" begin
        ρ2 = fill(2.0, size(ρ))
        ωw, ωu, _ = vortex_force!((void, void, void), model2, VortexEnergyCons(), (u, w), ρ2)
        ω(I, J) = (u[I, J] - u[I, J-1]) / dz - (w[I, J] - w[I-1, J]) / dx
        ωwᴵᴶ(I, J) = ω(I, J) * 0.5 * (w[I, J] + w[I-1, J])
        ωuᴵᴶ(I, J) = ω(I, J) * 0.5 * (u[I, J] + u[I, J-1])
        @test all(ωw[I, j] ≈ 0.5 * (ωwᴵᴶ(I, j+1) + ωwᴵᴶ(I, j)) for I in Iu, j in Ju)
        @test all(ωu[i, J] ≈ 0.5 * (ωuᴵᴶ(i+1, J) + ωuᴵᴶ(i, J)) for i in Iw, J in Jw)
    end

    # 3D fields independent of y, with v = 0, reproduce the 2D force: fx = ωw, fy = 0, fz = -ωu
    lift(a) = (b = zeros(size(a, 1), ncells(box3, 2), size(a, 2)); for j in axes(b, 2); b[:, j, :] .= a; end; b)
    u3, w3, ρ3 = lift(u), lift(w), lift(ρ)
    v3 = zeros(size(alloc(Float64, box3, (Cell(), Face(), Cell()))))
    @testset "3D, $scheme" for scheme in (VortexEnergyCons(), VortexEnstrophyCons())
        ωw, ωu, _ = vortex_force!((void, void, void), model2, scheme, (u, w), ρ)
        fx, fy, fz, _ = vortex_force!((void, void, void, void), model3, scheme, (u3, v3, w3), ρ3)
        for j in interior_cells(box3, 2)
            @test fx[Iu, j, Ju] ≈ ωw[Iu, Ju]
            @test fz[Iw, j, Jw] ≈ -ωu[Iw, Jw]
        end
        @test all(iszero, fy[interior_cells(box3, 1), interior_faces(box3, 2), interior_cells(box3, 3)])
    end

    # the energy-conserving 3D force does no work on random fields
    @testset "3D, no work" begin
        u3, v3, w3 = random(box3, (Face(), Cell(), Cell())), random(box3, (Cell(), Face(), Cell())), random(box3, (Cell(), Cell(), Face()))
        ρ3 = 1 .+ rand(size(alloc(Float64, box3, (Cell(), Cell(), Cell())))...)
        periodize!(model3, (u3, v3, w3, ρ3))
        fx, fy, fz, scratch = vortex_force!((void, void, void, void), model3, VortexEnergyCons(), (u3, v3, w3), ρ3)
        ρᴵ, ρᴶ, ρᴷ = scratch[1], scratch[2], scratch[3]
        X, Y, Z = interior_faces(box3, 1), interior_faces(box3, 2), interior_faces(box3, 3)
        x, y, z = interior_cells(box3, 1), interior_cells(box3, 2), interior_cells(box3, 3)
        terms = (ρᴵ[X, y, z] .* u3[X, y, z] .* fx[X, y, z], ρᴶ[x, Y, z] .* v3[x, Y, z] .* fy[x, Y, z],
                 ρᴷ[x, y, Z] .* w3[x, y, Z] .* fz[x, y, Z])
        @test abs(sum(sum, terms)) < 1e-13 * sum(t -> sum(abs, t), terms)

        @test vortex_bytes((fx, fy, fz, scratch), model3, VortexEnergyCons(), (u3, v3, w3), ρ3) == 0
    end
end

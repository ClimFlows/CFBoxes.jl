# Wall boundary conditions: construction checks, discretization, ghost velocities, and the halo functions, in 2D and 3D

wall(tangential, n = 2) = Wall(; tangential, tracers = ntuple(_ -> Flux(0.0), n))

# bytes allocated by fill_velocity_halo!, with fixed arguments: on Julia 1.10 the generic `allocations` boxes them
velocity_halo_bytes(bcs, box, velocity, spacing, μ) =
    (fill_velocity_halo!(bcs, box, velocity, spacing, μ); @allocated fill_velocity_halo!(bcs, box, velocity, spacing, μ))

@testset "walls" begin
    # walls only on bounded sides, with the right number of tracer conditions, and HeatFlux only for the first tracer
    @testset "construction checks" begin
        box = Box2D(Mx = 4, Mz = 3, boundary = (Periodic, Bounded))
        @test_throws ErrorException BoxBoundaryConditions(box; bottom = wall(NoSlip()))                      # top missing
        @test_throws ErrorException BoxBoundaryConditions(box; left = wall(NoSlip()), right = wall(NoSlip()),
                                                          bottom = wall(NoSlip()), top = wall(NoSlip()))     # x is periodic
        @test_throws ErrorException check_tracers(BoxBoundaryConditions(box; bottom = wall(NoSlip()), top = wall(FreeSlip(), 3)), 2)
        @test isnothing(check_tracers(BoxBoundaryConditions(box; bottom = wall(NoSlip()), top = wall(NoSlip())), 2))
        heat_in(n) = Wall(tangential = NoSlip(), tracers = n == 1 ? (HeatFlux(1.0), Flux(0.0)) : (Flux(0.0), HeatFlux(1.0)))
        @test isnothing(check_tracers(BoxBoundaryConditions(box; bottom = heat_in(1), top = wall(NoSlip())), 2))
        @test_throws ErrorException check_tracers(BoxBoundaryConditions(box; bottom = heat_in(2), top = wall(NoSlip())), 2)
        # a HeatFlux stays a HeatFlux when discretized; NoStress is FreeSlip
        bcs = discretize(BoxBoundaryConditions(box; bottom = heat_in(1), top = wall(NoSlip())), box, (0.1, 0.1))
        @test CGrid.isheat(bcs.walls[2][1].tracers[1]) && !CGrid.isheat(bcs.walls[2][1].tracers[2])
        @test NoStress() === FreeSlip()
    end

    @testset "2D" begin
        box = Box2D(Mx = 4, Mz = 3, boundary = (Bounded, Bounded))       # u: 7×5, w: 6×6, cells: 6×5
        dx, dz, μ = 0.25, 0.5, 1e-3
        U(X) = 2X
        W(Z) = 3Z
        bcs = discretize(BoxBoundaryConditions(box;
            left   = Wall(tangential = TangentialVelocity(W), tracers = (Value(z -> z), Flux(1.0))),
            right  = wall(FreeSlip()),
            bottom = Wall(tangential = TangentialVelocity(U), tracers = (Value(x -> x), Flux(0.0))),
            top    = wall(Stress(0.1))), box, (dx, dz))

        #=
        tracer conditions sampled at the cell centers along the wall, numbers kept as numbers;
        the left wall's tangential velocity sampled at the w-faces
        =#
        @test bc_value(bcs.walls[2][1].tracers[1], 3) ≈ cellpoint(box, dx, 1, 3)
        @test bc_value(bcs.walls[1][1].tracers[1], 4) ≈ cellpoint(box, dz, 2, 4)
        @test bc_value(bcs.walls[1][1].tracers[2], 4) == 1.0
        @test bcs.walls[1][1].tangential.value[2] ≈ W.(facegrid(box, dz, 2))

        u, w = rand(7, 5), rand(6, 6)
        u0, w0 = copy(u), copy(w)
        @test fill_velocity_halo!(bcs, box, (u, w), (dx, dz), μ) === (u, w)
        # interior unchanged, except the wall faces, where the normal velocity is set to zero
        @test u[3:5, 2:4] == u0[3:5, 2:4] && w[2:5, 3:4] == w0[2:5, 3:4]
        @test all(iszero, u[[2, 6], 2:4]) && all(iszero, w[:, [2, 5]])
        @test u[1, 2:4] == -u[3, 2:4] && w[:, 1] == -w[:, 3]                           # normal velocity is odd
        @test (w[1, 3:4] .+ w[2, 3:4]) ./ 2 ≈ W.(facepoint.(Ref(box), dz, 2, 3:4))      # moving left wall
        @test w[6, 3:5] == w[5, 3:5]                                                  # free slip: ∂w/∂x = 0 on the right wall
        @test (u[:, 1] .+ u[:, 2]) ./ 2 ≈ U.(facepoint.(Ref(box), dx, 1, 1:7))         # moving bottom wall
        @test μ .* (u[:, 5] .- u[:, 4]) ./ dz ≈ fill(0.1, 7)                         # stress on the top wall
        @test velocity_halo_bytes(bcs, box, (u, w), (dx, dz), μ) == 0
    end

    # with two halo cells, the second ghost mirrors the second fluid point, at distance 3Δ (cells) or 4Δ (faces)
    @testset "2D, two halo cells" begin
        box = Box2D(Mx = 4, Mz = 3, Hx = 2, Hz = 2, boundary = (Periodic, Bounded))   # u: 8×7, w: 8×8
        dx, dz, μ = 0.25, 0.5, 1e-3
        bcs = discretize(BoxBoundaryConditions(box; bottom = wall(NoSlip()), top = wall(Stress(0.1))), box, (dx, dz))
        u, w = rand(8, 7), rand(8, 8)
        fill_velocity_halo!(bcs, box, (u, w), (dx, dz), μ)
        @test u[:, 1] == -u[:, 4] && u[:, 2] == -u[:, 3]                              # no slip, mirrored about the wall
        @test μ .* (u[:, 7] .- u[:, 4]) ./ (3dz) ≈ fill(0.1, 8)                       # stress, across 3Δz
        @test w[:, 1] == -w[:, 5] && w[:, 8] == -w[:, 4]                              # normal velocity, across 4Δz
        @test u[1:2, :] == u[5:6, :] && u[7:8, :] == u[3:4, :]                        # periodic in x
    end

    @testset "3D" begin
        box = Box3D(Mx = 4, My = 3, Mz = 2, boundary = (Periodic, Bounded, Bounded))  # u: 6×5×4, v: 6×6×4, w: 6×5×5
        Δ, μ = (0.25, 0.3, 0.5), 1e-3
        bcs = discretize(BoxBoundaryConditions(box; front = wall(NoSlip()), back = wall(FreeSlip()),
            bottom = wall(TangentialVelocity((1.0, (x, y) -> y))), top = wall(Stress((0.0, 0.1)))), box, Δ)
        u, v, w = rand(6, 5, 4), rand(6, 6, 4), rand(6, 5, 5)
        fill_velocity_halo!(bcs, box, (u, v, w), Δ, μ)
        @test u[1, :, :] == u[5, :, :] && u[6, :, :] == u[2, :, :]                    # periodic in x
        @test all(iszero, v[:, [2, 5], 2:3]) && all(iszero, w[:, 2:4, [2, 4]])       # normal velocities on the walls
        @test u[:, 1, 2:3] == -u[:, 2, 2:3]                                          # no slip on the front wall
        @test (u[2:5, 2:4, 1] .+ u[2:5, 2:4, 2]) ./ 2 ≈ ones(4, 3)                     # bottom moves at (1, y)
        @test (v[2:5, 2:5, 1] .+ v[2:5, 2:5, 2]) ./ 2 ≈ [facepoint(box, Δ[2], 2, J) for i in 2:5, J in 2:5]
        @test μ .* (v[2:5, 2:5, 4] .- v[2:5, 2:5, 3]) ./ Δ[3] ≈ fill(0.1, 4, 4)        # stress on the top wall
        @test velocity_halo_bytes(bcs, box, (u, v, w), Δ, μ) == 0
    end

    # zero-gradient halos: periodic copies along x, mirrored cells across the walls; trailing dimensions allowed
    @testset "zero-gradient halos" begin
        box = Box3D(Mx = 4, My = 3, Mz = 2, boundary = (Periodic, Bounded, Bounded))
        a = rand(6, 5, 4, 2)
        @test fill_halo!(box, a, ZeroGradient()) === a
        @test a[1, :, :, :] == a[5, :, :, :] && a[6, :, :, :] == a[2, :, :, :]
        @test a[:, 1, :, :] == a[:, 2, :, :] && a[:, 5, :, :] == a[:, 4, :, :]
        @test a[:, :, 1, :] == a[:, :, 2, :] && a[:, :, 4, :] == a[:, :, 3, :]
        b = rand(6, 5, 4)
        @test fill_halo!(box, (a, b), ZeroGradient()) === (a, b) && b[:, 1, :] == b[:, 2, :]
        @test allocations(fill_halo!, box, (a, b), ZeroGradient()) == 0
    end

    # periodic copies along periodic dimensions only, at any grid location
    @testset "periodize!" begin
        box = Box2D(Mx = 4, Mz = 3, boundary = (Periodic, Bounded))       # u: 6×5, w: 6×6
        u, w = rand(6, 5), rand(6, 6)
        w0 = copy(w)
        @test periodize!(box, (u, w)) === (u, w)
        @test u[1, :] == u[5, :] && u[6, :] == u[2, :] && w[1, :] == w[5, :] && w[6, :] == w[2, :]
        @test w[2:5, :] == w0[2:5, :]                                       # bounded z: untouched
        @test periodize!(box, u) === u
        @test allocations(periodize!, box, (u, w)) == 0
    end

    # zeroed halos, and walls of the face arrays normal to them
    @testset "zero_halo! and zero_walls!" begin
        box = Box2D(Mx = 4, Mz = 3, boundary = (Periodic, Bounded))        # cells 6×5, u 6×5, w 6×6
        w = ones(6, 6)
        @test zero_halo!(box, w, (Cell(), Face())) === w
        @test all(iszero, w[[1, 6], :]) && all(iszero, w[:, [1, 2, 5, 6]]) && all(isone, w[2:5, 3:4])
        u = ones(6, 5)
        zero_halo!(box, u, (Face(), Cell()))                           # x periodic: no wall faces
        @test all(iszero, u[[1, 6], :]) && all(iszero, u[:, [1, 5]]) && all(isone, u[2:5, 2:4])
        J = ones(6, 6)
        @test zero_walls!(box, J, Val(2)) === J
        @test all(iszero, J[:, [2, 5]]) && count(iszero, J) == 12
        @test zero_walls!(box, ones(6, 5), Val(1)) == ones(6, 5)          # periodic dimension: untouched
        @test allocations(zero_halo!, box, w, (Cell(), Face())) == 0
    end

    # wall faces, adjacent cells and inward normals
    @testset "wall sides" begin
        box = Box2D(Mx = 4, Mz = 3, Hx = 1, Hz = 1, boundary = (Periodic, Bounded))
        lower, upper = wall_sides(box, Val(2))
        @test lower == (face = 2, cell = 2, n = 1) && upper == (face = 5, cell = 4, n = -1)
    end
end

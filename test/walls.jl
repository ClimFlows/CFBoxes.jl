# Wall boundary conditions: construction checks, discretization, ghost velocities and mirrored halos, in 2D and 3D

allocations(f, args...) = (f(args...); @allocated f(args...))
wall(tangential, n = 2) = Wall(; tangential, tracers = ntuple(_ -> Flux(0.0), n))

@testset "walls" begin
    @testset "construction checks" begin
        box = Box2D(Mx = 4, Mz = 3, boundary = (Periodic, Bounded))
        @test_throws ErrorException BoxBoundaryConditions(box; bottom = wall(NoSlip()))                      # top missing
        @test_throws ErrorException BoxBoundaryConditions(box; left = wall(NoSlip()), right = wall(NoSlip()),
                                                          bottom = wall(NoSlip()), top = wall(NoSlip()))     # x is periodic
        bcs = BoxBoundaryConditions(box; bottom = wall(NoSlip()), top = wall(FreeSlip(), 3))
        @test_throws ErrorException check_tracers(bcs, 2)
        @test isnothing(check_tracers(BoxBoundaryConditions(box; bottom = wall(NoSlip()), top = wall(NoSlip())), 2))
        heat_in(slot) = Wall(tangential = NoSlip(), tracers = slot == 1 ? (HeatFlux(1.0), Flux(0.0)) : (Flux(0.0), HeatFlux(1.0)))
        @test isnothing(check_tracers(BoxBoundaryConditions(box; bottom = heat_in(1), top = wall(NoSlip())), 2))
        @test_throws ErrorException check_tracers(BoxBoundaryConditions(box; bottom = heat_in(2), top = wall(NoSlip())), 2)
        @test isheat(discretize(BoxBoundaryConditions(box; bottom = heat_in(1), top = wall(NoSlip())), box, (0.1, 0.1)).walls[2][1].tracers[1])
    end

    @testset "2D" begin
        box = Box2D(Mx = 4, Mz = 3, boundary = (Bounded, Bounded))       # u: 7×5, w: 6×6, cells: 6×5
        dx, dz, μ = 0.25, 0.5, 1e-3
        U(X) = 2X
        bcs = discretize(BoxBoundaryConditions(box;
            left   = Wall(tangential = NoSlip(), tracers = (Value(z -> z), Flux(1.0))),
            right  = wall(FreeSlip()),
            bottom = Wall(tangential = TangentialVelocity(U), tracers = (Value(x -> x), Flux(0.0))),
            top    = wall(Stress(0.1))), box, (dx, dz))

        # tracer conditions sampled at the cell centres along the wall; numbers stay numbers
        @test bc_value(bcs.walls[2][1].tracers[1], 3) ≈ xpoint(box, dx, 3)
        @test bc_value(bcs.walls[1][1].tracers[1], 4) ≈ zpoint(box, dz, 4)
        @test bc_value(bcs.walls[1][1].tracers[2], 4) == 1.0

        u, w = rand(7, 5), rand(6, 6)
        u0, w0 = copy(u), copy(w)
        fill_velocity_halo!(bcs, box, (u, w), (dx, dz), μ)
        # interior unchanged, except the wall faces where the normal velocity is set to zero
        @test u[3:5, 2:4] == u0[3:5, 2:4] && w[2:5, 3:4] == w0[2:5, 3:4]
        @test all(iszero, u[[2, 6], 2:4]) && all(iszero, w[:, [2, 5]])
        @test u[1, 2:4] == -u[3, 2:4] && w[:, 1] == -w[:, 3]                            # normal velocity is odd
        @test w[1, 3:5] == -w[2, 3:5]                                                  # no slip: w = 0 on the left wall
        @test w[6, 3:5] == w[5, 3:5]                                                   # free slip: ∂w/∂x = 0 on the right wall
        @test (u[:, 1] .+ u[:, 2]) ./ 2 ≈ U.(Xpoint.(Ref(box), dx, 1:7))              # moving bottom wall
        @test μ .* (u[:, 5] .- u[:, 4]) ./ dz ≈ fill(0.1, 7)                          # traction on the top wall
        @test allocations(fill_velocity_halo!, bcs, box, (u, w), (dx, dz), μ) == 0
    end

    @testset "3D" begin
        box = Box3D(Mx = 4, My = 3, Mz = 2, boundary = (Periodic, Bounded, Bounded))  # u: 6×5×4, v: 6×6×4, w: 6×5×5
        Δ, μ = (0.25, 0.3, 0.5), 1e-3
        bcs = discretize(BoxBoundaryConditions(box; front = wall(NoSlip()), back = wall(FreeSlip()),
            bottom = wall(TangentialVelocity((1.0, (x, y) -> y))), top = wall(Stress((0.0, 0.1)))), box, Δ)
        u, v, w = rand(6, 5, 4), rand(6, 6, 4), rand(6, 5, 5)
        fill_velocity_halo!(bcs, box, (u, v, w), Δ, μ)
        @test u[1, :, :] == u[5, :, :] && u[6, :, :] == u[2, :, :]                    # periodic in x
        @test all(iszero, v[:, [2, 5], 2:3]) && all(iszero, w[:, 2:4, [2, 4]])       # normal velocities on walls
        @test u[:, 1, 2:3] == -u[:, 2, 2:3]                                          # no slip on the front wall
        @test (u[2:5, 2:4, 1] .+ u[2:5, 2:4, 2]) ./ 2 ≈ ones(4, 3)                     # bottom moves at (1, y)
        @test (v[2:5, 2:5, 1] .+ v[2:5, 2:5, 2]) ./ 2 ≈ [Ypoint(box, Δ[2], J) for i in 2:5, J in 2:5]
        @test μ .* (v[2:5, 2:5, 4] .- v[2:5, 2:5, 3]) ./ Δ[3] ≈ fill(0.1, 4, 4)        # traction on the top wall
        @test allocations(fill_velocity_halo!, bcs, box, (u, v, w), Δ, μ) == 0
    end

    @testset "zeroed halos and walls" begin
        box = Box2D(Mx = 4, Mz = 3, boundary = (Periodic, Bounded))        # cells 6×5, u 6×5, w 6×6
        w = ones(6, 6)
        zero_halo!(box, w, (Centred(), Faced()))
        @test all(iszero, w[[1, 6], :]) && all(iszero, w[:, [1, 2, 5, 6]]) && all(isone, w[2:5, 3:4])
        u = ones(6, 5)
        zero_halo!(box, u, (Faced(), Centred()))                           # x periodic: no wall faces
        @test all(iszero, u[[1, 6], :]) && all(iszero, u[:, [1, 5]]) && all(isone, u[2:5, 2:4])
        J = ones(6, 6)
        zero_walls!(box, J, Val(2))
        @test all(iszero, J[:, [2, 5]]) && count(iszero, J) == 12
        @test zero_walls!(box, ones(6, 5), Val(1)) == ones(6, 5)          # periodic direction: untouched
        @test allocations(zero_halo!, box, w, (Centred(), Faced())) == 0
    end

    @testset "mirrored halos" begin
        box = Box3D(Mx = 4, My = 3, Mz = 2, boundary = (Periodic, Bounded, Bounded))
        a = rand(6, 5, 4, 2)
        mirror_halo!(box, a)
        @test a[1, :, :, :] == a[5, :, :, :] && a[6, :, :, :] == a[2, :, :, :]
        @test a[:, 1, :, :] == a[:, 2, :, :] && a[:, 5, :, :] == a[:, 4, :, :]
        @test a[:, :, 1, :] == a[:, :, 2, :] && a[:, :, 4, :] == a[:, :, 3, :]
        @test allocations(mirror_halo!, box, a) == 0
    end
end

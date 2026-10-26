# Boxes and grids: type parameters, index ranges and coordinates along each dimension, grid locations, alloc

@testset "boxes and grids" begin
    # the type parameters are returned as given, for every topology
    @testset "boxes" begin
        for Bx in (Periodic, Bounded), By in (Periodic, Bounded), Bz in (Periodic, Bounded)
            box2 = Box2D(Mx = 5, Mz = 4, Hx = 1, Hz = 2, boundary = (Bx, Bz))
            @test (dims(box2), halo_size(box2), boundary_topology(box2)) == ((5, 4), (1, 2), (Bx(), Bz()))
            box3 = Box3D(Mx = 5, My = 3, Mz = 4, Hx = 2, Hy = 1, Hz = 1, boundary = (Bx, By, Bz))
            @test (dims(box3), halo_size(box3), boundary_topology(box3)) == ((5, 3, 4), (2, 1, 1), (Bx(), By(), Bz()))
        end
    end

    # sizes, index ranges and coordinates along each dimension, for every topology and halo width
    @testset "grid along a dimension" begin
        for (M, H) in (((5, 4), (1, 2)), ((5, 3, 4), (2, 1, 1))), topology in Iterators.product(ntuple(_ -> (Periodic, Bounded), length(M))...)
            box = length(M) == 2 ? Box2D(Mx = M[1], Mz = M[2], Hx = H[1], Hz = H[2], boundary = topology) :
                                   Box3D(Mx = M[1], My = M[2], Mz = M[3], Hx = H[1], Hy = H[2], Hz = H[3], boundary = topology)
            for d in eachindex(M)
                m, h, Δ, bounded = M[d], H[d], 0.1d, topology[d] == Bounded
                @test ncells(box, d) == npoints(box, Cell(), d) == m + 2h
                @test nfaces(box, d) == npoints(box, Face(), d) == m + 2h + bounded     # bounded: both walls
                @test cells(box, d) == 1:nfaces(box, d)-1 && faces(box, d) == 2:ncells(box, d)
                @test interior_cells(box, d) == h+1:h+m
                @test interior_faces(box, d) == h+1:h+m+bounded
                # the first interior face is at 0, cell centers half a cell further; the last face is at mΔ if bounded
                @test cellpoint(box, Δ, d, h+1) ≈ Δ / 2
                # in the float type of the spacing
                @test all(F -> cellpoint(box, F(Δ), d, h+1) isa F && facepoint(box, F(Δ), d, h+1) isa F, (Float32, BigFloat))
                @test facepoint(box, Δ, d, first(interior_faces(box, d))) ≈ 0
                @test facepoint(box, Δ, d, last(interior_faces(box, d))) ≈ (m - !bounded) * Δ
                @test cellgrid(box, Δ, d)[interior_cells(box, d)] ≈ ((1:m) .- 0.5) .* Δ
                @test facegrid(box, Δ, d) == facepoint.(Ref(box), Δ, d, 1:nfaces(box, d))
            end
        end
    end

    # every 2D grid location: the interior view and the coordinates have the same shape
    @testset "grid locations" begin
        for Bx in (Periodic, Bounded), Bz in (Periodic, Bounded)
            box = Box2D(Mx = 6, Mz = 4, Hx = 2, Hz = 1, boundary = (Bx, Bz))
            dx, dz = 1 / 6, 0.5 / 4
            for location in ((Cell(), Cell()), (Face(), Cell()), (Cell(), Face()), (Face(), Face()))
                a = alloc(Float64, nothing, box, location)
                x, z = coordinates(box, (dx, dz), location)
                @test interior(box, a) isa SubArray && size(interior(box, a)) == (length(x), length(z))
            end
            # w-faces: x at the cell centers, z from the bottom face to the top wall (bounded) or the last face (periodic)
            x, z = coordinates(box, (dx, dz), (Cell(), Face()))
            @test x[1] ≈ dx / 2 && z[1] ≈ 0 && z[end] ≈ (Bz == Bounded ? 4 : 3) * dz
        end
        box3 = Box3D(Mx = 4, My = 3, Mz = 2, boundary = (Bounded, Periodic, Bounded))
        @test size(interior(box3, alloc(Float64, nothing, box3, (Face(), Cell(), Face())))) == (5, 3, 3)
    end

    # alloc gives zeros on the location, with trailing dimensions, for any loop manager; alloc! reuses arrays
    @testset "alloc" begin
        box2 = Box2D(Mx = 6, Mz = 4, boundary = (Periodic, Bounded))
        box3 = Box3D(Mx = 4, My = 3, Mz = 2)
        for mgr in (nothing, PlainCPU(), MultiThread(VectorizedCPU()))
            a = alloc(Float32, mgr, box2, (Face(), Face()), 3)
            @test size(a) == (8, 7, 3) && eltype(a) == Float32 && all(iszero, a)
            @test all(iszero, alloc(Float64, mgr, box3, (Cell(), Face(), Cell()), 2))
        end
        a = alloc(Float64, nothing, box2, (Cell(), Cell()))
        @test alloc!(a, Float64, nothing, box2, (Cell(), Cell())) === a                     # an array is reused
        @test alloc!(void, Float64, nothing, box2, (Cell(), Cell())) == a                   # void: a new zero array
    end
end

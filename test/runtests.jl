using CFBoxes
using Test

Mx = 16
My = 16
Mz = 16
Hx = 1
Hy = 1
Hz = 1

F = Float64

@testset "CFBoxes.jl" begin
    # Write your tests here.

    for bdry_x in (Periodic, Bounded), bdry_y in (Periodic, Bounded), bdry_z in (Periodic, Bounded)
        # test box construction and properties

        box2d = Box2D(
            Mx = Mx,
            Mz = Mz,
            Hx = Hx,
            Hz = Hz,
            boundary = (bdry_x, bdry_z)
        )
        @test dims(box2d) == (Mx, Mz)
        @test halo_size(box2d) == (Hx, Hz)
        @test boundary_topology(box2d) == (bdry_x(), bdry_z())

        box3d = Box3D(
            Mx = Mx,
            My = My,
            Mz = Mz,
            Hx = Hx,
            Hy = Hy,
            Hz = Hz,
            boundary = (bdry_x, bdry_y, bdry_z)
        )
        @test dims(box3d) == (Mx, My, Mz)
        @test halo_size(box3d) == (Hx, Hy, Hz)
        @test boundary_topology(box3d) == (bdry_x(), bdry_y(), bdry_z())
    end

    # each grid location: interior is a view, and its coordinates match it, for every boundary topology
    @testset "grid locations" begin
        for bdry_x in (Periodic, Bounded), bdry_z in (Periodic, Bounded)
            box = Box2D(; Mx, Mz, Hx = 2, Hz = 1, boundary = (bdry_x, bdry_z))
            dx, dz = 1 / Mx, 0.5 / Mz
            for location in ((Cell(), Cell()), (Face(), Cell()), (Cell(), Face()), (Face(), Face()))
                data = alloc(F, box, location)
                x, z = coordinates(box, (dx, dz), location)
                @test interior(box, data) isa SubArray
                @test size(interior(box, data)) == (length(x), length(z))
            end
            # cells at (i - 1/2) Δx, faces at (I - 1) Δx from the first wall; bounded: both wall faces
            x, z = coordinates(box, (dx, dz), (Cell(), Face()))
            @test x[1] ≈ dx / 2 && z[1] ≈ 0
            @test bdry_z == Bounded ? z[end] ≈ Mz * dz : z[end] ≈ (Mz - 1) * dz
        end
        box3d = Box3D(; Mx, My, Mz, Hx, Hy, Hz, boundary = (Bounded, Periodic, Bounded))
        @test size(alloc(F, box3d, (Face(), Cell(), Face()), 2)) == (nfaces(box3d, 1), ncells(box3d, 2), nfaces(box3d, 3), 2)
    end

    # the grid along each dimension d: sizes, ranges and coordinates, for every topology and halo width
    @testset "grid along a dimension" begin
        for (Mb, H) in (((5, 4), (1, 2)), ((5, 3, 4), (2, 1, 1))), topology in Iterators.product(ntuple(_ -> (Periodic, Bounded), length(Mb))...)
            box = length(Mb) == 2 ? Box2D(Mx = Mb[1], Mz = Mb[2], Hx = H[1], Hz = H[2], boundary = topology) :
                                    Box3D(Mx = Mb[1], My = Mb[2], Mz = Mb[3], Hx = H[1], Hy = H[2], Hz = H[3], boundary = topology)
            for d in eachindex(Mb)
                M, h, Δ, bounded = Mb[d], H[d], 0.1d, topology[d] == Bounded
                @test ncells(box, d) == M + 2h
                @test nfaces(box, d) == M + 2h + bounded
                @test cells(box, d) == 1:nfaces(box, d)-1 && faces(box, d) == 2:ncells(box, d)
                @test interior_cells(box, d) == h+1:h+M
                @test interior_faces(box, d) == h+1:h+M+bounded
                # cell centres half a cell from the first wall; the interior faces start at 0 and, if bounded, end at MΔ
                @test cellpoint(box, Δ, d, h+1) ≈ Δ / 2
                @test facepoint(box, Δ, d, first(interior_faces(box, d))) ≈ 0
                @test facepoint(box, Δ, d, last(interior_faces(box, d))) ≈ (M - !bounded) * Δ
                @test cellgrid(box, Δ, d)[interior_cells(box, d)] ≈ ((1:M) .- 0.5) .* Δ
                @test length(facegrid(box, Δ, d)) == nfaces(box, d)
            end
        end
    end

    include("operators.jl")
    include("vortex_force.jl")
    include("walls.jl")
end

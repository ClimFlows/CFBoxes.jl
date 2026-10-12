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

        # test 2D grid allocation
        xz_test = alloc_xz(F, box2d)
        xZ_test = alloc_xZ(F, box2d)
        Xz_test = alloc_Xz(F, box2d)
        XZ_test = alloc_XZ(F, box2d)

        # test 2D centered operators
        for i in xrange(box2d), k in zrange(box2d)
            # single array
            avg_x(Xz_test, i, k)
            dif_x(Xz_test, i, k)
            avg_z(xZ_test, i, k)
            dif_z(xZ_test, i, k)
            avg_xz(XZ_test, i, k)
            # double array
            avg_x(Xz_test, Xz_test, i, k)
            dif_x(Xz_test, Xz_test, i, k)
            avg_z(xZ_test, xZ_test, i, k)
            dif_z(xZ_test, xZ_test, i, k)
            avg_xz(XZ_test, XZ_test, i, k)
        end
        for i in Xrange(box2d), k in zrange(box2d)
            # single array
            avg_X(xz_test, i, k)
            dif_X(xz_test, i, k)
            avg_z(XZ_test, i, k)
            dif_z(XZ_test, i, k)
            avg_Xz(xZ_test, i, k)
            # double array
            avg_X(xz_test, xz_test, i, k)
            dif_X(xz_test, xz_test, i, k)
            avg_z(XZ_test, XZ_test, i, k)
            dif_z(XZ_test, XZ_test, i, k)
            avg_Xz(xZ_test, xZ_test, i, k)
        end
        for i in xrange(box2d), k in Zrange(box2d)
            # single array
            avg_x(XZ_test, i, k)
            dif_x(XZ_test, i, k)
            avg_Z(xz_test, i, k)
            dif_Z(xz_test, i, k)
            avg_xZ(Xz_test, i, k)
            # double array
            avg_x(XZ_test, XZ_test, i, k)
            dif_x(XZ_test, XZ_test, i, k)
            avg_Z(xz_test, xz_test, i, k)
            dif_Z(xz_test, xz_test, i, k)
            avg_xZ(Xz_test, Xz_test, i, k)
        end
        for i in Xrange(box2d), k in Zrange(box2d)
            # single array
            avg_X(xZ_test, i, k)
            dif_X(xZ_test, i, k)
            avg_Z(Xz_test, i, k)
            dif_Z(Xz_test, i, k)
            avg_XZ(xz_test, i, k)
            # double array
            avg_X(xZ_test, xZ_test, i, k)
            dif_X(xZ_test, xZ_test, i, k)
            avg_Z(Xz_test, Xz_test, i, k)
            dif_Z(Xz_test, Xz_test, i, k)
            avg_XZ(xz_test, xz_test, i, k)
        end
    end

    # each grid location: interior is a view, and its coordinates match it, for every boundary topology
    @testset "grid locations" begin
        for bdry_x in (Periodic, Bounded), bdry_z in (Periodic, Bounded)
            box = Box2D(; Mx, Mz, Hx = 2, Hz = 1, boundary = (bdry_x, bdry_z))
            dx, dz = 1 / Mx, 0.5 / Mz
            for location in (:xz, :Xz, :xZ, :XZ)
                data = alloc(F, box, location)
                x, z = coordinates(box, (dx, dz), location)
                @test interior(box, data) isa SubArray
                @test size(interior(box, data)) == (length(x), length(z))
            end
            # cells at (i - 1/2) Δx, faces at (I - 1) Δx from the first wall; bounded: both wall faces
            x, z = coordinates(box, (dx, dz), :xZ)
            @test x[1] ≈ dx / 2 && z[1] ≈ 0
            @test bdry_z == Bounded ? z[end] ≈ Mz * dz : z[end] ≈ (Mz - 1) * dz
        end
        box3d = Box3D(; Mx, My, Mz, Hx, Hy, Hz, boundary = (Bounded, Periodic, Bounded))
        @test size(alloc(F, box3d, :XyZ, 2)) == (Xdim(box3d), ydim(box3d), Zdim(box3d), 2)
    end

    include("walls.jl")
end

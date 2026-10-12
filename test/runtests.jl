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

    include("walls.jl")
end

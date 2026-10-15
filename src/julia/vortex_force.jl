## the vortex force ω×u on the C-grid, from the velocity and the density at the cells
#
# `model` is any object with the fields `mgr`, `domain`, `spatial_scheme` and `inv_dx`, `inv_dz` (plus `inv_dy` in
# 3D). Face densities ρᴵ = ℑ(ρ, (1,), (Face(),)), ρᴶ, ρᴷ weight the velocities into mass fluxes, exactly as in the tracer fluxes
# u ℑ(ρ, (1,), (Face(),)), so that the vortex force does no work: Σ ρᴵ u (ω×u)_x + … = 0 with the energy-conserving scheme.

"""
    ωw, ωu, scratch = vortex_force!((ωw, ωu, scratch), model, scheme::VortexScheme, (u, w), ρ)

2D vortex force on the faces: `ωw = (ω×u)_x` on the u-faces and `ωu = -(ω×u)_z` on the w-faces, with
ω = ∂u/∂z - ∂w/∂x at the vertices, the velocity `(u, w)` on the faces and the density `ρ` at the cells: a cell array, or
a vertical profile (a vector, e.g. the anelastic reference density). The momentum tendencies are `-ωw` and `+ωu`. `scheme` is `VortexEnergyCons()` or `VortexEnstrophyCons()` (Sadourny 1975).

    fx, fy, fz, scratch = vortex_force!((fx, fy, fz, scratch), model, scheme, (u, v, w), ρ)

3D vortex force `(fx, fy, fz) = ω×u` on the u-, v- and w-faces, with ω = ∇×u on the edges.
"""
function vortex_force!((ωw_, ωu_, (ρᴵ_, ρᴶ_, ωwᴵᴶ_, ωuᴵᴶ_)), model, ::VortexEnergyCons, (u, w)::NTuple{2}, ρ)
    (; mgr, domain, inv_dx, inv_dz, spatial_scheme) = model
    (; ℑ, δ) = operators(spatial_scheme)
    F = eltype(u)

    ωw      = alloc!(ωw_, F, domain, (Face(), Cell()))
    ωu      = alloc!(ωu_, F, domain, (Cell(), Face()))
    ωwᴵᴶ    = alloc!(ωwᴵᴶ_, F, domain, (Face(), Face()))
    ωuᴵᴶ    = alloc!(ωuᴵᴶ_, F, domain, (Face(), Face()))
    ρᴵ, ρᴶ  = face_densities!((ρᴵ_, ρᴶ_), model, ρ)

    # vertices: ω ρw / ρ and ω ρu / ρ
    @with mgr, let (irange, jrange) = (faces(domain, 1), faces(domain, 2))
        @inbounds @vec for I in irange, J in jrange
            ω = inv_dz * δ(u, (2,), (Face(),), I, J) - inv_dx * δ(w, (1,), (Face(),), I, J)
            ωwᴵᴶ[I, J] = ω * ℑ((ρᴶ, w), (1,), (Face(),), I, J) * inv(ℑ(ρ, (1, 2), (Face(), Face()), I, J))
            ωuᴵᴶ[I, J] = ω * ℑ((ρᴵ, u), (2,), (Face(),), I, J) * inv(ℑ(ρ, (1, 2), (Face(), Face()), I, J))
        end
    end
    periodize!(model, (ωwᴵᴶ, ωuᴵᴶ))

    # u-faces: ωw
    @with mgr, let (irange, jrange) = (axes(ωw, 1), cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            ωw[I, j] = ℑ(ωwᴵᴶ, (2,), (Cell(),), I, j)
        end
    end
    # w-faces: ωu
    @with mgr, let (irange, jrange) = (cells(domain, 1), axes(ωu, 2))
        @inbounds @vec for i in irange, J in jrange
            ωu[i, J] = ℑ(ωuᴵᴶ, (1,), (Cell(),), i, J)
        end
    end
    periodize!(model, (ωw, ωu))

    return ωw, ωu, (ρᴵ, ρᴶ, ωwᴵᴶ, ωuᴵᴶ)
end

# enstrophy-conserving: ζ = ω/ρ and the mass fluxes interpolated separately to the faces, then multiplied
function vortex_force!((ωw_, ωu_, (ρᴵ_, ρᴶ_, ζᴵᴶ_, ρwᴵ_, ρuᴶ_)), model, ::VortexEnstrophyCons, (u, w)::NTuple{2}, ρ)
    (; mgr, domain, inv_dx, inv_dz, spatial_scheme) = model
    (; ℑ, δ) = operators(spatial_scheme)
    F = eltype(u)

    ωw      = alloc!(ωw_, F, domain, (Face(), Cell()))
    ωu      = alloc!(ωu_, F, domain, (Cell(), Face()))
    ζᴵᴶ     = alloc!(ζᴵᴶ_, F, domain, (Face(), Face()))
    ρwᴵ     = alloc!(ρwᴵ_, F, domain, (Face(), Face()))
    ρuᴶ     = alloc!(ρuᴶ_, F, domain, (Face(), Face()))
    ρᴵ, ρᴶ  = face_densities!((ρᴵ_, ρᴶ_), model, ρ)

    # vertices: ζ = ω / ρ and the mass fluxes ρw, ρu
    @with mgr, let (irange, jrange) = (faces(domain, 1), faces(domain, 2))
        @inbounds @vec for I in irange, J in jrange
            ω = inv_dz * δ(u, (2,), (Face(),), I, J) - inv_dx * δ(w, (1,), (Face(),), I, J)
            ζᴵᴶ[I, J] = ω * inv(ℑ(ρ, (1, 2), (Face(), Face()), I, J))
            ρwᴵ[I, J] = ℑ((ρᴶ, w), (1,), (Face(),), I, J)
            ρuᴶ[I, J] = ℑ((ρᴵ, u), (2,), (Face(),), I, J)
        end
    end
    periodize!(model, (ζᴵᴶ, ρwᴵ, ρuᴶ))

    # u-faces: ωw
    @with mgr, let (irange, jrange) = (axes(ωw, 1), cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            ωw[I, j] = ℑ(ζᴵᴶ, (2,), (Cell(),), I, j) * ℑ(ρwᴵ, (2,), (Cell(),), I, j)
        end
    end
    # w-faces: ωu
    @with mgr, let (irange, jrange) = (cells(domain, 1), axes(ωu, 2))
        @inbounds @vec for i in irange, J in jrange
            ωu[i, J] = ℑ(ζᴵᴶ, (1,), (Cell(),), i, J) * ℑ(ρuᴶ, (1,), (Cell(),), i, J)
        end
    end
    periodize!(model, (ωw, ωu))

    return ωw, ωu, (ρᴵ, ρᴶ, ζᴵᴶ, ρwᴵ, ρuᴶ)
end

# 3D, energy-conserving: products of ω and the mass fluxes on the edges, interpolated to the faces. Edges: ωx on
# xYZ (ᴶᴷ), ωy on XyZ (ᴵᴷ), ωz on XYz (ᴵᴶ).
function vortex_force!((fx_, fy_, fz_, (ρᴵ_, ρᴶ_, ρᴷ_, ωywᴵᴷ_, ωyuᴵᴷ_, ωzuᴵᴶ_, ωzvᴵᴶ_, ωxvᴶᴷ_, ωxwᴶᴷ_)), model,
                       ::VortexEnergyCons, (u, v, w)::NTuple{3}, ρ)
    (; mgr, domain, inv_dx, inv_dy, inv_dz, spatial_scheme) = model
    (; ℑ, δ) = operators(spatial_scheme)
    F = eltype(u)

    fx          = alloc!(fx_, F, domain, (Face(), Cell(), Cell()))
    fy          = alloc!(fy_, F, domain, (Cell(), Face(), Cell()))
    fz          = alloc!(fz_, F, domain, (Cell(), Cell(), Face()))
    ωywᴵᴷ       = alloc!(ωywᴵᴷ_, F, domain, (Face(), Cell(), Face()))
    ωyuᴵᴷ       = alloc!(ωyuᴵᴷ_, F, domain, (Face(), Cell(), Face()))
    ωzuᴵᴶ       = alloc!(ωzuᴵᴶ_, F, domain, (Face(), Face(), Cell()))
    ωzvᴵᴶ       = alloc!(ωzvᴵᴶ_, F, domain, (Face(), Face(), Cell()))
    ωxvᴶᴷ       = alloc!(ωxvᴶᴷ_, F, domain, (Cell(), Face(), Face()))
    ωxwᴶᴷ       = alloc!(ωxwᴶᴷ_, F, domain, (Cell(), Face(), Face()))
    ρᴵ, ρᴶ, ρᴷ  = face_densities!((ρᴵ_, ρᴶ_, ρᴷ_), model, ρ)

    # XyZ edges: ωy = ∂u/∂z - ∂w/∂x, times ρw and ρu, over ρ
    @with mgr, let (irange, jrange, krange) = (faces(domain, 1), axes(ωywᴵᴷ, 2), faces(domain, 3))
        for K in krange, j in jrange
            @inbounds @vec for I in irange
                ωy = inv_dz * δ(u, (3,), (Face(),), I, j, K) - inv_dx * δ(w, (1,), (Face(),), I, j, K)
                ωywᴵᴷ[I, j, K] = ωy * ℑ((ρᴷ, w), (1,), (Face(),), I, j, K) * inv(ℑ(ρ, (1, 3), (Face(), Face()), I, j, K))
                ωyuᴵᴷ[I, j, K] = ωy * ℑ((ρᴵ, u), (3,), (Face(),), I, j, K) * inv(ℑ(ρ, (1, 3), (Face(), Face()), I, j, K))
            end
        end
    end
    # XYz edges: ωz = ∂v/∂x - ∂u/∂y, times ρu and ρv, over ρ
    @with mgr, let (irange, jrange, krange) = (faces(domain, 1), faces(domain, 2), axes(ωzuᴵᴶ, 3))
        for k in krange, J in jrange
            @inbounds @vec for I in irange
                ωz = inv_dx * δ(v, (1,), (Face(),), I, J, k) - inv_dy * δ(u, (2,), (Face(),), I, J, k)
                ωzuᴵᴶ[I, J, k] = ωz * ℑ((ρᴵ, u), (2,), (Face(),), I, J, k) * inv(ℑ(ρ, (1, 2), (Face(), Face()), I, J, k))
                ωzvᴵᴶ[I, J, k] = ωz * ℑ((ρᴶ, v), (1,), (Face(),), I, J, k) * inv(ℑ(ρ, (1, 2), (Face(), Face()), I, J, k))
            end
        end
    end
    # xYZ edges: ωx = ∂w/∂y - ∂v/∂z, times ρv and ρw, over ρ
    @with mgr, let (irange, jrange, krange) = (axes(ωxvᴶᴷ, 1), faces(domain, 2), faces(domain, 3))
        for K in krange, J in jrange
            @inbounds @vec for i in irange
                ωx = inv_dy * δ(w, (2,), (Face(),), i, J, K) - inv_dz * δ(v, (3,), (Face(),), i, J, K)
                ωxvᴶᴷ[i, J, K] = ωx * ℑ((ρᴶ, v), (3,), (Face(),), i, J, K) * inv(ℑ(ρ, (2, 3), (Face(), Face()), i, J, K))
                ωxwᴶᴷ[i, J, K] = ωx * ℑ((ρᴷ, w), (2,), (Face(),), i, J, K) * inv(ℑ(ρ, (2, 3), (Face(), Face()), i, J, K))
            end
        end
    end
    periodize!(model, (ωywᴵᴷ, ωyuᴵᴷ, ωzuᴵᴶ, ωzvᴵᴶ, ωxvᴶᴷ, ωxwᴶᴷ))

    # faces: (ω×u)_x = ωy w - ωz v, (ω×u)_y = ωz u - ωx w, (ω×u)_z = ωx v - ωy u
    @with mgr, let (irange, jrange, krange) = (faces(domain, 1), cells(domain, 2), cells(domain, 3))
        for k in krange, j in jrange
            @inbounds @vec for I in irange
                fx[I, j, k] = ℑ(ωywᴵᴷ, (3,), (Cell(),), I, j, k) - ℑ(ωzvᴵᴶ, (2,), (Cell(),), I, j, k)
            end
        end
    end
    @with mgr, let (irange, jrange, krange) = (cells(domain, 1), faces(domain, 2), cells(domain, 3))
        for k in krange, J in jrange
            @inbounds @vec for i in irange
                fy[i, J, k] = ℑ(ωzuᴵᴶ, (1,), (Cell(),), i, J, k) - ℑ(ωxwᴶᴷ, (3,), (Cell(),), i, J, k)
            end
        end
    end
    @with mgr, let (irange, jrange, krange) = (cells(domain, 1), cells(domain, 2), faces(domain, 3))
        for K in krange, j in jrange
            @inbounds @vec for i in irange
                fz[i, j, K] = ℑ(ωxvᴶᴷ, (2,), (Cell(),), i, j, K) - ℑ(ωyuᴵᴷ, (1,), (Cell(),), i, j, K)
            end
        end
    end
    periodize!(model, (fx, fy, fz))

    return fx, fy, fz, (ρᴵ, ρᴶ, ρᴷ, ωywᴵᴷ, ωyuᴵᴷ, ωzuᴵᴶ, ωzvᴵᴶ, ωxvᴶᴷ, ωxwᴶᴷ)
end

# 3D, enstrophy-type: ζ = ω/ρ and the mass fluxes interpolated separately to the faces, then multiplied (enstrophy-
# conserving in 2D)
function vortex_force!((fx_, fy_, fz_, (ρᴵ_, ρᴶ_, ρᴷ_, ζyᴵᴷ_, ρwᴵᴷ_, ρuᴵᴷ_, ζzᴵᴶ_, ρuᴵᴶ_, ρvᴵᴶ_, ζxᴶᴷ_, ρvᴶᴷ_, ρwᴶᴷ_)),
                       model, ::VortexEnstrophyCons, (u, v, w)::NTuple{3}, ρ)
    (; mgr, domain, inv_dx, inv_dy, inv_dz, spatial_scheme) = model
    (; ℑ, δ) = operators(spatial_scheme)
    F = eltype(u)

    fx          = alloc!(fx_, F, domain, (Face(), Cell(), Cell()))
    fy          = alloc!(fy_, F, domain, (Cell(), Face(), Cell()))
    fz          = alloc!(fz_, F, domain, (Cell(), Cell(), Face()))
    ζyᴵᴷ        = alloc!(ζyᴵᴷ_, F, domain, (Face(), Cell(), Face()))
    ρwᴵᴷ        = alloc!(ρwᴵᴷ_, F, domain, (Face(), Cell(), Face()))
    ρuᴵᴷ        = alloc!(ρuᴵᴷ_, F, domain, (Face(), Cell(), Face()))
    ζzᴵᴶ        = alloc!(ζzᴵᴶ_, F, domain, (Face(), Face(), Cell()))
    ρuᴵᴶ        = alloc!(ρuᴵᴶ_, F, domain, (Face(), Face(), Cell()))
    ρvᴵᴶ        = alloc!(ρvᴵᴶ_, F, domain, (Face(), Face(), Cell()))
    ζxᴶᴷ        = alloc!(ζxᴶᴷ_, F, domain, (Cell(), Face(), Face()))
    ρvᴶᴷ        = alloc!(ρvᴶᴷ_, F, domain, (Cell(), Face(), Face()))
    ρwᴶᴷ        = alloc!(ρwᴶᴷ_, F, domain, (Cell(), Face(), Face()))
    ρᴵ, ρᴶ, ρᴷ  = face_densities!((ρᴵ_, ρᴶ_, ρᴷ_), model, ρ)

    # XyZ edges: ζy = ωy / ρ, ρw and ρu
    @with mgr, let (irange, jrange, krange) = (faces(domain, 1), axes(ζyᴵᴷ, 2), faces(domain, 3))
        for K in krange, j in jrange
            @inbounds @vec for I in irange
                ζyᴵᴷ[I, j, K] = (inv_dz * δ(u, (3,), (Face(),), I, j, K) - inv_dx * δ(w, (1,), (Face(),), I, j, K)) * inv(ℑ(ρ, (1, 3), (Face(), Face()), I, j, K))
                ρwᴵᴷ[I, j, K] = ℑ((ρᴷ, w), (1,), (Face(),), I, j, K)
                ρuᴵᴷ[I, j, K] = ℑ((ρᴵ, u), (3,), (Face(),), I, j, K)
            end
        end
    end
    # XYz edges: ζz = ωz / ρ, ρu and ρv
    @with mgr, let (irange, jrange, krange) = (faces(domain, 1), faces(domain, 2), axes(ζzᴵᴶ, 3))
        for k in krange, J in jrange
            @inbounds @vec for I in irange
                ζzᴵᴶ[I, J, k] = (inv_dx * δ(v, (1,), (Face(),), I, J, k) - inv_dy * δ(u, (2,), (Face(),), I, J, k)) * inv(ℑ(ρ, (1, 2), (Face(), Face()), I, J, k))
                ρuᴵᴶ[I, J, k] = ℑ((ρᴵ, u), (2,), (Face(),), I, J, k)
                ρvᴵᴶ[I, J, k] = ℑ((ρᴶ, v), (1,), (Face(),), I, J, k)
            end
        end
    end
    # xYZ edges: ζx = ωx / ρ, ρv and ρw
    @with mgr, let (irange, jrange, krange) = (axes(ζxᴶᴷ, 1), faces(domain, 2), faces(domain, 3))
        for K in krange, J in jrange
            @inbounds @vec for i in irange
                ζxᴶᴷ[i, J, K] = (inv_dy * δ(w, (2,), (Face(),), i, J, K) - inv_dz * δ(v, (3,), (Face(),), i, J, K)) * inv(ℑ(ρ, (2, 3), (Face(), Face()), i, J, K))
                ρvᴶᴷ[i, J, K] = ℑ((ρᴶ, v), (3,), (Face(),), i, J, K)
                ρwᴶᴷ[i, J, K] = ℑ((ρᴷ, w), (2,), (Face(),), i, J, K)
            end
        end
    end
    periodize!(model, (ζyᴵᴷ, ρwᴵᴷ, ρuᴵᴷ, ζzᴵᴶ, ρuᴵᴶ, ρvᴵᴶ, ζxᴶᴷ, ρvᴶᴷ, ρwᴶᴷ))

    # faces: (ω×u)_x = ωy w - ωz v, (ω×u)_y = ωz u - ωx w, (ω×u)_z = ωx v - ωy u
    @with mgr, let (irange, jrange, krange) = (faces(domain, 1), cells(domain, 2), cells(domain, 3))
        for k in krange, j in jrange
            @inbounds @vec for I in irange
                fx[I, j, k] = ℑ(ζyᴵᴷ, (3,), (Cell(),), I, j, k) * ℑ(ρwᴵᴷ, (3,), (Cell(),), I, j, k) - ℑ(ζzᴵᴶ, (2,), (Cell(),), I, j, k) * ℑ(ρvᴵᴶ, (2,), (Cell(),), I, j, k)
            end
        end
    end
    @with mgr, let (irange, jrange, krange) = (cells(domain, 1), faces(domain, 2), cells(domain, 3))
        for k in krange, J in jrange
            @inbounds @vec for i in irange
                fy[i, J, k] = ℑ(ζzᴵᴶ, (1,), (Cell(),), i, J, k) * ℑ(ρuᴵᴶ, (1,), (Cell(),), i, J, k) - ℑ(ζxᴶᴷ, (3,), (Cell(),), i, J, k) * ℑ(ρwᴶᴷ, (3,), (Cell(),), i, J, k)
            end
        end
    end
    @with mgr, let (irange, jrange, krange) = (cells(domain, 1), cells(domain, 2), faces(domain, 3))
        for K in krange, j in jrange
            @inbounds @vec for i in irange
                fz[i, j, K] = ℑ(ζxᴶᴷ, (2,), (Cell(),), i, j, K) * ℑ(ρvᴶᴷ, (2,), (Cell(),), i, j, K) - ℑ(ζyᴵᴷ, (1,), (Cell(),), i, j, K) * ℑ(ρuᴵᴷ, (1,), (Cell(),), i, j, K)
            end
        end
    end
    periodize!(model, (fx, fy, fz))

    return fx, fy, fz, (ρᴵ, ρᴶ, ρᴷ, ζyᴵᴷ, ρwᴵᴷ, ρuᴵᴷ, ζzᴵᴶ, ρuᴵᴶ, ρvᴵᴶ, ζxᴶᴷ, ρvᴶᴷ, ρwᴶᴷ)
end

# face densities ρᴵ = ℑ(ρ, (1,), (Face(),)), ρᴶ = ℑ(ρ, (2,), (Face(),)) or ℑ(ρ, (2,), (Face(),)) (2D), ρᴷ = ℑ(ρ, (2,), (Face(),)), over the full axes along which they are not
# interpolated; ρ is a cell array or a vertical profile
function face_densities!(scratch, model, ρ)
    (; domain) = model
    return face_densities!(scratch, model, domain, ρ)
end
function face_densities!((ρᴵ_, ρᴶ_), model, ::Box2D, ρ)
    (; mgr, domain, spatial_scheme) = model
    F = eltype(ρ)
    (; ℑ) = operators(spatial_scheme)

    ρᴵ = alloc!(ρᴵ_, F, domain, (Face(), Cell()))
    ρᴶ = alloc!(ρᴶ_, F, domain, (Cell(), Face()))

    @with mgr, let (irange, jrange) = (faces(domain, 1), axes(ρᴵ, 2))
        @inbounds @vec for I in irange, j in jrange
            ρᴵ[I, j] = ℑ(ρ, (1,), (Face(),), I, j)
        end
    end
    @with mgr, let (irange, jrange) = (axes(ρᴶ, 1), faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            ρᴶ[i, J] = ℑ(ρ, (2,), (Face(),), i, J)
        end
    end
    periodize!(model, (ρᴵ, ρᴶ))
    return ρᴵ, ρᴶ
end
# vertical profile (e.g. an anelastic reference density): the face densities are profiles too, ρᴵ = ρ on the cells and
# ρᴶ = ℑ(ρ, (2,), (Face(),)) on the w-faces
function face_densities!((ρᴵ_, ρᴶ_), model, ::Box2D, ρ::AbstractVector)
    (; domain, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)

    ρᴶ = profile!(ρᴶ_, eltype(ρ), nfaces(domain, 2))

    for J in faces(domain, 2)
        ρᴶ[J] = ℑ(ρ, (2,), (Face(),), J)
    end
    return ρ, ρᴶ
end
profile!(a::AbstractVector, F, n) = a
profile!(_, F, n) = zeros(F, n)
function face_densities!((ρᴵ_, ρᴶ_, ρᴷ_), model, ::Box3D, ρ)
    (; mgr, domain, spatial_scheme) = model
    F = eltype(ρ)
    (; ℑ) = operators(spatial_scheme)

    ρᴵ = alloc!(ρᴵ_, F, domain, (Face(), Cell(), Cell()))
    ρᴶ = alloc!(ρᴶ_, F, domain, (Cell(), Face(), Cell()))
    ρᴷ = alloc!(ρᴷ_, F, domain, (Cell(), Cell(), Face()))

    @with mgr, let (irange, jrange, krange) = (faces(domain, 1), axes(ρᴵ, 2), axes(ρᴵ, 3))
        for k in krange, j in jrange
            @inbounds @vec for I in irange
                ρᴵ[I, j, k] = ℑ(ρ, (1,), (Face(),), I, j, k)
            end
        end
    end
    @with mgr, let (irange, jrange, krange) = (axes(ρᴶ, 1), faces(domain, 2), axes(ρᴶ, 3))
        for k in krange, J in jrange
            @inbounds @vec for i in irange
                ρᴶ[i, J, k] = ℑ(ρ, (2,), (Face(),), i, J, k)
            end
        end
    end
    @with mgr, let (irange, jrange, krange) = (axes(ρᴷ, 1), axes(ρᴷ, 2), faces(domain, 3))
        for K in krange, j in jrange
            @inbounds @vec for i in irange
                ρᴷ[i, j, K] = ℑ(ρ, (3,), (Face(),), i, j, K)
            end
        end
    end
    periodize!(model, (ρᴵ, ρᴶ, ρᴷ))
    return ρᴵ, ρᴶ, ρᴷ
end

## the vortex force ω×u on the C-grid, from the velocity and the density at the cells (Sadourny 1975, Arakawa & Lamb 1981)
#
# Notation, after Sadourny (1975) with his y → z (his rot V is -ω here); ℑ, δ in lower case onto the cells, upper case
# onto the faces, as in operators.jl:
#   ω  = δZ u / Δz - δX w / Δx     vorticity at the vertices (XZ)
#   η  = ω / ℑXZ(ρ)                potential vorticity at the vertices (q in Arakawa & Lamb 1981 and Salmon 2004)
#   U  = ρᴵ u,  W = ρᴶ w           mass fluxes on the u- and w-faces, with the face densities ρᴵ = ℑX(ρ), ρᴶ = ℑZ(ρ)
#   Uᴶ = ℑZ(U), Wᴵ = ℑX(W)         mass fluxes interpolated to the vertices
# vortex_x = (ω×u)_x = ω w on the u-faces and vortex_z = (ω×u)_z = -ω u on the w-faces; the momentum tendency is -ω×u.
# Sadourny's two schemes share η, Uᴶ, Wᴵ and differ only in how the products reach the faces (his eqs 3 and 4).
# The face densities are those of the tracer fluxes, so that with the energy-conserving scheme
# Σ U vortex_x + Σ W vortex_z = 0: the vortex force does no work.
#
# `model` is any object with the fields `mgr`, `domain`, `spatial_scheme` and `inv_dx`, `inv_dz` (plus `inv_dy` in 3D).

"""
    vortex_x, vortex_z, scratch = vortex_force!((vortex_x, vortex_z, scratch), model, scheme::VortexScheme, (u, w), ρ)

2D vortex force ω×u [m s⁻²]: `vortex_x` = (ω×u)_x = ω w on the u-faces and `vortex_z` = (ω×u)_z = -ω u on the
w-faces, with ω = ∂u/∂z - ∂w/∂x at the vertices, the velocity `(u, w)` on the faces and the density `ρ` at the cells:
a cell array, or a vertical profile (a vector, e.g. the anelastic reference density). The momentum tendencies are
`-vortex_x` and `-vortex_z`. `scheme` is `VortexEnergyCons()` or `VortexEnstrophyCons()` (Sadourny 1975, eqs 3 and 4),
or `VortexEnergyEnstrophyCons()` (Arakawa & Lamb 1981).

    vortex_x, vortex_y, vortex_z, scratch = vortex_force!((vortex_x, vortex_y, vortex_z, scratch), model,
                                                          VortexEnergyCons(), (u, v, w), ρ)

3D vortex force ω×u on the u-, v- and w-faces, with ω = ∇×u on the edges (energy-conserving only: enstrophy is not an
invariant of 3D flow).
"""
function vortex_force!((vortex_x_, vortex_z_, (ρᴵ_, ρᴶ_, η_, Uᴶ_, Wᴵ_)), model, scheme::VortexScheme, (u, w)::NTuple{2}, ρ)
    (; mgr, domain, inv_dx, inv_dz, spatial_scheme) = model
    (; ℑ, δ) = operators(spatial_scheme)
    F = eltype(u)

    η       = alloc!(η_, F, domain, (Face(), Face()))
    Uᴶ      = alloc!(Uᴶ_, F, domain, (Face(), Face()))
    Wᴵ      = alloc!(Wᴵ_, F, domain, (Face(), Face()))
    ρᴵ, ρᴶ  = face_densities!((ρᴵ_, ρᴶ_), model, ρ)

    # vertices: η = ω / ℑXZ(ρ), Uᴶ = ℑZ(ρᴵ u), Wᴵ = ℑX(ρᴶ w)
    @with mgr, let (irange, jrange) = (faces(domain, 1), faces(domain, 2))
        @inbounds @vec for I in irange, J in jrange
            ω = inv_dz * δ(u, (2,), (Face(),), I, J) - inv_dx * δ(w, (1,), (Face(),), I, J)
            η[I, J]  = ω * inv(ℑ(ρ, (1, 2), (Face(), Face()), I, J))
            Uᴶ[I, J] = ℑ((ρᴵ, u), (2,), (Face(),), I, J)
            Wᴵ[I, J] = ℑ((ρᴶ, w), (1,), (Face(),), I, J)
        end
    end
    periodize!(model, (η, Uᴶ, Wᴵ))

    vortex_x, vortex_z = vortex_faces!((vortex_x_, vortex_z_), model, scheme, (η, Uᴶ, Wᴵ))
    return vortex_x, vortex_z, (ρᴵ, ρᴶ, η, Uᴶ, Wᴵ)
end

# Sadourny (1975) eq. 3, energy-conserving: the products at the vertices, interpolated to the faces,
#   vortex_x = ℑz(η Wᴵ),  vortex_z = -ℑx(η Uᴶ)
function vortex_faces!((vortex_x_, vortex_z_), model, ::VortexEnergyCons, (η, Uᴶ, Wᴵ))
    (; mgr, domain, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    F = eltype(η)

    vortex_x = alloc!(vortex_x_, F, domain, (Face(), Cell()))
    vortex_z = alloc!(vortex_z_, F, domain, (Cell(), Face()))

    # u-faces
    @with mgr, let (irange, jrange) = (axes(vortex_x, 1), cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            vortex_x[I, j] = ℑ((η, Wᴵ), (2,), (Cell(),), I, j)
        end
    end
    # w-faces
    @with mgr, let (irange, jrange) = (cells(domain, 1), axes(vortex_z, 2))
        @inbounds @vec for i in irange, J in jrange
            vortex_z[i, J] = -ℑ((η, Uᴶ), (1,), (Cell(),), i, J)
        end
    end
    periodize!(model, (vortex_x, vortex_z))
    return vortex_x, vortex_z
end

# Sadourny (1975) eq. 4, enstrophy-conserving (2D only): η and the mass fluxes interpolated to the faces separately,
# then multiplied,
#   vortex_x = ℑz(η) ℑz(Wᴵ),  vortex_z = -ℑx(η) ℑx(Uᴶ)
function vortex_faces!((vortex_x_, vortex_z_), model, ::VortexEnstrophyCons, (η, Uᴶ, Wᴵ))
    (; mgr, domain, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)
    F = eltype(η)

    vortex_x = alloc!(vortex_x_, F, domain, (Face(), Cell()))
    vortex_z = alloc!(vortex_z_, F, domain, (Cell(), Face()))

    # u-faces
    @with mgr, let (irange, jrange) = (axes(vortex_x, 1), cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            vortex_x[I, j] = ℑ(η, (2,), (Cell(),), I, j) * ℑ(Wᴵ, (2,), (Cell(),), I, j)
        end
    end
    # w-faces
    @with mgr, let (irange, jrange) = (cells(domain, 1), axes(vortex_z, 2))
        @inbounds @vec for i in irange, J in jrange
            vortex_z[i, J] = -ℑ(η, (1,), (Cell(),), i, J) * ℑ(Uᴶ, (1,), (Cell(),), i, J)
        end
    end
    periodize!(model, (vortex_x, vortex_z))
    return vortex_x, vortex_z
end

# Arakawa & Lamb (1981) eqs 3.5, 3.6 and 3.34, energy- and potential-enstrophy-conserving (2D only), with their
# (i, j) → our (I, J) vertices, (i+½, j+½) → (i, j) cells, u* → U, v* → W, q → -η (their ζ is -ω); their rotation
# terms, linear in q, are then -vortex. Each mass flux on a face of a cell is weighted by a combination of the four η
# around that cell:
#   α, β, γ, δ = ηⁱʲ/4 ± δη_xz/48     ηⁱʲ = ℑxz(η), δη_xz = δxδz(η) (the diagonal minus the anti-diagonal pair)
#   ε = Δz/Δx δz ℑx(η) / 12,  φ = -Δx/Δz δx ℑz(η) / 12   (AL's "unphysical Coriolis" terms, U on U, W on W)
# AL's grid is square: on rectangular cells, their scheme holds for the circulations uΔx, wΔz and the transports UΔz,
# WΔx, which leaves the aspect ratio in ε and φ only.
# With Uⁱ = ℑx(U), Wʲ = ℑz(W) and the differences δU_x = δx(U), δW_z = δz(W) at the cells, eqs 3.5 and 3.6 read
#   vortex_x =  ℑX(ηⁱʲ Wʲ) + δX(δη_xz δW_z) / 48 - 2 δX(ε Uⁱ) + U δX(ε)
#   vortex_z = -ℑZ(ηⁱʲ Uⁱ) - δZ(δη_xz δU_x) / 48 - 2 δZ(φ Wʲ) + W δZ(φ)
# The cell values come from ηʲ = ℑz(η) and δη_z = δz(η) on the u-faces. The force does no work, and conserves the
# potential enstrophy Σ ℑXZ(ρ) η²/2 for any mass flux, divergent or not.
function vortex_force!((vortex_x_, vortex_z_, (ρᴵ_, ρᴶ_, η_, ηʲ_, δη_z_, ηⁱʲ_, δη_xz_, ε_, φ_, Uⁱ_, Wʲ_, δU_x_, δW_z_)),
                       model, ::VortexEnergyEnstrophyCons, (u, w)::NTuple{2}, ρ)
    (; mgr, domain, inv_dx, inv_dz, spatial_scheme) = model
    (; ℑ, δ) = operators(spatial_scheme)
    F = eltype(u)

    vortex_x    = alloc!(vortex_x_, F, domain, (Face(), Cell()))
    vortex_z    = alloc!(vortex_z_, F, domain, (Cell(), Face()))
    η           = alloc!(η_, F, domain, (Face(), Face()))
    ηʲ          = alloc!(ηʲ_, F, domain, (Face(), Cell()))
    δη_z        = alloc!(δη_z_, F, domain, (Face(), Cell()))
    ηⁱʲ         = alloc!(ηⁱʲ_, F, domain, (Cell(), Cell()))
    δη_xz       = alloc!(δη_xz_, F, domain, (Cell(), Cell()))
    ε           = alloc!(ε_, F, domain, (Cell(), Cell()))
    φ           = alloc!(φ_, F, domain, (Cell(), Cell()))
    Uⁱ          = alloc!(Uⁱ_, F, domain, (Cell(), Cell()))
    Wʲ          = alloc!(Wʲ_, F, domain, (Cell(), Cell()))
    δU_x        = alloc!(δU_x_, F, domain, (Cell(), Cell()))
    δW_z        = alloc!(δW_z_, F, domain, (Cell(), Cell()))
    ρᴵ, ρᴶ      = face_densities!((ρᴵ_, ρᴶ_), model, ρ)

    # vertices: η = ω / ℑXZ(ρ)
    @with mgr, let (irange, jrange) = (faces(domain, 1), faces(domain, 2))
        @inbounds @vec for I in irange, J in jrange
            ω = inv_dz * δ(u, (2,), (Face(),), I, J) - inv_dx * δ(w, (1,), (Face(),), I, J)
            η[I, J] = ω * inv(ℑ(ρ, (1, 2), (Face(), Face()), I, J))
        end
    end
    periodize!(model, (η,))
    # u-faces: ηʲ = ℑz(η), δη_z = δz(η)
    @with mgr, let (irange, jrange) = (axes(ηʲ, 1), cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            ηʲ[I, j]   = ℑ(η, (2,), (Cell(),), I, j)
            δη_z[I, j] = δ(η, (2,), (Cell(),), I, j)
        end
    end
    # cells: AL's coefficients, and the mass fluxes interpolated and differenced
    @with mgr, let (irange, jrange) = (cells(domain, 1), cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            ηⁱʲ[i, j]   = ℑ(ηʲ, (1,), (Cell(),), i, j)
            δη_xz[i, j] = δ(δη_z, (1,), (Cell(),), i, j)
            ε[i, j]     = inv_dx / inv_dz * ℑ(δη_z, (1,), (Cell(),), i, j) / 12
            φ[i, j]     = -inv_dz / inv_dx * δ(ηʲ, (1,), (Cell(),), i, j) / 12
            Uⁱ[i, j]    = ℑ((ρᴵ, u), (1,), (Cell(),), i, j)
            δU_x[i, j]  = δ((ρᴵ, u), (1,), (Cell(),), i, j)
            Wʲ[i, j]    = ℑ((ρᴶ, w), (2,), (Cell(),), i, j)
            δW_z[i, j]  = δ((ρᴶ, w), (2,), (Cell(),), i, j)
        end
    end
    periodize!(model, (ηⁱʲ, δη_xz, ε, φ, Uⁱ, δU_x, Wʲ, δW_z))

    # u-faces
    @with mgr, let (irange, jrange) = (faces(domain, 1), cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            vortex_x[I, j] = ℑ((ηⁱʲ, Wʲ), (1,), (Face(),), I, j) + δ((δη_xz, δW_z), (1,), (Face(),), I, j) / 48 -
                             2 * δ((ε, Uⁱ), (1,), (Face(),), I, j) + point((ρᴵ, u), I, j) * δ(ε, (1,), (Face(),), I, j)
        end
    end
    # w-faces
    @with mgr, let (irange, jrange) = (cells(domain, 1), faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            vortex_z[i, J] = -ℑ((ηⁱʲ, Uⁱ), (2,), (Face(),), i, J) - δ((δη_xz, δU_x), (2,), (Face(),), i, J) / 48 -
                             2 * δ((φ, Wʲ), (2,), (Face(),), i, J) + point((ρᴶ, w), i, J) * δ(φ, (2,), (Face(),), i, J)
        end
    end
    periodize!(model, (vortex_x, vortex_z))

    return vortex_x, vortex_z, (ρᴵ, ρᴶ, η, ηʲ, δη_z, ηⁱʲ, δη_xz, ε, φ, Uⁱ, Wʲ, δU_x, δW_z)
end

# 3D, Sadourny's energy-conserving products on the edges, interpolated to the faces. Edges: η_x on xYZ (ᴶᴷ), η_y on
# XyZ (ᴵᴷ), η_z on XYz (ᴵᴶ), each with the two mass fluxes U = ρᴵ u, V = ρᴶ v, W = ρᴷ w interpolated to it.
function vortex_force!((vortex_x_, vortex_y_, vortex_z_, (ρᴵ_, ρᴶ_, ρᴷ_, η_x_, η_y_, η_z_, Uᴶ_, Uᴷ_, Vᴵ_, Vᴷ_, Wᴵ_, Wᴶ_)),
                       model, ::VortexEnergyCons, (u, v, w)::NTuple{3}, ρ)
    (; mgr, domain, inv_dx, inv_dy, inv_dz, spatial_scheme) = model
    (; ℑ, δ) = operators(spatial_scheme)
    F = eltype(u)

    vortex_x    = alloc!(vortex_x_, F, domain, (Face(), Cell(), Cell()))
    vortex_y    = alloc!(vortex_y_, F, domain, (Cell(), Face(), Cell()))
    vortex_z    = alloc!(vortex_z_, F, domain, (Cell(), Cell(), Face()))
    η_y         = alloc!(η_y_, F, domain, (Face(), Cell(), Face()))
    Wᴵ          = alloc!(Wᴵ_, F, domain, (Face(), Cell(), Face()))
    Uᴷ          = alloc!(Uᴷ_, F, domain, (Face(), Cell(), Face()))
    η_z         = alloc!(η_z_, F, domain, (Face(), Face(), Cell()))
    Uᴶ          = alloc!(Uᴶ_, F, domain, (Face(), Face(), Cell()))
    Vᴵ          = alloc!(Vᴵ_, F, domain, (Face(), Face(), Cell()))
    η_x         = alloc!(η_x_, F, domain, (Cell(), Face(), Face()))
    Vᴷ          = alloc!(Vᴷ_, F, domain, (Cell(), Face(), Face()))
    Wᴶ          = alloc!(Wᴶ_, F, domain, (Cell(), Face(), Face()))
    ρᴵ, ρᴶ, ρᴷ  = face_densities!((ρᴵ_, ρᴶ_, ρᴷ_), model, ρ)

    # XyZ edges: η_y = (∂u/∂z - ∂w/∂x) / ℑXZ(ρ), Wᴵ = ℑX(ρᴷ w), Uᴷ = ℑZ(ρᴵ u)
    @with mgr, let (irange, jrange, krange) = (faces(domain, 1), axes(η_y, 2), faces(domain, 3))
        for K in krange, j in jrange
            @inbounds @vec for I in irange
                ω_y = inv_dz * δ(u, (3,), (Face(),), I, j, K) - inv_dx * δ(w, (1,), (Face(),), I, j, K)
                η_y[I, j, K] = ω_y * inv(ℑ(ρ, (1, 3), (Face(), Face()), I, j, K))
                Wᴵ[I, j, K]  = ℑ((ρᴷ, w), (1,), (Face(),), I, j, K)
                Uᴷ[I, j, K]  = ℑ((ρᴵ, u), (3,), (Face(),), I, j, K)
            end
        end
    end
    # XYz edges: η_z = (∂v/∂x - ∂u/∂y) / ℑXY(ρ), Uᴶ = ℑY(ρᴵ u), Vᴵ = ℑX(ρᴶ v)
    @with mgr, let (irange, jrange, krange) = (faces(domain, 1), faces(domain, 2), axes(η_z, 3))
        for k in krange, J in jrange
            @inbounds @vec for I in irange
                ω_z = inv_dx * δ(v, (1,), (Face(),), I, J, k) - inv_dy * δ(u, (2,), (Face(),), I, J, k)
                η_z[I, J, k] = ω_z * inv(ℑ(ρ, (1, 2), (Face(), Face()), I, J, k))
                Uᴶ[I, J, k]  = ℑ((ρᴵ, u), (2,), (Face(),), I, J, k)
                Vᴵ[I, J, k]  = ℑ((ρᴶ, v), (1,), (Face(),), I, J, k)
            end
        end
    end
    # xYZ edges: η_x = (∂w/∂y - ∂v/∂z) / ℑYZ(ρ), Vᴷ = ℑZ(ρᴶ v), Wᴶ = ℑY(ρᴷ w)
    @with mgr, let (irange, jrange, krange) = (axes(η_x, 1), faces(domain, 2), faces(domain, 3))
        for K in krange, J in jrange
            @inbounds @vec for i in irange
                ω_x = inv_dy * δ(w, (2,), (Face(),), i, J, K) - inv_dz * δ(v, (3,), (Face(),), i, J, K)
                η_x[i, J, K] = ω_x * inv(ℑ(ρ, (2, 3), (Face(), Face()), i, J, K))
                Vᴷ[i, J, K]  = ℑ((ρᴶ, v), (3,), (Face(),), i, J, K)
                Wᴶ[i, J, K]  = ℑ((ρᴷ, w), (2,), (Face(),), i, J, K)
            end
        end
    end
    periodize!(model, (η_y, Wᴵ, Uᴷ, η_z, Uᴶ, Vᴵ, η_x, Vᴷ, Wᴶ))

    # u-faces: (ω×u)_x = ω_y w - ω_z v = ℑz(η_y Wᴵ) - ℑy(η_z Vᴵ)
    @with mgr, let (irange, jrange, krange) = (faces(domain, 1), cells(domain, 2), cells(domain, 3))
        for k in krange, j in jrange
            @inbounds @vec for I in irange
                vortex_x[I, j, k] = ℑ((η_y, Wᴵ), (3,), (Cell(),), I, j, k) - ℑ((η_z, Vᴵ), (2,), (Cell(),), I, j, k)
            end
        end
    end
    # v-faces: (ω×u)_y = ω_z u - ω_x w = ℑx(η_z Uᴶ) - ℑz(η_x Wᴶ)
    @with mgr, let (irange, jrange, krange) = (cells(domain, 1), faces(domain, 2), cells(domain, 3))
        for k in krange, J in jrange
            @inbounds @vec for i in irange
                vortex_y[i, J, k] = ℑ((η_z, Uᴶ), (1,), (Cell(),), i, J, k) - ℑ((η_x, Wᴶ), (3,), (Cell(),), i, J, k)
            end
        end
    end
    # w-faces: (ω×u)_z = ω_x v - ω_y u = ℑy(η_x Vᴷ) - ℑx(η_y Uᴷ)
    @with mgr, let (irange, jrange, krange) = (cells(domain, 1), cells(domain, 2), faces(domain, 3))
        for K in krange, j in jrange
            @inbounds @vec for i in irange
                vortex_z[i, j, K] = ℑ((η_x, Vᴷ), (2,), (Cell(),), i, j, K) - ℑ((η_y, Uᴷ), (1,), (Cell(),), i, j, K)
            end
        end
    end
    periodize!(model, (vortex_x, vortex_y, vortex_z))

    return vortex_x, vortex_y, vortex_z, (ρᴵ, ρᴶ, ρᴷ, η_x, η_y, η_z, Uᴶ, Uᴷ, Vᴵ, Vᴷ, Wᴵ, Wᴶ)
end

# face densities ρᴵ = ℑX(ρ), ρᴶ = ℑY(ρ) (ℑZ(ρ) in 2D), ρᴷ = ℑZ(ρ), over the full axes along which they are not
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
# ρᴶ = ℑZ(ρ) on the w-faces
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

## schemes for the vortex force ω×u on the C-grid (Sadourny 1975, Arakawa & Lamb 1981); notation in vortex_force.jl

"""
    VortexScheme

Discretisation of the vortex force ω×u: `VortexEnergyCons()`, `VortexEnstrophyCons()` or `VortexEnergyEnstrophyCons()`.
"""
abstract type VortexScheme end

"""
    VortexEnergyCons()

Energy-conserving vortex force (Sadourny 1975, eq. 3), in 2D and 3D: the potential vorticity η = ω/ρ and the mass flux
are multiplied at the vertices (edges in 3D), then interpolated to the faces, e.g. (ω×u)_x = ℑz(η Wᴵ). The force does
no work.
"""
struct VortexEnergyCons <: VortexScheme end

"""
    VortexEnstrophyCons()

Enstrophy-conserving vortex force (Sadourny 1975, eq. 4), in 2D only: η = ω/ρ and the mass flux are interpolated to the
faces separately, then multiplied, e.g. (ω×u)_x = ℑz(η) ℑz(Wᴵ). It conserves the potential enstrophy Σ ρ η²/2 of the
advection, but not the energy.
"""
struct VortexEnstrophyCons <: VortexScheme end

"""
    VortexEnergyEnstrophyCons()

Energy- and potential-enstrophy-conserving vortex force (Arakawa & Lamb 1981, eqs 3.5, 3.6 and 3.34), in 2D only: each
mass flux on a face of a cell is weighted by a combination of the four η = ω/ρ around that cell, with extra terms along
the flux direction ("unphysical Coriolis" terms). The force does no work, and conserves Σ ρ η²/2 for any mass flux.
"""
struct VortexEnergyEnstrophyCons <: VortexScheme end

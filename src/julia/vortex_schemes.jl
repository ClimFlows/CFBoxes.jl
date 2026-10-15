## schemes for the vortex force ω×u on the C-grid

"""
    VortexScheme

Discretisation of the vortex force ω×u: `VortexEnergyCons()` or `VortexEnstrophyCons()`.
"""
abstract type VortexScheme end

"""
    VortexEnergyCons()

Energy-conserving vortex force (Sadourny 1975): ω and the transport are multiplied at the vertices, then interpolated to
the faces.
"""
struct VortexEnergyCons <: VortexScheme end

"""
    VortexEnstrophyCons()

Enstrophy-conserving vortex force (Sadourny 1975): ω/ρ and the mass flux are interpolated separately to the faces, then
multiplied.
"""
struct VortexEnstrophyCons <: VortexScheme end

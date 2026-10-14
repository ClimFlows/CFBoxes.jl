## Boundary conditions on a 2D or 3D box. Each side is either periodic (set by the box topology) or an
## impermeable wall. A wall prescribes the tangential momentum and a condition for each tracer.
## Models apply momentum conditions through ghost velocities, and tracer conditions as fluxes on the wall faces.
## Dimensions are numbered (x, z) in 2D and (x, y, z) in 3D; all routines work along any dimension `d`.

#####
##### Conditions
#####

"""
    FreeSlip()

Zero tangential stress on the wall (same as `NoStress()`).
"""
struct FreeSlip end

"""
    NoStress()

Zero tangential stress on the wall (same as `FreeSlip()`).
"""
const NoStress = FreeSlip

"""
    NoSlip()

Zero tangential velocity on the wall.
"""
struct NoSlip end

"""
    TangentialVelocity(U)

Prescribed tangential velocity of a moving wall (m s⁻¹). In 2D, `U` is along the wall's tangential axis (+x on
bottom/top walls, +z on left/right walls); in 3D, `U = (U₁, U₂)` gives the two tangential components in
increasing axis order. Each value is a number or a function of the coordinates along the wall.
"""
struct TangentialVelocity{V}
    value::V
end

"""
    Stress(τ)

Prescribed tangential traction exerted by the wall on the fluid, τ = μ ∂uₜ/∂n with n the outward normal (N m⁻²
with μ the dynamic viscosity), along the tangential axes as for `TangentialVelocity`. Requires μ ≠ 0.
"""
struct Stress{V}
    value::V
end

"""
    Flux(F)

Prescribed flux of a tracer density ρc into the fluid (units of ρc times m s⁻¹). `F` is a number or a function of
the coordinates along the wall. See also `HeatFlux`.
"""
struct Flux{V, Kind}
    value::V
end
Flux(F) = Flux{typeof(F), :density}(F)

"""
    HeatFlux(F)

Prescribed heat flux `F` into the fluid (W m⁻²), for the conservative variable only (the first tracer). The model
converts it to a flux of ρ consvar, dividing by the conjugate variable of the cell next to the wall, so that the
energy input is exactly `F`.
"""
HeatFlux(F) = Flux{typeof(F), :heat}(F)

"""
    isheat(c)

Whether the tracer condition `c` is a `HeatFlux`.
"""
isheat(::Flux{V, :heat}) where V = true
isheat(_) = false

"""
    Value(c)

Prescribed tracer value `c` on the wall. `c` is a number or a function of the coordinates along the wall.
"""
struct Value{V}
    value::V
end

"""
    Wall(; tangential, tracers)

Impermeable wall with a tangential momentum condition (`FreeSlip()`/`NoStress()`, `NoSlip()`,
`TangentialVelocity(U)`, `Stress(τ)`) and one condition per tracer, `Flux(F)`, `HeatFlux(F)` or `Value(c)`, given
in the order in which the model stores its tracers, e.g. `tracers = (Value(300.0), Flux(0.0))` for (consvar, q).
"""
struct Wall{T, C<:Tuple}
    tangential::T
    tracers::C
end
Wall(; tangential, tracers) = Wall(tangential, Tuple(tracers))

"""
    BoxBoundaryConditions(domain::Box2D; left, right, bottom, top)
    BoxBoundaryConditions(domain::Box3D; left, right, front, back, bottom, top)

Boundary conditions on a box: a `Wall` for each side in a `Bounded` direction; sides in a `Periodic` direction are
left out. Models call `discretize` on it at construction. `walls[d]` holds the (lower, upper) walls along dimension
`d`, `nothing` if periodic.
"""
struct BoxBoundaryConditions{W<:Tuple}
    walls::W
end

function BoxBoundaryConditions(domain::Box2D; left = nothing, right = nothing, bottom = nothing, top = nothing)
    topology = boundary_topology(domain)
    sides = ((left, right), (bottom, top))
    names = ("left/right", "bottom/top")
    walls = map(check_sides, topology, sides, names)    # (lower, upper) walls per direction, nothing if periodic
    return BoxBoundaryConditions(walls)
end

function BoxBoundaryConditions(domain::Box3D; left = nothing, right = nothing, front = nothing, back = nothing,
                               bottom = nothing, top = nothing)
    topology = boundary_topology(domain)
    sides = ((left, right), (front, back), (bottom, top))
    names = ("left/right", "front/back", "bottom/top")
    walls = map(check_sides, topology, sides, names)
    return BoxBoundaryConditions(walls)
end

# Return the (lower, upper) walls of one direction: nothing if periodic, both Walls if bounded; anything else is an error.
check_sides(::Periodic, ::Tuple{Nothing, Nothing}, names) = nothing
check_sides(::Periodic, sides, names) = error("Sides $names are periodic and take no boundary condition.")
check_sides(::Bounded, sides::Tuple{Wall, Wall}, names) = sides
check_sides(::Bounded, sides, names) = error("Sides $names are bounded: give each a Wall(; tangential, tracers).")

"""
    check_tracers(bcs::BoxBoundaryConditions, n)

Check that every wall has exactly `n` tracer conditions, and that `HeatFlux` appears only for the first tracer
(the conservative variable).
"""
check_tracers((; walls)::BoxBoundaryConditions, n) = (foreach(sides -> check_tracers(sides, n), walls); nothing)
check_tracers(::Nothing, n) = nothing
check_tracers(sides::Tuple{Wall, Wall}, n) = foreach(sides) do (; tracers)
    length(tracers) == n || error("Each wall needs $n tracer conditions, got $(length(tracers)).")
    any(isheat, Base.tail(tracers)) && error("HeatFlux is only valid for the first tracer (consvar).")
end

#####
##### Discretization: functions of the coordinates along each wall become arrays, numbers stay numbers
#####

"""
    discretize(bcs::BoxBoundaryConditions, domain, spacing)

Evaluate functions of the coordinates along each wall once, with `spacing = (dx, dz)` or `(dx, dy, dz)`. Tracer
conditions are sampled at the cell centres next to the wall; tangential conditions at the points of the
corresponding velocity component. Arrays are indexed like a slice of the field across the wall.
"""
discretize((; walls)::BoxBoundaryConditions, domain, spacing) =
    BoxBoundaryConditions(ntuple(d -> discretize(walls[d], d, domain, spacing), length(walls)))
discretize(::Nothing, d, domain, spacing) = nothing
discretize(sides::Tuple, d, domain, spacing) = map(sides) do (; tangential, tracers)
    centred = ntuple(_ -> false, length(spacing))
    Wall(discretize(tangential, d, domain, spacing), map(c -> rebuild(c, sample(c.value, d, centred, domain, spacing)), tracers))
end
discretize(t::Union{FreeSlip, NoSlip}, d, domain, spacing) = t
# one value per velocity component (nothing for the normal one), each sampled at that component's points
discretize(t, d, domain, spacing) = rebuild(t, ntuple(length(spacing)) do c
    c == d ? nothing : sample(components(t.value, domain)[c - (c > d)], d, ntuple(==(c), length(spacing)), domain, spacing)
end)

components(value, ::Box2D) = (value,)
components(value, ::Box3D) = Tuple(value)

# the same condition with a new value
rebuild(::Flux{V, K}, value) where {V, K} = Flux{typeof(value), K}(value)
rebuild(c, value) = Base.typename(typeof(c)).wrapper(value)

# sample `f` on the slice across dimension `d` of a field staggered (dual) along the dimensions where `stagger` is true
sample(v::Number, _...) = v
function sample(f, d, stagger, domain, spacing)
    H, others = halo_size(domain), Tuple(e for e in 1:length(spacing) if e != d)
    n = map(e -> (stagger[e] ? dual_size : primal_size)(domain)[e], others)
    coordinate(e, i) = (i - H[e] - (stagger[e] ? 1 : 0.5)) * spacing[e]
    return [f(map(coordinate, others, Tuple(I))...) for I in CartesianIndices(n)]
end

primal_size(domain::Box2D) = (xdim(domain), zdim(domain))
primal_size(domain::Box3D) = (xdim(domain), ydim(domain), zdim(domain))
dual_size(domain::Box2D) = (Xdim(domain), Zdim(domain))
dual_size(domain::Box3D) = (Xdim(domain), Ydim(domain), Zdim(domain))

"""
    bc_value(c, I...)

Value of a discretized condition `c` at slice index `I` along the wall.
"""
@inline bc_value(c, I...) = at(c.value, I...)
@inline at(v::Number, _...) = v
@inline at(v::AbstractArray, I...) = @inbounds v[I...]

#####
##### Halos: one kernel along a dimension, dispatched on topology, staggering and ghost rule
#####

"""
    Centred(), Faced()

Staggering of an array along a dimension: values at cell centres, or on cell faces (where walls lie).
"""
struct Centred end
struct Faced end

# ghost rules: ghost(rule, v, I, dist) is the ghost value mirroring the interior value v across the wall, at
# distance dist; I indexes the position along the wall
"""
    ZeroGradient()

Ghost rule for `fill_halo!`: each ghost equals its mirror image in the fluid, so that the normal gradient (every
difference across a wall) vanishes.
"""
struct ZeroGradient end
struct Odd end                          # zero value on the wall
struct Tangential{c, T, F}              # tangential velocity component c under the wall condition T
    condition::T
    μ::F
end
Tangential{c}(condition, μ) where c = Tangential{c, typeof(condition), typeof(μ)}(condition, μ)

@inline ghost(::Union{ZeroGradient, Tangential{c, FreeSlip} where c}, v, I, dist) = v
@inline ghost(::Union{Odd, Tangential{c, NoSlip} where c}, v, I, dist) = -v
@inline ghost(t::Tangential{c, <:TangentialVelocity}, v, I, dist) where c = 2 * at(t.condition.value[c], I) - v
# traction τ = μ ∂v/∂n (outward normal n) ⇒ v_ghost = v + dist τ / μ
@inline ghost(t::Tangential{c, <:Stress}, v, I, dist) where c = v + dist * at(t.condition.value[c], I) / t.μ

"""
    fill_halo!(topology, a, D::Val, domain, staggering, (lower, upper), Δ)

Fill the halo of `a` along dimension `D`: periodic copy, or ghost values from the rules `lower` and `upper` of
the two walls. Cell-centred ghosts H+1-h mirror cells H+h, at distance (2h-1)Δ; on faces, the wall values
(normal velocity) are set to zero and ghosts H+1-h mirror faces H+1+h, at distance 2hΔ.
"""
function fill_halo!(::Periodic, a, D::Val{d}, domain, staggering, rules, Δ) where d
    M, H = dims(domain)[d], halo_size(domain)[d]
    for I in slice(a, D), h in 1:H
        a[across(I, D, h)] = a[across(I, D, h+M)]
        a[across(I, D, H+M+h)] = a[across(I, D, H+h)]
    end
    return a
end
function fill_halo!(::Bounded, a, D::Val{d}, domain, ::Centred, (lower, upper), Δ) where d
    M, H = dims(domain)[d], halo_size(domain)[d]
    for I in slice(a, D), h in 1:H
        a[across(I, D, H+1-h)] = ghost(lower, a[across(I, D, H+h)], I, (2h-1) * Δ)
        a[across(I, D, H+M+h)] = ghost(upper, a[across(I, D, H+M+1-h)], I, (2h-1) * Δ)
    end
    return a
end
function fill_halo!(::Bounded, a, D::Val{d}, domain, ::Faced, (lower, upper), Δ) where d
    M, H = dims(domain)[d], halo_size(domain)[d]
    for I in slice(a, D)
        a[across(I, D, H+1)] = 0
        a[across(I, D, H+M+1)] = 0
        for h in 1:H
            a[across(I, D, H+1-h)] = ghost(lower, a[across(I, D, H+1+h)], I, 2h * Δ)
            a[across(I, D, H+M+1+h)] = ghost(upper, a[across(I, D, H+M+1-h)], I, 2h * Δ)
        end
    end
    return a
end

"""
    fill_velocity_halo!(bcs, domain, velocity, spacing, dyn_visc)

Fill the halo of the C-grid `velocity = (u, w)` or `(u, v, w)`. Periodic directions are periodized. On walls,
the normal component vanishes and is odd across the wall; tangential components get ghost values such that centred
differences across the wall realize the tangential condition. `bcs` must have been `discretize`d.
"""
function fill_velocity_halo!((; walls)::BoxBoundaryConditions, domain, velocity, spacing, μ)
    foreach_dim(walls) do D
        foreach_dim(velocity) do C
            fill_halo!(pick(boundary_topology(domain), D), pick(velocity, C), D, domain, staggering(C, D),
                       velocity_rules(pick(walls, D), C, D, μ), pick(spacing, D))
        end
    end
    return velocity
end
staggering(::Val{d}, ::Val{d}) where d = Faced()          # normal component
staggering(_, _) = Centred()                              # tangential component
velocity_rules(::Nothing, C, D, μ) = nothing
velocity_rules(::Tuple{Wall, Wall}, ::Val{d}, ::Val{d}, μ) where d = (Odd(), Odd())
velocity_rules((lower, upper)::Tuple{Wall, Wall}, ::Val{c}, D, μ) where c = (Tangential{c}(lower.tangential, μ), Tangential{c}(upper.tangential, μ))

"""
    fill_halo!(a, domain, ZeroGradient())
    fill_halo!(arrays::Tuple, domain, ZeroGradient())

Fill the halo of the cell-centred array `a` (trailing dimensions allowed): periodic copy in periodic directions, and
zero normal gradient across walls (ghosts mirror the fluid cells). Return `a`.
"""
fill_halo!(a::AbstractArray, domain, rule::ZeroGradient) =
    (foreach_dim(D -> fill_halo!(pick(boundary_topology(domain), D), a, D, domain, Centred(), (rule, rule), 1), domain); a)
fill_halo!(arrays::Tuple, domain, rule::ZeroGradient) = (map(a -> fill_halo!(a, domain, rule), arrays); nothing)

"""
    periodize!(model, a)
    periodize!(model, arrays::Tuple)

Fill the halo of `a` (any staggering; trailing dimensions allowed) in the periodic directions of `model.domain`.
Directions bounded by walls are left untouched.
"""
periodize!(model, a::AbstractArray) = (foreach_dim(D -> periodize!(pick(boundary_topology(model.domain), D), a, D, model.domain), model.domain); nothing)
periodize!(model, arrays::Tuple) = (map(a -> periodize!(model, a), arrays); nothing)
periodize!(topology::Periodic, a, D, domain) = fill_halo!(topology, a, D, domain, nothing, nothing, nothing)
periodize!(::Bounded, a, D, domain) = a

"""
    zero_halo!(domain, a, staggering)

Set the halo of `a` to zero, and its values on walls along dimensions where it is `Faced()` (normal components),
`staggering` giving `Centred()` or `Faced()` per dimension.
"""
zero_halo!(domain, a, staggering) =
    (foreach_dim(D -> zero_halo!(pick(boundary_topology(domain), D), a, D, domain, pick(staggering, D)), domain); a)
function zero_halo!(topology, a, D::Val{d}, domain, s) where d
    M, H = dims(domain)[d], halo_size(domain)[d]
    w = wall_width(topology, s)
    for I in slice(a, D)
        for k in 1:H+w
            a[across(I, D, k)] = 0
        end
        for k in H+M+1:H+M+H+w
            a[across(I, D, k)] = 0
        end
    end
    return a
end
wall_width(::Bounded, ::Faced) = 1
wall_width(_, _) = 0

"""
    zero_walls!(domain, a, D::Val)

Set to zero the values of a face array `a` on the walls along dimension `D` (no effect if periodic).
"""
zero_walls!(domain, a, D::Val) = zero_walls!(pick(boundary_topology(domain), D), domain, a, D)
zero_walls!(::Periodic, domain, a, D) = a
function zero_walls!(::Bounded, domain, a, D::Val)
    lower, upper = wall_sides(domain, D)
    for I in slice(a, D)
        a[across(I, D, lower.face)] = 0
        a[across(I, D, upper.face)] = 0
    end
    return a
end

"""
    lower, upper = wall_sides(domain, D::Val)

The two walls across the bounded dimension `D`, each as `(; face, cell, n)`: the index along `D` of the wall faces
and of the adjacent cells, and the normal `n` (+1 lower, -1 upper) pointing into the fluid. Combine with
`slice`/`across` to address the wall along the other dimensions.
"""
wall_sides(domain, ::Val{d}) where d = let M = dims(domain)[d], H = halo_size(domain)[d]
    ((face = H+1, cell = H+1, n = 1), (face = H+M+1, cell = H+M, n = -1))
end

# apply f(Val(d)) for each dimension d of a tuple or a box (unrolled, so that d is known at compile time)
@inline foreach_dim(f, t::Tuple) = (map(f, ntuple(Val, Val(length(t)))); nothing)
@inline foreach_dim(f, domain) = foreach_dim(f, dims(domain))
# element d of a tuple, with d known at compile time
@inline pick(t::Tuple, ::Val{d}) where d = t[d]
# indices of `a` with dimension d removed, and the index with k inserted at dimension d
@inline slice(a, ::Val{d}) where d = CartesianIndices(ntuple(e -> axes(a, e < d ? e : e + 1), Val(ndims(a) - 1)))
@inline across(I::CartesianIndex{N}, ::Val{d}, k) where {N, d} = CartesianIndex(ntuple(e -> e < d ? I[e] : e == d ? k : I[e-1], Val(N + 1)))

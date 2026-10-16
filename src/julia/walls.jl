#=
Boundary conditions on a box. Each side is either periodic (set by the box topology) or an impermeable wall, which
prescribes the tangential momentum and one condition per tracer. Models apply the momentum condition through ghost
velocities (`fill_velocity_halo!`) and the tracer conditions as fluxes on the wall faces.
=#

# Conditions

"""
    FreeSlip()

Zero tangential stress on the wall (also called `NoStress()`).
"""
struct FreeSlip end

"""
    NoStress()

Same as `FreeSlip()`.
"""
const NoStress = FreeSlip

"""
    NoSlip()

Zero tangential velocity on the wall.
"""
struct NoSlip end

"""
    TangentialVelocity(U)

Prescribed tangential velocity `U` [m s⁻¹] of a moving wall. In 2D, `U` is along the wall: +x on the bottom and top
walls, +z on the left and right walls. In 3D, `U = (U₁, U₂)` gives the two tangential components in increasing axis
order. Each value is a number or a function of the coordinates along the wall.
"""
struct TangentialVelocity{V}
    value::V
end

"""
    Stress(τ)

Prescribed tangential stress `τ = μ ∂uₜ/∂n` [N m⁻²] exerted by the wall on the fluid, with `n` the outward normal and
`μ` the dynamic viscosity, which must be nonzero. Components and values as for `TangentialVelocity`.
"""
struct Stress{V}
    value::V
end

"""
    Flux(F)

Prescribed flux `F` of a tracer density ρc into the fluid [ρc m s⁻¹]. `F` is a number or a function of the
coordinates along the wall. See also `HeatFlux`.
"""
struct Flux{V, Kind}
    value::V
end
Flux(F) = Flux{typeof(F), :density}(F)

"""
    HeatFlux(F)

Prescribed heat flux `F` into the fluid [W m⁻²], for the first tracer (the conservative variable) only. The model
divides it by the conjugate variable of the cell next to the wall to get the flux of ρ consvar, so that the energy
input is exactly `F`.
"""
HeatFlux(F) = Flux{typeof(F), :heat}(F)

"""
    isheat(c) -> Bool

Whether the tracer condition `c` is a `HeatFlux`.
"""
isheat(::Flux{V, :heat}) where V = true
isheat(_) = false

"""
    Value(c)

Prescribed tracer value `c` on the wall: a number or a function of the coordinates along the wall.
"""
struct Value{V}
    value::V
end

"""
    Wall(; tangential, tracers)

Impermeable wall with a tangential momentum condition (`FreeSlip()`, `NoSlip()`, `TangentialVelocity(U)` or
`Stress(τ)`) and one condition per tracer (`Flux(F)`, `HeatFlux(F)` or `Value(c)`), in the order in which the model
stores its tracers, e.g. `tracers = (Value(300.0), Flux(0.0))` for (consvar, q).
"""
struct Wall{T, C<:Tuple}
    tangential::T
    tracers::C
end
Wall(; tangential, tracers) = Wall(tangential, Tuple(tracers))

"""
    BoxBoundaryConditions(domain::Box2D; left, right, bottom, top)
    BoxBoundaryConditions(domain::Box3D; left, right, front, back, bottom, top)

A `Wall` on each side of the box in a `Bounded` dimension; sides in a `Periodic` dimension take none. The field
`walls[d]` holds the (lower, upper) walls along dimension `d`, or `nothing` if it is periodic. Models `discretize` it
when they are built.
"""
struct BoxBoundaryConditions{W<:Tuple}
    walls::W
end

function BoxBoundaryConditions(domain::Box2D; left = nothing, right = nothing, bottom = nothing, top = nothing)
    sides = ((left, right), (bottom, top))
    names = ("left/right", "bottom/top")
    return BoxBoundaryConditions(map(check_sides, boundary_topology(domain), sides, names))
end
function BoxBoundaryConditions(domain::Box3D; left = nothing, right = nothing, front = nothing, back = nothing,
                               bottom = nothing, top = nothing)
    sides = ((left, right), (front, back), (bottom, top))
    names = ("left/right", "front/back", "bottom/top")
    return BoxBoundaryConditions(map(check_sides, boundary_topology(domain), sides, names))
end

"""
    check_sides(topology, (lower, upper), names)

The (lower, upper) walls of one dimension: `nothing` if periodic, two `Wall`s if bounded; anything else is an error
naming the sides `names`.
"""
check_sides(::Periodic, ::Tuple{Nothing, Nothing}, names) = nothing
check_sides(::Periodic, sides, names) = error("Sides $names are periodic and take no boundary condition.")
check_sides(::Bounded, sides::Tuple{Wall, Wall}, names) = sides
check_sides(::Bounded, sides, names) = error("Sides $names are bounded: give each a Wall(; tangential, tracers).")

"""
    check_tracers(bcs::BoxBoundaryConditions, n)

Check that every wall has `n` tracer conditions, and a `HeatFlux` only for the first tracer.
"""
check_tracers((; walls)::BoxBoundaryConditions, n) = (foreach(sides -> check_tracers(sides, n), walls); nothing)
check_tracers(::Nothing, n) = nothing
check_tracers(sides::Tuple{Wall, Wall}, n) = foreach(sides) do (; tracers)
    length(tracers) == n || error("Each wall needs $n tracer conditions, got $(length(tracers)).")
    any(isheat, Base.tail(tracers)) && error("HeatFlux is only valid for the first tracer (consvar).")
end

# Discretization: functions of the coordinates along each wall become arrays, numbers stay numbers

"""
    discretize(bcs::BoxBoundaryConditions, domain, spacing) -> BoxBoundaryConditions

Evaluate once the conditions given as functions of the coordinates along each wall, with `spacing = (dx, dz)` or
`(dx, dy, dz)`: tracer conditions at the cell centers next to the wall, tangential conditions at the points of their
velocity component. The arrays are indexed like the slice of a field across the wall (see `bc_value`).
"""
discretize((; walls)::BoxBoundaryConditions, domain, spacing) =
    BoxBoundaryConditions(ntuple(d -> discretize(walls[d], d, domain, spacing), length(walls)))
discretize(::Nothing, d, domain, spacing) = nothing
discretize(sides::Tuple, d, domain, spacing) = map(sides) do (; tangential, tracers)
    centered = ntuple(_ -> false, length(spacing))
    Wall(discretize(tangential, d, domain, spacing), map(c -> rebuild(c, sample(c.value, d, centered, domain, spacing)), tracers))
end
discretize(t::Union{FreeSlip, NoSlip}, d, domain, spacing) = t
# one value per velocity component (nothing for the normal one), each sampled at the points of that component
discretize(t, d, domain, spacing) = rebuild(t, ntuple(length(spacing)) do c
    c == d ? nothing : sample(components(t.value, domain)[c - (c > d)], d, ntuple(==(c), length(spacing)), domain, spacing)
end)

"""
    components(value, box) -> Tuple

The tangential components of the value of a `TangentialVelocity` or `Stress`: one in 2D, two in 3D.
"""
components(value, ::Box2D) = (value,)
components(value, ::Box3D) = Tuple(value)

"""
    rebuild(c, value)

The condition `c` (of the same type, and kind for a `Flux`) with a new value.
"""
rebuild(::Flux{V, K}, value) where {V, K} = Flux{typeof(value), K}(value)
rebuild(c, value) = Base.typename(typeof(c)).wrapper(value)

"""
    sample(f, d, stagger, domain, spacing)

`f` evaluated on the slice across dimension `d` of a field that is on the faces along the dimensions where `stagger`
is true and at the cell centers along the others. A number is returned as it is.
"""
sample(v::Number, _...) = v
function sample(f, d, stagger, domain, spacing)
    others = Tuple(e for e in 1:length(spacing) if e != d)
    n = map(e -> (stagger[e] ? nfaces : ncells)(domain, e), others)
    coordinate(e, i) = (stagger[e] ? facepoint : cellpoint)(domain, spacing[e], e, i)
    return [f(map(coordinate, others, Tuple(I))...) for I in CartesianIndices(n)]
end

"""
    bc_value(c, I...)

Value of the discretized condition `c` at the slice index `I` along the wall.
"""
@inline bc_value(c, I...) = at(c.value, I...)

"""
    at(v, I...)

`v[I...]` for an array, `v` itself for a number.
"""
@inline at(v::Number, _...) = v
@inline at(v::AbstractArray, I...) = @inbounds v[I...]

# Halos: one kernel per dimension, dispatched on topology, staggering and ghost rule

"""
    ZeroGradient()

Ghost rule for `fill_halo!`: each ghost equals its mirror image in the fluid, so that every difference across a wall
vanishes.
"""
struct ZeroGradient end

"""
    Odd()

Ghost rule: each ghost is minus its mirror image, so that the value on the wall is zero.
"""
struct Odd end

"""
    Tangential{c}(condition, μ)

Ghost rule for the tangential velocity component `c` under the wall condition `condition` (`FreeSlip()`, `NoSlip()`,
`TangentialVelocity` or `Stress`), with the dynamic viscosity `μ`.
"""
struct Tangential{c, T, F}
    condition::T
    μ::F
end
Tangential{c}(condition, μ) where c = Tangential{c, typeof(condition), typeof(μ)}(condition, μ)

"""
    ghost(rule, v, I, dist)

Ghost value mirroring the fluid value `v` across the wall, at the distance `dist` from it, under the ghost `rule`;
`I` is the slice index along the wall.
"""
@inline ghost(::Union{ZeroGradient, Tangential{c, FreeSlip} where c}, v, I, dist) = v
@inline ghost(::Union{Odd, Tangential{c, NoSlip} where c}, v, I, dist) = -v
@inline ghost(t::Tangential{c, <:TangentialVelocity}, v, I, dist) where c = 2 * at(t.condition.value[c], I) - v
# stress τ = μ ∂v/∂n along the outward normal n: v_ghost = v + dist τ / μ
@inline ghost(t::Tangential{c, <:Stress}, v, I, dist) where c = v + dist * at(t.condition.value[c], I) / t.μ

#=
Halo of `a` along dimension D: a periodic copy, or ghosts from the rules (lower, upper) of the two walls. At the
cells, ghost H+1-h mirrors cell H+h at distance (2h-1)Δ. On the faces, the walls (normal velocity) are set to zero
and ghost H+1-h mirrors face H+1+h at distance 2hΔ. Likewise at the upper wall.
=#
function fill_halo!(::Periodic, domain, a, D::Val{d}, staggering, rules, Δ) where d
    M, H = dims(domain)[d], halo_size(domain)[d]
    for I in slice(a, D), h in 1:H
        a[across(I, D, h)] = a[across(I, D, h+M)]
        a[across(I, D, H+M+h)] = a[across(I, D, H+h)]
    end
    return a
end
function fill_halo!(::Bounded, domain, a, D::Val{d}, ::Cell, (lower, upper), Δ) where d
    M, H = dims(domain)[d], halo_size(domain)[d]
    for I in slice(a, D), h in 1:H
        a[across(I, D, H+1-h)] = ghost(lower, a[across(I, D, H+h)], I, (2h-1) * Δ)
        a[across(I, D, H+M+h)] = ghost(upper, a[across(I, D, H+M+1-h)], I, (2h-1) * Δ)
    end
    return a
end
function fill_halo!(::Bounded, domain, a, D::Val{d}, ::Face, (lower, upper), Δ) where d
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
    fill_velocity_halo!(bcs, domain, velocity, spacing, μ) -> velocity

Fill the halo of the velocity `(u, w)` or `(u, v, w)`, on the faces normal to each component, with the dynamic
viscosity `μ`. Periodic dimensions are copied. On a wall, the normal component is zero and odd across it, and the
tangential components get ghosts such that the centered differences across the wall realize the tangential condition.
`bcs` must have been `discretize`d.
"""
function fill_velocity_halo!((; walls)::BoxBoundaryConditions, domain, velocity, spacing, μ)
    foreach_dim(walls) do D
        foreach_dim(velocity) do C
            fill_halo!(pick(boundary_topology(domain), D), domain, pick(velocity, C), D, staggering(C, D),
                       velocity_rules(pick(walls, D), C, D, μ), pick(spacing, D))
        end
    end
    return velocity
end
"""
    staggering(C::Val, D::Val)

Staggering of the velocity component `C` along dimension `D`: `Face()` if normal to the faces, else `Cell()`.
"""
staggering(::Val{d}, ::Val{d}) where d = Face()
staggering(_, _) = Cell()

"""
    velocity_rules(walls, C::Val, D::Val, μ)

The ghost rules (lower, upper) of the velocity component `C` on the walls normal to dimension `D`: `Odd()` for the
normal component, `Tangential` for the others, `nothing` if `D` is periodic.
"""
velocity_rules(::Nothing, C, D, μ) = nothing
velocity_rules(::Tuple{Wall, Wall}, ::Val{d}, ::Val{d}, μ) where d = (Odd(), Odd())
velocity_rules((lower, upper)::Tuple{Wall, Wall}, ::Val{c}, D, μ) where c =
    (Tangential{c}(lower.tangential, μ), Tangential{c}(upper.tangential, μ))

"""
    fill_halo!(domain, a, ZeroGradient()) -> a
    fill_halo!(domain, arrays::Tuple, ZeroGradient()) -> arrays

Fill the halo of the cell array `a` (trailing dimensions allowed): a periodic copy along periodic dimensions, and a zero
normal gradient across walls.
"""
fill_halo!(domain::AbstractBox, a::AbstractArray, rule::ZeroGradient) =
    (foreach_dim(D -> fill_halo!(pick(boundary_topology(domain), D), domain, a, D, Cell(), (rule, rule), 1), domain); a)
fill_halo!(domain::AbstractBox, arrays::Tuple, rule::ZeroGradient) = (foreach(a -> fill_halo!(domain, a, rule), arrays); arrays)

"""
    periodize!(domain, a) -> a
    periodize!(domain, arrays::Tuple) -> arrays

Fill the halo of `a` (any grid location; trailing dimensions allowed) along the periodic dimensions of `domain`, and
leave it untouched along bounded ones.
"""
periodize!(domain::AbstractBox, a::AbstractArray) =
    (foreach_dim(D -> periodize!(pick(boundary_topology(domain), D), domain, a, D), domain); a)
periodize!(domain::AbstractBox, arrays::Tuple) = (foreach(a -> periodize!(domain, a), arrays); arrays)
periodize!(topology::Periodic, domain, a, D) = fill_halo!(topology, domain, a, D, nothing, nothing, nothing)
periodize!(::Bounded, domain, a, D) = a

"""
    zero_halo!(domain, a, location::Tuple) -> a

Set to zero the halo of `a` on the grid `location` (e.g. `(Face(), Cell())`; trailing dimensions allowed), and its
values on the walls of the bounded dimensions where it is on the faces (e.g. u on the x walls).
"""
zero_halo!(domain::AbstractBox, a, location::Tuple) =
    (foreach_dim(D -> zero_halo!(pick(boundary_topology(domain), D), domain, a, D, pick(location, D)), domain); a)
function zero_halo!(topology, domain, a, D::Val{d}, s) where d
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
"""
    wall_width(topology, s) -> Int

Number of wall faces on each side of an array with staggering `s` along a dimension: 1 on the faces of a bounded
dimension, else 0.
"""
wall_width(::Bounded, ::Face) = 1
wall_width(_, _) = 0

"""
    zero_walls!(domain, a, D::Val) -> a

Set to zero the values of the face array `a` on the two walls normal to dimension `D` (no effect if it is periodic).
"""
zero_walls!(domain::AbstractBox, a, D::Val) = zero_walls!(pick(boundary_topology(domain), D), domain, a, D)
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

The two walls normal to the bounded dimension `D`, each as `(; face, cell, n)`: the index along `D` of the wall face
and of the adjacent cell, and the normal `n` into the fluid (+1 lower, -1 upper). Combine with `slice` and `across`
to address the wall along the other dimensions.
"""
wall_sides(domain, ::Val{d}) where d = let faces = interior_faces(domain, d), cells = interior_cells(domain, d)
    ((face = first(faces), cell = first(cells), n = 1), (face = last(faces), cell = last(cells), n = -1))
end

"""
    foreach_dim(f, t::Tuple)
    foreach_dim(f, box)

Call `f(Val(d))` for each dimension `d` of a tuple or a box, unrolled so that `d` is known at compile time.
"""
@inline foreach_dim(f, t::Tuple) = (map(f, ntuple(Val, Val(length(t)))); nothing)
@inline foreach_dim(f, domain) = foreach_dim(f, dims(domain))
"""
    pick(t::Tuple, ::Val{d})

Element `d` of the tuple `t`, with `d` known at compile time.
"""
@inline pick(t::Tuple, ::Val{d}) where d = t[d]

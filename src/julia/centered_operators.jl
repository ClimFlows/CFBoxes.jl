## centered average and difference operators for a 2D C-grid

# average operators
"""
    avg_x(arr::AbstractArray{T, 2}, i, j) -> T
    avg_z(arr::AbstractArray{T, 2}, i, j) -> T
    avg_X(arr::AbstractArray{T, 2}, i, j) -> T
    avg_Z(arr::AbstractArray{T, 2}, i, j) -> T

Average of array `arr`in direction specified by `x`, `X`, `z`, `Z`.
"""
Base.@propagate_inbounds avg_x(arr::AbstractArray{T, 2}, i, j) where {T} = 0.5 * ( arr[i+1, j] + arr[i, j] )
Base.@propagate_inbounds avg_z(arr::AbstractArray{T, 2}, i, j) where {T} = 0.5 * ( arr[i, j+1] + arr[i, j] )
Base.@propagate_inbounds avg_X(arr::AbstractArray{T, 2}, i, j) where {T} = 0.5 * ( arr[i, j] + arr[i-1, j] )
Base.@propagate_inbounds avg_Z(arr::AbstractArray{T, 2}, i, j) where {T} = 0.5 * ( arr[i, j] + arr[i, j-1] )

"""
    avg_xz(arr::AbstractArray{T, 2}, i, j) -> T
    avg_xZ(arr::AbstractArray{T, 2}, i, j) -> T
    avg_Xz(arr::AbstractArray{T, 2}, i, j) -> T
    avg_XZ(arr::AbstractArray{T, 2}, i, j) -> T

Double average of array `arr` in directions specified by `x`, `X`, `z`, `Z`.
"""
Base.@propagate_inbounds avg_xz(arr::AbstractArray{T, 2}, i, j) where {T} = 0.25 * ( arr[i, j] +  arr[i, j+1] + arr[i+1, j] + arr[i+1, j+1] )
Base.@propagate_inbounds avg_xZ(arr::AbstractArray{T, 2}, i, j) where {T} = 0.25 * ( arr[i, j] +  arr[i+1, j] + arr[i, j-1] + arr[i+1, j-1] )
Base.@propagate_inbounds avg_Xz(arr::AbstractArray{T, 2}, i, j) where {T} = 0.25 * ( arr[i, j] +  arr[i-1, j] + arr[i, j+1] + arr[i-1, j+1] )
Base.@propagate_inbounds avg_XZ(arr::AbstractArray{T, 2}, i, j) where {T} = 0.25 * ( arr[i, j] +  arr[i-1, j] + arr[i, j-1] + arr[i-1, j-1] )

"""
    avg_x(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) -> T
    avg_z(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) -> T
    avg_X(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) -> T
    avg_Z(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) -> T

Average of product of arrays `arr1` and `arr2` in direction specified by `x`, `X`, `z`, `Z`.
"""
Base.@propagate_inbounds avg_x(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) where {T} = 0.5 * ( arr1[i+1, j] * arr2[i+1, j] + arr1[i, j] * arr2[i, j] )
Base.@propagate_inbounds avg_z(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) where {T} = 0.5 * ( arr1[i, j+1] * arr2[i, j+1] + arr1[i, j] * arr2[i, j] )
Base.@propagate_inbounds avg_X(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) where {T} = 0.5 * ( arr1[i-1, j] * arr2[i-1, j] + arr1[i, j] * arr2[i, j] )
Base.@propagate_inbounds avg_Z(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) where {T} = 0.5 * ( arr1[i, j-1] * arr2[i, j-1] + arr1[i, j] * arr2[i, j] )

""" 
    avg_xz(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) -> T
    avg_XZ(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) -> T

Double average of product of arrays `arr1` and `arr2` in directions specified by `x`, `X`, `z`, `Z`.
"""
Base.@propagate_inbounds avg_xz(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) where {T} = 0.25 * ( arr1[i, j] * arr2[i, j] + arr1[i, j+1] * arr2[i, j+1] + arr1[i+1, j] * arr2[i+1, j] + arr1[i+1, j+1] * arr2[i+1, j+1] )
Base.@propagate_inbounds avg_xZ(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) where {T} = 0.25 * ( arr1[i, j] * arr2[i, j] + arr1[i, j-1] * arr2[i, j-1] + arr1[i+1, j] * arr2[i+1, j] + arr1[i+1, j-1] * arr2[i+1, j-1] )
Base.@propagate_inbounds avg_Xz(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) where {T} = 0.25 * ( arr1[i, j] * arr2[i, j] + arr1[i, j+1] * arr2[i, j+1] + arr1[i-1, j] * arr2[i-1, j] + arr1[i-1, j+1] * arr2[i-1, j+1] )
Base.@propagate_inbounds avg_XZ(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) where {T} = 0.25 * ( arr1[i, j] * arr2[i, j] + arr1[i-1, j] * arr2[i-1, j] + arr1[i, j-1] * arr2[i, j-1] + arr1[i-1, j-1] * arr2[i-1, j-1] )

"""
    avg_x(vec::AbstractArray{T, 1}, i) -> T
    avg_z(vec::AbstractArray{T, 1}, j) -> T
    avg_X(vec::AbstractArray{T, 1}, i) -> T
    avg_Z(vec::AbstractArray{T, 1}, j) -> T

Average of vector `vec` in direction specified by `x`, `X`, `z`, `Z`.
"""
Base.@propagate_inbounds avg_x(vec::AbstractArray{T, 1}, i) where {T} = 0.5 * ( vec[i+1] + vec[i] )
Base.@propagate_inbounds avg_z(vec::AbstractArray{T, 1}, j) where {T} = 0.5 * ( vec[j+1] + vec[j] )
Base.@propagate_inbounds avg_X(vec::AbstractArray{T, 1}, i) where {T} = 0.5 * ( vec[i] + vec[i-1] )
Base.@propagate_inbounds avg_Z(vec::AbstractArray{T, 1}, j) where {T} = 0.5 * ( vec[j] + vec[j-1] )

"""
    avg_x(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) -> T
    avg_z(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) -> T
    avg_X(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) -> T
    avg_Z(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) -> T

Average of product of vector `vec` and array `arr` in direction specified by `x`, `X`, `z`, `Z`.
"""
Base.@propagate_inbounds avg_x(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) where {T} = 0.5 * ( vec[i+1] * arr[i+1, j] + vec[i] * arr[i, j] )
Base.@propagate_inbounds avg_z(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) where {T} = 0.5 * ( vec[j+1] * arr[i, j+1] + vec[j] * arr[i, j] )
Base.@propagate_inbounds avg_X(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) where {T} = 0.5 * ( vec[i] * arr[i, j] + vec[i-1] * arr[i-1, j] )
Base.@propagate_inbounds avg_Z(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) where {T} = 0.5 * ( vec[j] * arr[i, j] + vec[j-1] * arr[i, j-1] )

"""
    avg_xz(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) -> T
    avg_XZ(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) -> T

Double average of product of vector `vec` and array `arr` in directions specified by `x`, `X`, `z`, `Z`.
"""
Base.@propagate_inbounds avg_xz(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) where {T} = 0.25 * ( vec[i] * arr[i, j] + vec[i] * arr[i, j+1] + vec[i+1] * arr[i+1, j] + vec[i+1] * arr[i+1, j+1] )
Base.@propagate_inbounds avg_xZ(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) where {T} = 0.25 * ( vec[i] * arr[i, j] + vec[i] * arr[i, j-1] + vec[i+1] * arr[i+1, j] + vec[i+1] * arr[i+1, j-1] )
Base.@propagate_inbounds avg_Xz(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) where {T} = 0.25 * ( vec[i] * arr[i, j] + vec[i] * arr[i, j+1] + vec[i-1] * arr[i-1, j] + vec[i-1] * arr[i-1, j+1] )
Base.@propagate_inbounds avg_XZ(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) where {T} = 0.25 * ( vec[i] * arr[i, j] + vec[i-1] * arr[i-1, j] + vec[i] * arr[i, j-1] + vec[i-1] * arr[i-1, j-1] )

"""
    avg_x(f::F, arr::AbstractArray{T, 2}, i, j) -> T
    avg_z(f::F, arr::AbstractArray{T, 2}, i, j) -> T
    avg_X(f::F, arr::AbstractArray{T, 2}, i, j) -> T
    avg_Z(f::F, arr::AbstractArray{T, 2}, i, j) -> T

Average of function `f` applied to array `arr` in direction specified by `x`, `X`, `z`, `Z`.
"""
Base.@propagate_inbounds avg_x(f::F, arr::AbstractArray{T, 2}, i, j) where {F<:Function, T} = 0.5 * ( f(arr[i+1, j]) + f(arr[i, j]) )
Base.@propagate_inbounds avg_z(f::F, arr::AbstractArray{T, 2}, i, j) where {F<:Function, T} = 0.5 * ( f(arr[i, j+1]) + f(arr[i, j]) )
Base.@propagate_inbounds avg_X(f::F, arr::AbstractArray{T, 2}, i, j) where {F<:Function, T} = 0.5 * ( f(arr[i, j]) + f(arr[i-1, j]) )
Base.@propagate_inbounds avg_Z(f::F, arr::AbstractArray{T, 2}, i, j) where {F<:Function, T} = 0.5 * ( f(arr[i, j]) + f(arr[i, j-1]) )

"""
    avg_x(f::F, vec::AbstractArray{T, 1}, i) -> T
    avg_z(f::F, vec::AbstractArray{T, 1}, j) -> T
    avg_X(f::F, vec::AbstractArray{T, 1}, i) -> T
    avg_Z(f::F, vec::AbstractArray{T, 1}, j) -> T

Average of function `f` applied to vector `vec` in direction specified by `x`, `X`, `z`, `Z`.
"""
Base.@propagate_inbounds avg_x(f::F, vec::AbstractArray{T, 1}, i) where {T, F<:Function} = 0.5 * ( f(vec[i+1]) + f(vec[i]) )
Base.@propagate_inbounds avg_z(f::F, vec::AbstractArray{T, 1}, j) where {T, F<:Function} = 0.5 * ( f(vec[j+1]) + f(vec[j]) )
Base.@propagate_inbounds avg_Z(f::F, vec::AbstractArray{T, 1}, j) where {T, F<:Function} = 0.5 * ( f(vec[j]) + f(vec[j-1]) )
Base.@propagate_inbounds avg_X(f::F, vec::AbstractArray{T, 1}, i) where {T, F<:Function} = 0.5 * ( f(vec[i]) + f(vec[i-1]) )

# difference operators
"""
    dif_x(arr::AbstractArray{T, 2}, i, j) -> T
    dif_z(arr::AbstractArray{T, 2}, i, j) -> T
    dif_X(arr::AbstractArray{T, 2}, i, j) -> T
    dif_Z(arr::AbstractArray{T, 2}, i, j) -> T

Difference of array `arr` in direction specified by `x`, `X`, `z`, `Z`.
"""
Base.@propagate_inbounds dif_x(arr::AbstractArray{T, 2}, i, j) where {T} = arr[i+1, j] - arr[i, j]
Base.@propagate_inbounds dif_z(arr::AbstractArray{T, 2}, i, j) where {T} = arr[i, j+1] - arr[i, j]
Base.@propagate_inbounds dif_X(arr::AbstractArray{T, 2}, i, j) where {T} = arr[i, j] - arr[i-1, j]
Base.@propagate_inbounds dif_Z(arr::AbstractArray{T, 2}, i, j) where {T} = arr[i, j] - arr[i, j-1]

"""
    dif_x(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) -> T
    dif_z(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) -> T
    dif_X(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) -> T
    dif_Z(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) -> T

Difference of product of arrays `arr1` and `arr2` in direction specified by `x`, `X`, `z`, `Z`.
"""
Base.@propagate_inbounds dif_x(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) where {T} = arr1[i+1, j] * arr2[i+1, j] - arr1[i, j] * arr2[i, j]
Base.@propagate_inbounds dif_z(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) where {T} = arr1[i, j+1] * arr2[i, j+1] - arr1[i, j] * arr2[i, j]
Base.@propagate_inbounds dif_X(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) where {T} = arr1[i, j] * arr2[i, j] - arr1[i-1, j] * arr2[i-1, j]
Base.@propagate_inbounds dif_Z(arr1::AbstractArray{T, 2}, arr2::AbstractArray{T, 2}, i, j) where {T} = arr1[i, j] * arr2[i, j] - arr1[i, j-1] * arr2[i, j-1]

"""
    dif_x(vec::AbstractArray{T, 1}, i) -> T
    dif_z(vec::AbstractArray{T, 1}, j) -> T
    dif_X(vec::AbstractArray{T, 1}, i) -> T
    dif_Z(vec::AbstractArray{T, 1}, j) -> T

Difference of vector `vec` in direction specified by `x`, `X`, `z`, `Z`.
"""
Base.@propagate_inbounds dif_x(vec::AbstractArray{T, 1}, i) where {T} = vec[i+1] - vec[i]
Base.@propagate_inbounds dif_z(vec::AbstractArray{T, 1}, j) where {T} = vec[j+1] - vec[j]
Base.@propagate_inbounds dif_X(vec::AbstractArray{T, 1}, i) where {T} = vec[i] - vec[i-1]
Base.@propagate_inbounds dif_Z(vec::AbstractArray{T, 1}, j) where {T} = vec[j] - vec[j-1]

"""
    dif_x(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) -> T
    dif_z(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) -> T
    dif_X(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) -> T
    dif_Z(vec::AbstractArray{T, 1}, dif_z::AbstractArray{T, 2}, i, j) -> T

Difference of vector `vec` multiplied by array `arr` in direction specified by `x`, `X`, `z`, `Z`.
"""
Base.@propagate_inbounds dif_x(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) where {T} = vec[i+1] * arr[i+1, j] - vec[i] * arr[i, j]
Base.@propagate_inbounds dif_z(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) where {T} = vec[j+1] * arr[i, j+1] - vec[j] * arr[i, j]
Base.@propagate_inbounds dif_X(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) where {T} = vec[i] * arr[i, j] - vec[i-1] * arr[i-1, j]
Base.@propagate_inbounds dif_Z(vec::AbstractArray{T, 1}, arr::AbstractArray{T, 2}, i, j) where {T} = vec[j] * arr[i, j] - vec[j-1] * arr[i, j-1]

"""
    dif_x(f::F, arr::AbstractArray{T, 2}, i, j) -> T
    dif_z(f::F, arr::AbstractArray{T, 2}, i, j) -> T
    dif_X(f::F, arr::AbstractArray{T, 2}, i, j) -> T
    dif_Z(f::F, arr::AbstractArray{T, 2}, i, j) -> T

Difference of function `f` applied to array `arr` in direction specified by `x`, `X`, `z`, `Z`.
"""
Base.@propagate_inbounds dif_x(f::F, arr::AbstractArray{T, 2}, i, j) where {F<:Function, T} = f(arr[i+1, j]) - f(arr[i, j])
Base.@propagate_inbounds dif_z(f::F, arr::AbstractArray{T, 2}, i, j) where {F<:Function, T} = f(arr[i, j+1]) - f(arr[i, j])
Base.@propagate_inbounds dif_X(f::F, arr::AbstractArray{T, 2}, i, j) where {F<:Function, T} = f(arr[i, j]) - f(arr[i-1, j])
Base.@propagate_inbounds dif_Z(f::F, arr::AbstractArray{T, 2}, i, j) where {F<:Function, T} = f(arr[i, j]) - f(arr[i, j-1])
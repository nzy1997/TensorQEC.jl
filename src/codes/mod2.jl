"""
    Mod2 <: Number
    Mod2(x::Bool)

A type representing a binary number in the field of integers modulo 2 (GF(2)).
In this package, Mod2 algebra is used to represent the parity check matrix, and perform Gaussian elimination.

### Fields
- `x::Bool`: the binary number.

### Examples
```jldoctest
julia> Mod2(true) + Mod2(false)
1₂

julia> Mod2(true) + Mod2(true)
0₂
```
"""
struct Mod2 <: Number
    x::Bool
end
Mod2(x::Mod2) = x

# display
Base.show(io::IO, x::Mod2) = print(io, x.x ? "1₂" : "0₂")
Base.show(io::IO, ::MIME"text/plain", x::Mod2) = show(io, x)

# algebra
Base.:+(x::Mod2, y::Mod2) = Mod2(x.x ⊻ y.x)
Base.:-(x::Mod2, y::Mod2) = Mod2(x.x ⊻ y.x)
Base.:-(x::Mod2) = x
Base.:*(x::Mod2, y::Mod2) = Mod2(x.x && y.x)
Base.zero(::Type{Mod2}) = Mod2(false)
Base.one(::Type{Mod2}) = Mod2(true)
Base.iszero(x::Mod2) = !x.x
Base.conj(x::Mod2) = x
Base.oneunit(::Type{Mod2}) = Mod2(true)
Base.inv(x::Mod2) = iszero(x) ? error("Inverse of zero is undefined") : x
Base.:(/)(x::Mod2, y::Mod2) = x * inv(y)
Base.abs(x::Mod2) = x
Base.:(<)(x::Mod2, y::Mod2) = x.x < y.x

# conversion
Base.Int64(x::Mod2) = x.x ? 1 : 0

# using bit storage to speed up the matrix multiplication
function bitmul!(C::AbstractMatrix{Mod2}, A::AbstractMatrix{Mod2}, B::AbstractMatrix{Mod2})
    ca = compresscol(A')
    cb = compresscol(B)
    @inbounds for j in axes(C, 2)
        for i in axes(C, 1)
            n = 0
            for k in axes(ca, 1)
                n += count_ones(ca[k, i] & cb[k, j])
            end
            C[i, j] = Mod2(isodd(n))
        end
    end
    return C
end
function compresscol(A::AbstractMatrix{Mod2})
    m = ceil(Int, size(A, 1) / 64)
    ca = zeros(UInt64, m, size(A, 2))
    for j = axes(A, 2)
        for idx = 1:m
            entry = UInt64(0)
            for k = 0:63
                i = (idx - 1) * 64 + k + 1
                i <= size(A, 1) && A[i, j].x && (entry += UInt(1) << k)
            end
            ca[idx, j] = entry
        end
    end
    return ca
end
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

# conversion
Base.Int64(x::Mod2) = x.x ? 1 : 0
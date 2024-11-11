struct Mod2 <: Number
    x::Bool
end
Base.:+(x::Mod2, y::Mod2) = Mod2(x.x ⊻ y.x)
Base.:-(x::Mod2, y::Mod2) = Mod2(x.x ⊻ y.x)
Base.:-(x::Mod2) = x
Base.:*(x::Mod2, y::Mod2) = Mod2(x.x && y.x)
Base.zero(::Type{Mod2}) = Mod2(false)
Base.one(::Type{Mod2}) = Mod2(true)
Base.show(io::IO, x::Mod2) = print(io, x.x ? "1₂" : "0₂")
Base.show(io::IO, ::MIME"text/plain", x::Mod2) = show(io, x)
Base.iszero(x::Mod2) = !x.x
Base.conj(x::Mod2) = x
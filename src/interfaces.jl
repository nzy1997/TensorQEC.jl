"""
    AbstractDecoder

The abstract type for a decoder.
"""
abstract type AbstractDecoder end

"""
    decode(decoder::AbstractDecoder, tanner::AbstractTannerGraph, syndrome::AbstractSyndrome)

Decode the syndrome using the decoder.
"""
function decode end

struct DecodingResult
    success_tag::Bool
    error_qubits::Vector{Mod2}
end

struct CSSDecodingResult
    success_tag::Bool
    xerror_qubits::Vector{Mod2}
    zerror_qubits::Vector{Mod2}
end

Base.show(io::IO, ::MIME"text/plain", cdr::CSSDecodingResult) = show(io, cdr)
function Base.show(io::IO, cdr::CSSDecodingResult)
    println(io, cdr.success_tag ? "Success" : "Failure")
    if cdr.success_tag
        println(io, "X error:", findall(v->v.x, cdr.xerror_qubits))
        println(io, "Z error:", findall(v->v.x,cdr.zerror_qubits))
    end
end

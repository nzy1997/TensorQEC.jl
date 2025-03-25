"""
    AbstractDecodingProblem

The abstract type for a decoding problem.
"""
abstract type AbstractDecodingProblem end

""" 
    SimpleDecodingProblem(tanner::SimpleTannerGraph, pvec::Vector{Float64})

A simple decoding problem.
Fields:
- `tanner::SimpleTannerGraph`: the Tanner graph
- `pvec::Vector{Float64}`: the independent probability distributions on each bit
"""
struct SimpleDecodingProblem <: AbstractDecodingProblem
    tanner::SimpleTannerGraph
    pvec::Vector{Float64}
end

""" 
    CSSDecodingProblem(tanner::CSSTannerGraph, pvec::Vector{Float64})

A CSS decoding problem.
Fields:
- `tanner::CSSTannerGraph`: the Tanner graph
- `pvec::Vector{DepolarizingError}`: the independent probability distributions on each qubit
"""
struct CSSDecodingProblem <: AbstractDecodingProblem
    tanner::CSSTannerGraph
    pvec::Vector{DepolarizingError}
end

"""
    GeneralDecodingProblem(tanner::SimpleTannerGraph, ptn::TensorNetwork)

A general decoding problem.
Fields:
- `tanner::SimpleTannerGraph`: the Tanner graph
- `ptn::TensorNetwork`: the probability distributions
"""
struct GeneralDecodingProblem <: AbstractDecodingProblem
    tanner::SimpleTannerGraph
    ptn::TensorNetwork # probability distributions
end

struct FlatDecodingProblem
    tanner::SimpleTannerGraph
    code::Vector{Vector{Int}}
    pvec::Vector{Vector{Float64}}
end

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
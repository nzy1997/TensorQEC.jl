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
- `ptn::TensorNetwork`: the probability distributions, qubits are labeled as 1:qubit_num
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
    decode(decoder::AbstractDecoder, problem::AbstractDecodingProblem, syndrome::AbstractSyndrome)
    decode(decoder::AbstractDecoder, problem::AbstractDecodingProblem, syndrome::AbstractSyndrome, p::Float64)

Decode the syndrome using the decoder.
"""
function decode(decoder::AbstractDecoder, tanner::SimpleTannerGraph, syndrome::Vector{Mod2})
    return decode(decoder, tanner, syndrome, fill(0.05, nq(tanner)))
end
function decode(decoder::AbstractDecoder, tanner::SimpleTannerGraph, syndrome::Vector{Mod2}, pvec::Vector{Float64})
    return decode(decoder, SimpleDecodingProblem(tanner,pvec), syndrome)
end
function decode(decoder::AbstractDecoder, tanner::CSSTannerGraph, syndrome::CSSSyndrome,pvec::Vector{DepolarizingError})
    return decode(decoder, CSSDecodingProblem(tanner,pvec), syndrome)
end
function decode(decoder::AbstractDecoder, tanner::CSSTannerGraph, syndrome::CSSSyndrome)
    return decode(decoder, tanner, syndrome, fill(DepolarizingError(0.05, 0.05, 0.05), nq(tanner)))
end

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

function decode(decoder::AbstractDecoder, problem::CSSDecodingProblem, syndrome::CSSSyndrome)
    resz = decode(decoder,problem.tanner.stgx,syndrome.sx,[em.pz + em.py for em in problem.pvec])
    resx = decode(decoder,problem.tanner.stgz,syndrome.sz,[em.px + em.py for em in problem.pvec])
    return CSSDecodingResult(resx.success_tag && resz.success_tag,resx.error_qubits,resz.error_qubits)
end
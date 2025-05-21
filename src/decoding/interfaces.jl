"""
    AbstractDecodingProblem

The abstract type for a decoding problem.
"""
abstract type AbstractDecodingProblem end

""" 
    ClassicalDecodingProblem(tanner::SimpleTannerGraph, pvec::Vector{Float64})

A classical decoding problem.
Fields:
- `tanner::SimpleTannerGraph`: the Tanner graph
- `pvec::Vector{Float64}`: the independent probability distributions on each bit
"""
struct ClassicalDecodingProblem <: AbstractDecodingProblem
    tanner::SimpleTannerGraph
    pvec::Vector{Float64}
end
get_problem(tanner::SimpleTannerGraph,pvec::Vector{Float64}) = ClassicalDecodingProblem(tanner,pvec)
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
get_problem(tanner::CSSTannerGraph,pvec::Vector{DepolarizingError}) = CSSDecodingProblem(tanner,pvec)

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

"""
    AbstractDecoder

The abstract type for a decoder.
"""
abstract type AbstractDecoder end

"""
    CompiledDecoder

Compile the decoder to specific tanner graph and prior probability distributions.
"""
abstract type CompiledDecoder end

function compile(decoder::AbstractDecoder, tanner::AbstractTannerGraph)
    return compile(decoder, tanner, uniform_error_vector(0.05,tanner))
end
function compile(decoder::AbstractDecoder, tanner::AbstractTannerGraph, pvec::Vector)
    return compile(decoder, get_problem(tanner,pvec))
end

"""
    decode(decoder::AbstractDecoder, problem::AbstractDecodingProblem, syndrome::AbstractSyndrome)

Decode the syndrome using the decoder.
"""
function decode(decoder::AbstractDecoder, tanner::AbstractTannerGraph, syndrome::AbstractSyndrome)
    return decode(decoder, tanner, syndrome, uniform_error_vector(0.05,tanner))
end
function decode(decoder::AbstractDecoder, tanner::AbstractTannerGraph, syndrome::AbstractSyndrome, pvec::Vector)
    return decode(decoder, get_problem(tanner,pvec), syndrome)
end
function decode(decoder::AbstractDecoder, problem::AbstractDecodingProblem, syndrome::AbstractSyndrome)
    ct = compile(decoder, problem)
    return decode(ct, syndrome)
end

struct DecodingResult{ET}
    success_tag::Bool
    error_qubits::ET
end

Base.show(io::IO, ::MIME"text/plain", cdr::DecodingResult) = show(io, cdr)
function Base.show(io::IO, cdr::DecodingResult)
    println(io, cdr.success_tag ? "Success" : "Failure")
    println(io, "$(cdr.error_qubits)")
end

function decode(decoder::AbstractDecoder, problem::CSSDecodingProblem, syndrome::CSSSyndrome)
    resz = decode(decoder,problem.tanner.stgx,SimpleSyndrome(syndrome.sx),[em.pz + em.py for em in problem.pvec])
    resx = decode(decoder,problem.tanner.stgz,SimpleSyndrome(syndrome.sz),[em.px + em.py for em in problem.pvec])
    return DecodingResult(resx.success_tag && resz.success_tag,CSSErrorPattern(resx.error_qubits,resz.error_qubits))
end

abstract type AbstractReductionResult end
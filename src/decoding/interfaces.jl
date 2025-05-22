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
get_problem(tanner::SimpleTannerGraph,pvec::IndependentFlipError) = ClassicalDecodingProblem(tanner,pvec.p)
""" 
    IndependentDepolarizingDecodingProblem(tanner::CSSTannerGraph, pvec::IndependentDepolarizingError)

A decoding problem with independent depolarizing error model.
Fields:
- `tanner::CSSTannerGraph`: the Tanner graph
- `pvec::IndependentDepolarizingError`: the independent probability distributions on each qubit
"""
struct IndependentDepolarizingDecodingProblem <: AbstractDecodingProblem
    tanner::CSSTannerGraph
    pvec::IndependentDepolarizingError
end
get_problem(tanner::CSSTannerGraph,pvec::IndependentDepolarizingError) = IndependentDepolarizingDecodingProblem(tanner,pvec)

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
abstract type AbstractClassicalDecoder <: AbstractDecoder end
abstract type AbstractGeneralDecoder <: AbstractDecoder end

"""
    CompiledDecoder

Compile the decoder to specific tanner graph and prior probability distributions.
"""
abstract type CompiledDecoder end

function compile(decoder::AbstractDecoder, tanner::AbstractTannerGraph)
    return compile(decoder, tanner, iid_error(0.05,tanner))
end
function compile(decoder::AbstractDecoder, tanner::AbstractTannerGraph, pvec::AbstractErrorModel)
    return compile(decoder, get_problem(tanner,pvec))
end

"""
    decode(decoder::AbstractDecoder, problem::AbstractDecodingProblem, syndrome::AbstractSyndrome)

Decode the syndrome using the decoder.
"""
function decode(decoder::AbstractDecoder, tanner::AbstractTannerGraph, syndrome::AbstractSyndrome)
    return decode(decoder, tanner, syndrome, iid_error(0.05,tanner))
end
function decode(decoder::AbstractDecoder, tanner::AbstractTannerGraph, syndrome::AbstractSyndrome, pvec::AbstractErrorModel)
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

# function decode(decoder::AbstractDecoder, problem::IndependentDepolarizingDecodingProblem, syndrome::CSSSyndrome)
#     resz = decode(decoder,problem.tanner.stgx,SimpleSyndrome(syndrome.sx),[em.pz + em.py for em in problem.pvec])
#     resx = decode(decoder,problem.tanner.stgz,SimpleSyndrome(syndrome.sz),[em.px + em.py for em in problem.pvec])
#     return DecodingResult(resx.success_tag && resz.success_tag,CSSErrorPattern(resx.error_qubits,resz.error_qubits))
# end

abstract type AbstractReductionResult end

# Compile IndependentDepolarizingDecodingProblem and ClassicalDecodingProblem to GeneralDecodingProblem for a general decoder
struct CompiledGeneralDecoder{CDT,RT} <: CompiledDecoder
    cd::CDT
    reduction::RT
end

function compile(decoder::AbstractGeneralDecoder, iddp::IndependentDepolarizingDecodingProblem)
    gdp, c2g = reduce2general(iddp.tanner,iddp.pvec)
    cd = compile(decoder, gdp)
    return CompiledGeneralDecoder(cd, c2g)
end

function decode(cd::CompiledGeneralDecoder, syndrome::CSSSyndrome)
    return extract_decoding(cd.reduction, decode(cd.cd, SimpleSyndrome([syndrome.sx...,syndrome.sz...])).error_qubits)
end

function compile(decoder::AbstractGeneralDecoder, cdp::ClassicalDecodingProblem)
    gdp = GeneralDecodingProblem(cdp.tanner, TensorNetwork(DynamicEinCode([[i] for i in 1:cdp.tanner.nq],Int[]),[[1-p, p] for p in cdp.pvec]))
    return compile(decoder, gdp)
end

# Compile IndependentDepolarizingDecodingProblem to ClassicalDecodingProblem for a classical decoder
struct CompiledClassicalDecoder{CT1,CT2} <: CompiledDecoder
    ccx::CT1
    ccz::CT2
end

function compile(decoder::AbstractClassicalDecoder, problem::IndependentDepolarizingDecodingProblem)
    qubit_num = nq(problem.tanner)
    cbx = compile(decoder, ClassicalDecodingProblem(problem.tanner.stgx, [problem.pvec.pz[j] + problem.pvec.py[j] for j in 1:qubit_num]))
    cbz = compile(decoder, ClassicalDecodingProblem(problem.tanner.stgz, [problem.pvec.px[j] + problem.pvec.py[j] for j in 1:qubit_num]))
    return CompiledClassicalDecoder(cbx,cbz)
end

function decode(cb::CompiledClassicalDecoder,syndrome::CSSSyndrome)
    bp_resx = decode(cb.ccz,SimpleSyndrome(syndrome.sz))
    bp_resz = decode(cb.ccx,SimpleSyndrome(syndrome.sx))
    return DecodingResult(bp_resx.success_tag && bp_resz.success_tag, CSSErrorPattern(bp_resx.error_qubits,bp_resz.error_qubits))
end
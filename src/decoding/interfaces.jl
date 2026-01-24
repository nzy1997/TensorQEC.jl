"""
    AbstractDecodingProblem

The abstract type for a decoding problem.
"""
abstract type AbstractDecodingProblem end

""" 
    ClassicalDecodingProblem(tanner::SimpleTannerGraph, pvec::IndependentFlipError

A classical decoding problem.
Fields:
- `tanner::SimpleTannerGraph`: the Tanner graph
- `pvec::IndependentFlipError`: the independent probability distributions on each bit
"""
struct ClassicalDecodingProblem <: AbstractDecodingProblem
    tanner::SimpleTannerGraph
    pvec::IndependentFlipError
end

function ClassicalDecodingProblem(tanner::SimpleTannerGraph, pvec::Vector)
    return ClassicalDecodingProblem(tanner, IndependentFlipError(pvec))
end

get_problem(tanner::SimpleTannerGraph,pvec::IndependentFlipError) = ClassicalDecodingProblem(tanner,pvec)
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
    DecodingProblem(code::AbstractCSSCode, error_model::IndependentDepolarizingError)

Construct a decoding problem from a quantum code and an error model.
This is the recommended entry point for the decoding pipeline.

# Example
```julia
code = SteaneCode()
em = iid_error(0.01, 0.01, 0.01, code_n(code))
problem = DecodingProblem(code, em)
compiled = compile(BPDecoder(), problem)

error = random_error_pattern(em)
tanner = CSSTannerGraph(code)
syndrome = syndrome_extraction(error, tanner)
result = decode(compiled, syndrome)
```
"""
function DecodingProblem(code::AbstractCSSCode, error_model::IndependentDepolarizingError)
    tanner = CSSTannerGraph(code)
    return IndependentDepolarizingDecodingProblem(tanner, error_model)
end

"""
    DecodingProblem(code::AbstractCSSCode, error_model::IndependentFlipError)

Construct a classical decoding problem from a quantum code and a flip error model.
Uses the X-stabilizer Tanner graph for decoding.
"""
function DecodingProblem(code::AbstractCSSCode, error_model::IndependentFlipError)
    tanner = CSSTannerGraph(code)
    return ClassicalDecodingProblem(tanner.stgx, error_model)
end

"""
    GeneralDecodingProblem(tanner::SimpleTannerGraph, ptn::SimpleTensorNetwork)

A general decoding problem.
Fields:
- `tanner::SimpleTannerGraph`: the Tanner graph
- `ptn::TensorNetwork`: the probability distributions, qubits are labeled as 1:qubit_num
"""
struct GeneralDecodingProblem <: AbstractDecodingProblem
    tanner::SimpleTannerGraph
    ptn::SimpleTensorNetwork # probability distributions
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

"""
    compile(decoder::AbstractDecoder, problem::AbstractDecodingProblem) -> CompiledDecoder
    compile(decoder::AbstractDecoder, tanner::AbstractTannerGraph)
    compile(decoder::AbstractDecoder, tanner::AbstractTannerGraph, pvec::AbstractErrorModel)

Compile a decoder for a specific decoding problem. Pre-compiling amortizes setup
cost when decoding many syndromes with the same code and error model.

# Arguments
- `decoder::AbstractDecoder`: The decoder algorithm (e.g., `BPDecoder()`, `IPDecoder()`, `TNMAP()`, `TNMMAP()`).
- `problem::AbstractDecodingProblem`: Created via [`DecodingProblem`](@ref).

# Returns
- `CompiledDecoder`: A compiled decoder ready for use with [`decode`](@ref).

# Example
```julia
problem = DecodingProblem(SteaneCode(), iid_error(0.01, 0.01, 0.01, 7))
compiled = compile(BPDecoder(), problem)
# Now use `decode(compiled, syndrome)` repeatedly
```
"""
function compile(decoder::AbstractDecoder, tanner::AbstractTannerGraph)
    return compile(decoder, tanner, iid_error(0.05,tanner))
end
function compile(decoder::AbstractDecoder, tanner::AbstractTannerGraph, pvec::AbstractErrorModel)
    return compile(decoder, get_problem(tanner,pvec))
end

"""
    decode(compiled::CompiledDecoder, syndrome::AbstractSyndrome) -> DecodingResult
    decode(decoder::AbstractDecoder, problem::AbstractDecodingProblem, syndrome::AbstractSyndrome) -> DecodingResult

Decode a syndrome using a pre-compiled decoder (recommended) or by compiling on the fly.

# Arguments
- `compiled::CompiledDecoder`: A decoder compiled for a specific problem via [`compile`](@ref).
- `syndrome::AbstractSyndrome`: The measured syndrome (`SimpleSyndrome` or `CSSSyndrome`).

# Returns
- `DecodingResult`: Contains `success_tag::Bool` indicating whether decoding succeeded,
  and `error_pattern` with the estimated error.

# Example
```julia
code = SteaneCode()
em = iid_error(0.01, 0.01, 0.01, code_n(code))
problem = DecodingProblem(code, em)
compiled = compile(BPDecoder(), problem)

error = random_error_pattern(em)
tanner = CSSTannerGraph(code)
syndrome = syndrome_extraction(error, tanner)
result = decode(compiled, syndrome)
result.success_tag  # true if decoding succeeded
```
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

"""
    DecodingResult(success_tag::Bool, error_pattern::ET)

The result of decoding.

### Fields:
- `success_tag::Bool`: whether the decoding is successful.
- `error_pattern::ET`: the error pattern.
"""
struct DecodingResult{ET}
    success_tag::Bool
    error_pattern::ET
end

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
    return extract_decoding(cd.reduction, decode(cd.cd, SimpleSyndrome([syndrome.sx...,syndrome.sz...])).error_pattern)
end

function compile(decoder::AbstractGeneralDecoder, cdp::ClassicalDecodingProblem)
    gdp = GeneralDecodingProblem(cdp.tanner, SimpleTensorNetwork(DynamicEinCode([[i] for i in 1:cdp.tanner.nq],Int[]),[[1-p, p] for p in cdp.pvec.p]))
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
    return DecodingResult(bp_resx.success_tag && bp_resz.success_tag, CSSErrorPattern(bp_resx.error_pattern,bp_resz.error_pattern))
end
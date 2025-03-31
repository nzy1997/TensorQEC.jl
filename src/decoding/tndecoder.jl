struct TNMAP <: AbstractDecoder end
abstract type CompiledTN <: CompiledDecoder end

struct CompiledTNMAP{ET, FT} <: CompiledTN
    uaimodel::UAIModel{ET, FT}
    qubit_num::Int
end
extract_decoding(ct::CompiledTNMAP, error_qubits::Vector{Int}) = DecodingResult(true, Mod2.(error_qubits[1:ct.qubit_num]))
struct CompiledTNMAPCSS{ET, FT} <: CompiledTN
    uaimodel::UAIModel{ET, FT}
    qubit_num::Int
    reduction::CSSToGeneralDecodingProblem
end
extract_decoding(ct::CompiledTNMAPCSS, error_qubits::Vector{Int}) = extract_decoding(ct.reduction,Mod2.(error_qubits .== 1))

function stg2uaimodel(tanner::SimpleTannerGraph, ptn::TensorNetwork)
    nvars = tanner.nq + tanner.ns 
    cards = fill(2, nvars)

    factors = [Factor(((tanner.s2q[s] ∪ (s+tanner.nq))...,), parity_check_matrix(length(tanner.s2q[s]))) for s in 1:(tanner.ns)]
    pfac = [Factor((c...,),t) for (c,t) in zip(ptn.code.ixs,ptn.tensors)]
    return UAIModel(nvars, cards, factors ∪ pfac)
end

function parity_check_matrix(qubit_num::Int)
    return reshape([Float64(1- count_ones(i-1)%2) for i in 1:2^(qubit_num+1)], (fill(2,qubit_num+1)...))
end

function compile(decoder::TNMAP, problem::SimpleDecodingProblem)
    code = DynamicEinCode([[i] for i in 1:problem.tanner.nq],Int[])
    tensors= [[1-p,p] for p in problem.pvec]
    ptn = TensorNetwork(code,tensors)
    return CompiledTNMAP(stg2uaimodel(problem.tanner, ptn),problem.tanner.nq)
end
function decode(ct::CompiledTN,syndrome::CSSSyndrome)
    decode(ct::CompiledTN,SimpleSyndrome([syndrome.sx...,syndrome.sz...]))
end
function decode(ct::CompiledTN,syndrome::SimpleSyndrome)
    tn = TensorNetworkModel(ct.uaimodel,evidence=Dict([(i+ct.qubit_num,s.x ? 1 : 0) for (i,s) in enumerate(syndrome.s)]))
    _,config = most_probable_config(tn)
    return extract_decoding(ct, config)
end

function compile(decoder::TNMAP, problem::GeneralDecodingProblem)
    return CompiledTNMAP(stg2uaimodel(problem.tanner, problem.ptn),problem.tanner.nq)
end

function compile(decoder::TNMAP, problem::CSSDecodingProblem)
    c2g = reduce2general(problem.tanner,[[p.px,p.py,p.pz] for p in problem.pvec])
    return CompiledTNMAPCSS(stg2uaimodel(c2g.gdp.tanner, c2g.gdp.ptn),c2g.gdp.tanner.nq,c2g)
end
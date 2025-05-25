"""
    TNMAP(;optimizer::CodeOptimizer=default_optimizer()) <: AbstractGeneralDecoder

A tensor network based maximum a posteriori (MAP) decoder, which finds the most probable configuration of the error pattern.

### Keyword Arguments
- `optimizer::CodeOptimizer = TreeSA()`: The optimizer to use for optimizing the tensor network contraction order.
"""
Base.@kwdef struct TNMAP <: AbstractGeneralDecoder
    optimizer::CodeOptimizer = TreeSA()  # contraction order optimizer
end

struct CompiledTNMAP{ET, FT} <: CompiledDecoder
    net::TensorNetworkModel{ET, FT}   # The tensor network model
    qubit_num::Int                    # The number of qubits in the Tanner graph
end

"""
    stg2uaimodel(tanner::SimpleTannerGraph, ptn::TensorNetwork)

Convert a Tanner graph and a tensor network to a UAIModel.

### Arguments
- `tanner::SimpleTannerGraph`: The Tanner graph.
- `ptn::TensorNetwork`: The tensor network.

### Returns
- `uai::UAIModel`: The UAIModel.
"""
function stg2uaimodel(tanner::SimpleTannerGraph, ptn::TensorNetwork)
    nvars = tanner.nq + tanner.ns 
    cards = fill(2, nvars)

    factors = [Factor(((tanner.s2q[s] ∪ (s+tanner.nq))...,), parity_check_matrix(length(tanner.s2q[s]))) for s in 1:(tanner.ns)]
    pfac = [Factor((c...,),t) for (c,t) in zip(ptn.code.ixs, ptn.tensors)]
    return UAIModel(nvars, cards, factors ∪ pfac)
end

function parity_check_matrix(qubit_num::Int)
    return reshape([Float64(1- count_ones(i-1)%2) for i in 1:2^(qubit_num+1)], (fill(2,qubit_num+1)...))
end

function compile(decoder::TNMAP, problem::GeneralDecodingProblem)
    uai = stg2uaimodel(problem.tanner, problem.ptn)
    tn = TensorNetworkModel(uai; evidence=Dict([i+problem.tanner.nq => 0 for i in 1:problem.tanner.ns]), optimizer=decoder.optimizer)
    return CompiledTNMAP(tn, problem.tanner.nq)
end

# NOTE: decode will change the evidence inplace! But it is safe to use it multiple times.
function decode(ct::CompiledTNMAP, syndrome::SimpleSyndrome)
    TensorInference.update_evidence!(ct.net, Dict([(i+ct.qubit_num,s.x ? 1 : 0) for (i,s) in enumerate(syndrome.s)]))
    _, config = most_probable_config(ct.net)
    return DecodingResult(true, Mod2.(config[1:ct.qubit_num]))
end
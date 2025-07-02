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

"""
    TNMMAP(;optimizer::CodeOptimizer=default_optimizer()) <: AbstractGeneralDecoder

A tensor network based marginal maximum a posteriori (MMAP) decoder, which finds the most probable logical sector after marginalizing out the error pattern on qubits.

### Keyword Arguments
- `optimizer::CodeOptimizer = TreeSA()`: The optimizer to use for optimizing the tensor network contraction order.
"""
Base.@kwdef struct TNMMAP <: AbstractGeneralDecoder
    optimizer::CodeOptimizer = TreeSA()  # contraction order optimizer
end

struct CompiledTNMMAP{LT, AT} <: CompiledDecoder
    mmap::MMAPModel{LT, AT}   # The tensor network model
    tanner::CSSTannerGraph
    lx::Matrix{Mod2}
    lz::Matrix{Mod2}
end

# nvars is the number of variables in the tensor network. 
# |<--- xerror --->|<--- zerror --->|<--- xsyndromes --->|<--- zsyndromes --->|<--- logical x --->|<--- logical z --->|
# |<------------------------------------------------------- nvars --------------------------------------------------->|
function compile(decoder::TNMMAP, problem::IndependentDepolarizingDecodingProblem)
    tanner = problem.tanner
    qubit_num = nq(tanner)
    lx,lz = logical_operator(tanner)
    lxs = [findall(j->j.x,lx[i,:]) for i in axes(lx, 1)]
    lzs = [findall(j->j.x,lz[i,:]) for i in axes(lz, 1)]

    nvars = 2 * qubit_num + ns(tanner) + size(lx,1) + size(lz,1)
    # mars = [[i for i in 2 * qubit_num + ns(tanner)+1:nvars]]
    cards = fill(2, nvars)

    xfactors = [Factor(((c.+ qubit_num)...,i + 2 * qubit_num),parity_check_matrix(length(c))) for (i,c) in enumerate(tanner.stgx.s2q)]
    zfactors = [Factor((c...,i + 2 * qubit_num + ns(tanner.stgx)),parity_check_matrix(length(c))) for (i,c) in enumerate(tanner.stgz.s2q)]

    pfac = [Factor((i,i+qubit_num),[1-px-py-pz pz;px py]) for (i,(px,py,pz)) in enumerate(zip(problem.pvec.px,problem.pvec.py,problem.pvec.pz))]
    
    logicalx_checked_by_lz = [Factor((ilz...,i+2 * qubit_num + ns(tanner) + size(lz,1)),parity_check_matrix(length(ilz))) for (i,ilz) in enumerate(lzs)]

    logicalz_checked_by_lx = [Factor(((ilx.+ qubit_num)...,i+2 * qubit_num + ns(tanner)),parity_check_matrix(length(ilx))) for (i,ilx) in enumerate(lxs)]
    factors = vcat(xfactors,zfactors,pfac,logicalx_checked_by_lz,logicalz_checked_by_lx)

    uai = UAIModel(nvars, cards, factors)

    mmap = MMAPModel(uai; evidence = Dict([i+2*qubit_num => 0 for i in 1:ns(tanner)]) , queryvars = [i for i in 2 * qubit_num + ns(tanner)+1:nvars],optimizer = decoder.optimizer)

    return CompiledTNMMAP(mmap, tanner, lx, lz)
end

function decode(ct::CompiledTNMMAP, syndrome::CSSSyndrome)
    qubit_num = nq(ct.tanner)
    for (i,s) in enumerate(syndrome.sx)
        ct.mmap.evidence[i+2*qubit_num] = s.x ? 1 : 0
    end
    for (i,s) in enumerate(syndrome.sz)
        ct.mmap.evidence[i+2*qubit_num+length(syndrome.sx)] = s.x ? 1 : 0
    end
    _, config = most_probable_config(ct.mmap)
    ex,ez = _mixed_integer_programming_for_one_solution(ct.tanner, syndrome)
    for i in axes(ct.lx,1)
        (sum(ct.lz[i,:].* ex).x == (config[ns(ct.tanner)+i+size(ct.lx,1)]  == 1)) || (ex += ct.lx[i,:])
        (sum(ct.lx[i,:].* ez).x == (config[ns(ct.tanner)+i] == 1)) || (ez += ct.lz[i,:])
    end
    return DecodingResult(true, CSSErrorPattern(ex,ez))
end
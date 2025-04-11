@kwdef struct TNMAP <: AbstractDecoder 
    usecuda::Bool = false
end

abstract type CompiledTN <: CompiledDecoder end

struct CompiledTNMAP{ET, FT} <: CompiledTN
    uaimodel::UAIModel{ET, FT}
    qubit_num::Int
end
extract_decoding(ct::CompiledTNMAP, error_qubits::Vector{Int}) = DecodingResult(true, Mod2.(error_qubits[1:ct.qubit_num]))
struct CompiledTNMAPCSS{LT, ET, MT} <: CompiledTN
    tn::TensorNetworkModel{LT, ET, MT}
    tanner::CSSTannerGraph
    lx::Matrix{Mod2}
    lz::Matrix{Mod2}
    usecuda::Bool
end
# extract_decoding(ct::CompiledTNMAPCSS, error_qubits::Vector{Int}) = extract_decoding(ct.reduction,Mod2.(error_qubits .== 1))

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

function decode(ct::CompiledTN,syndrome::SimpleSyndrome)
    tn = TensorNetworkModel(ct.uaimodel,evidence=Dict([(i+ct.qubit_num,s.x ? 1 : 0) for (i,s) in enumerate(syndrome.s)]))
    _,config = most_probable_config(tn)
    return extract_decoding(ct, config)
end

function compile(decoder::TNMAP, problem::GeneralDecodingProblem)
    return CompiledTNMAP(stg2uaimodel(problem.tanner, problem.ptn),problem.tanner.nq)
end

# nvars is the number of variables in the tensor network. 
# |<--- xerror --->|<--- zerror --->|<--- xsyndromes --->|<--- zsyndromes --->|<--- logical x --->|<--- logical z --->|
function compile(decoder::TNMAP, problem::CSSDecodingProblem)
    tanner = problem.tanner
    qubit_num = nq(tanner)
    lx,lz = logical_operator(tanner)
    lxs = [findall(j->j.x,lx[i,:]) for i in axes(lx, 1)]
    lzs = [findall(j->j.x,lz[i,:]) for i in axes(lz, 1)]

    nvars = 2 * qubit_num + ns(tanner) + size(lx,1) + size(lz,1)
    mars = [[i] for i in 2 * qubit_num + ns(tanner)+1:nvars]
    cards = fill(2, nvars)

    xfactors = [Factor(((c.+ qubit_num)...,i + 2 * qubit_num),parity_check_matrix(length(c))) for (i,c) in enumerate(tanner.stgx.s2q)]
    zfactors = [Factor((c...,i + 2 * qubit_num + ns(tanner.stgx)),parity_check_matrix(length(c))) for (i,c) in enumerate(tanner.stgz.s2q)]

    pfac = [Factor((i,i+qubit_num),[1-p.px-p.py-p.pz p.pz;p.px p.py]) for (i,p) in enumerate(problem.pvec)]
    
    logicalx_checked_by_lz = [Factor((lz...,i+2 * qubit_num + ns(tanner)),parity_check_matrix(length(lz))) for (i,lz) in enumerate(lzs)]

    logicalz_checked_by_lx = [Factor(((lx.+ qubit_num)...,i+2 * qubit_num + ns(tanner) + size(lz,1)),parity_check_matrix(length(lx))) for (i,lx) in enumerate(lxs)]
    factors = vcat(xfactors,zfactors,pfac,logicalx_checked_by_lz,logicalz_checked_by_lx)

    tn = TensorNetworkModel(1:nvars,cards,factors;mars=mars,evidence=Dict([(i+2*qubit_num, 0) for (i) in 1:ns(tanner)]))

    return CompiledTNMAPCSS(tn,tanner,lx,lz,decoder.usecuda)
end

function decode(ct::CompiledTNMAPCSS,syn::CSSSyndrome)
    tn = ct.tn
    for (i,s) in enumerate(syn.sx)
        tn.evidence[(i+2*nq(ct.tanner))] = s.x ? 1 : 0
    end
    for (i,s) in enumerate(syn.sz)
        tn.evidence[(i+2*nq(ct.tanner))+length(syn.sx)] = s.x ? 1 : 0
    end
    start_var = ns(ct.tanner) + 2 * nq(ct.tanner)
    ex,ez = _mixed_integer_programming_for_one_solution(ct.tanner, syn)
    # @show contraction_complexity(tn)
    # return
    mar = marginals(tn; usecuda=ct.usecuda)
    for i in axes(ct.lx,1)
        if ct.usecuda
            (sum(ct.lz[i,:].* ex).x == (Array(mar[[start_var+i]])[2] > 0.5)) || (ex += ct.lx[i,:])
            (sum(ct.lx[i,:].* ez).x == (Array(mar[[start_var+i+size(ct.lx,1)]])[2] > 0.5)) || (ez += ct.lz[i,:])
        else
            (sum(ct.lz[i,:].* ex).x == (mar[[start_var+i]][2] > 0.5)) || (ex += ct.lx[i,:])
            (sum(ct.lx[i,:].* ez).x == (mar[[start_var+i+size(ct.lx,1)]][2] > 0.5)) || (ez += ct.lz[i,:])
        end
    end
    return CSSDecodingResult(true,CSSErrorPattern(ex,ez))
end
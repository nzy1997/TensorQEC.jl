"""
    TNMAP(;optimizer::CodeOptimizer=default_optimizer()) <: AbstractGeneralDecoder

A tensor network based maximum a posteriori (MAP) decoder, which finds the most probable configuration of the error pattern.

### Keyword Arguments
- `optimizer::CodeOptimizer = TreeSA()`: The optimizer to use for optimizing the tensor network contraction order.
"""
Base.@kwdef struct TNMAP <: AbstractGeneralDecoder
    optimizer::CodeOptimizer = TreeSA()  # contraction order optimizer
end

Base.show(io::IO, ::MIME"text/plain", p::TNMAP) = show(io, p)
Base.show(io::IO, p::TNMAP) = print(io, "TNMAP")

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

Base.show(io::IO, ::MIME"text/plain", p::TNMMAP) = show(io, p)
Base.show(io::IO, p::TNMMAP) = print(io, "TNMMAP")

struct CompiledTNMMAP{CT, AT} <: CompiledDecoder
    tanner::CSSTannerGraph
    lx::Matrix{Mod2}
    lz::Matrix{Mod2}
    optcode::CT
    tensors::Vector{AT}
    syndrome_indices::Vector{Int}
    zero_tensor::AT
    one_tensor::AT
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

    ixs = Vector{Vector{Int64}}()
    tensors = Vector{Array{Float64}}()
    for (i,c) in enumerate(tanner.stgx.s2q)
        push!(ixs, [(c.+ qubit_num)...,i + 2 * qubit_num])
        push!(tensors, parity_check_matrix(length(c)))
    end

    for (i,c) in enumerate(tanner.stgz.s2q)
        push!(ixs, [c...,i + 2 * qubit_num + ns(tanner.stgx)])
        push!(tensors, parity_check_matrix(length(c)))
    end

    for (i,(px,py,pz)) in enumerate(zip(problem.pvec.px,problem.pvec.py,problem.pvec.pz))
        push!(ixs, [i,i+qubit_num])
        push!(tensors, [1-px-py-pz pz;px py])
    end

    for (i,ilz) in enumerate(lzs)
        push!(ixs, [ilz...,i + 2 * qubit_num + ns(tanner)+ size(lz,1)])
        push!(tensors, parity_check_matrix(length(ilz)))
    end

    for (i,ilx) in enumerate(lxs)
        push!(ixs, [(ilx.+ qubit_num)...,i + 2 * qubit_num + ns(tanner)])
        push!(tensors, parity_check_matrix(length(ilx)))
    end

    syndrome_indices = collect(2*qubit_num+1:2*qubit_num+ns(tanner))
    iy = collect(2 * qubit_num + ns(tanner) +1:nvars)
   
    zero_tensor = [1.0, 0.0]
    one_tensor = [0.0, 1.0]

    for v in syndrome_indices
        push!(ixs, [v])
        push!(tensors, zero_tensor)
    end
    code = DynamicEinCode(ixs,iy)
    size_dict = uniformsize(code, 2)
    optcode = optimize_code(code, size_dict, decoder.optimizer)
    return CompiledTNMMAP(tanner, lx, lz, optcode, tensors, syndrome_indices, zero_tensor, one_tensor)
end

function update_syndrome!(tensors::Vector{AT}, syndrome::CSSSyndrome, zero_tensor::AT,one_tensor::AT) where AT
    num_sx = length(syndrome.sx)
    num_sz = length(syndrome.sz)
    for (i,s) in enumerate(syndrome.sx)
        tensors[end-num_sx-num_sz+i] = s.x ? one_tensor : zero_tensor
    end
    for (i,s) in enumerate(syndrome.sz)
        tensors[end-num_sz+i] = s.x ? one_tensor : zero_tensor
    end
    return tensors
end

function decode(ct::CompiledTNMMAP, syndrome::CSSSyndrome)
    update_syndrome!(ct.tensors, syndrome, ct.zero_tensor, ct.one_tensor)
    mar = ct.optcode(ct.tensors...)
    ex,ez = _mixed_integer_programming_for_one_solution(ct.tanner, syndrome)
    _, pos = findmax(mar)
    for i in axes(ct.lx,1)
        (sum(ct.lz[i,:].* ex).x == (pos.I[i+size(ct.lx,1)] == 2)) || (ex += ct.lx[i,:])
        (sum(ct.lx[i,:].* ez).x == (pos.I[i] == 2)) || (ez += ct.lz[i,:])
    end
    return DecodingResult(true, CSSErrorPattern(ex,ez))
end
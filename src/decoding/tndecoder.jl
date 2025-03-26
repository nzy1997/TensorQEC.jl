struct TNOSD <: AbstractDecoder end

function decode(decoder::TNOSD, tanner::SimpleTannerGraph, syndrome::Vector{Mod2}, p::Float64)
    eqs = tensor_osd(syndrome, tanner, p)
    return DecodingResult(true, eqs)
end

function stg2tensornetwork(tanner::SimpleTannerGraph, ptn::TensorNetwork)
    tensors = Vector{Array{Float64}}()
    ixs = copy(ptn.code.ixs)

    [push!(tensors, ptn.tensors[i]) for i in 1:length(ptn.tensors)]
    for s in 1:tanner.ns
        push!(tensors, parity_check_matrix(length(tanner.s2q[s])))
        push!(ixs, [s+tanner.nq,tanner.s2q[s]...])
    end
    return TensorNetwork( DynamicEinCode(ixs,Int[]),tensors)
end

function syndrome_tn(tn, syndrome)
    for i in 1:length(syndrome)
        tn.tensors[tanner.nq+i] = syndrome[i].x ? [1.0, 0.0] : [0.0, 1.0]
    end
    return tn
    
end

function ldpc2tensor(tanner::SimpleTannerGraph, p::Float64, syndrome::Vector{Mod2})
    nvars = tanner.nq + tanner.ns 
    cards = fill(2, nvars)
    factors = [Factor(((tanner.s2q[s] ∪ (s+tanner.nq))...,), parity_check_matrix(length(tanner.s2q[s]))) for s in 1:(tanner.ns)]
    pfac = [Factor((q,), [1-p, p]) for q in 1:tanner.nq]
    sfac = [Factor((s+tanner.nq,), syndrome[s].x ? [1.0, 0.0] : [0.0, 1.0 ]) for s in 1:tanner.ns]
    return TensorNetworkModel(
		1:nvars,
		cards,
		factors ∪ pfac ∪ sfac;
		openvars = [],
		mars = [[i] for i in 1:tanner.nq],
	)
end

function parity_check_matrix(qubit_num::Int)
    return reshape([Float64(count_ones(i-1)%2) for i in 1:2^(qubit_num+1)], (fill(2,qubit_num+1)...))
end

function tensor_infer(tanner::SimpleTannerGraph, p::Float64, syndrome::Vector{Mod2})
    tn = ldpc2tensor(tanner, p, syndrome)
    mp = marginals(tn)
    return [ mp[[k]][2] for k in 1:tanner.nq]
end

function tensor_osd(syndrome::Vector{Mod2}, tanner::SimpleTannerGraph, p::Float64)
    error_p = tensor_infer(tanner, p, syndrome)
    return osd(tanner, sortperm(-error_p), syndrome)
end
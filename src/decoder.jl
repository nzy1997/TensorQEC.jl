abstract type AbstractDecoder end
struct BPDecoder <: AbstractDecoder
    bp_max_iter::Int
end

struct BPOSD <: AbstractDecoder
    bp_max_iter::Int
end

struct DecodingResult
    success_tag::Bool
    error_qubits::Vector{Mod2}
end
struct IPDecoder <: AbstractDecoder end

function decode(decoder::IPDecoder, tanner::SimpleTannerGraph, p::Float64, sydrome::Vector{Mod2})
    return decode(decoder,tanner,sydrome)
end
function decode(decoder::IPDecoder, tanner::SimpleTannerGraph, sydrome::Vector{Mod2};verbose = false)
    H = [a.x for a in tanner.H]
    m,n = size(H)
    model = Model(HiGHS.Optimizer)
    !verbose && set_silent(model)

    @variable(model, 0 <= z[i = 1:n] <= 1, Int)
    @variable(model, 0 <= k[i = 1:m], Int)
    
    for i in 1:m
        @constraint(model, sum(z[j] for j in 1:n if H[i,j] == 1) == 2 * k[i] + (sydrome[i].x ? 1 : 0))
    end

    @objective(model, Min, sum(z[j] for j in 1:n))
    optimize!(model)
    @assert is_solved_and_feasible(model) "The problem is infeasible!"
    return DecodingResult(true, Mod2.(value.(z) .> 0.5))
end

struct BPResult
    success_tag::Bool
    error_qubits::Vector{Mod2}
    iter::Int
    error_perm::Vector{Int}
end

function decode(decoder::BPDecoder, tanner::SimpleTannerGraph, p::Float64, sydrome::Vector{Mod2})
    res = belief_propagation(sydrome, tanner, p;max_iter=decoder.bp_max_iter)
    return DecodingResult(res.success_tag, res.error_qubits)
end

function decode(decoder::BPOSD, tanner::SimpleTannerGraph, p::Float64, sydrome::Vector{Mod2})
    eqs = bp_osd(sydrome, tanner, p;max_iter=decoder.bp_max_iter)
    return DecodingResult(true, eqs)
end

function belief_propagation(sydrome::Vector{Mod2}, tanner::SimpleTannerGraph, p::Float64;max_iter=100)
    return belief_propagation(sydrome, tanner, fill(p, tanner.nq);max_iter=max_iter)
end

function belief_propagation(sydrome::Vector{Mod2}, tanner::SimpleTannerGraph, p::Vector{Float64};max_iter=100)
    mu = [log((1-p[i])/p[i]) for i in 1:tanner.nq]
    mq2s =[[mu[i] for _ in 1:length(tanner.q2s[i])] for i in 1:tanner.nq]
    ms2q = [[0.0 for _ in 1:length(tanner.s2q[s])] for s in 1:tanner.ns]
    # mq2s =[[rand() for _ in v] for v in tanner.q2s]
    errored_qubits = Mod2[]
    success_tag = false
    iter = 0
    for iter in 1:max_iter
        ms2q = [[messages2q(message_list(mq2s,s,tanner.q2s,tanner.s2q;exampt_qubit = q),sydrome[s]) for q in tanner.s2q[s]] for s in 1:tanner.ns]
        mq2s = [[messageq2s(message_list(ms2q,qubit,tanner.s2q,tanner.q2s;exampt_qubit = s),mu[qubit]) for s in tanner.q2s[qubit]] for qubit in 1:tanner.nq]
        errored_qubits = error_qubits(ms2q,tanner,mu)
        if sydrome_extraction(errored_qubits, tanner) == sydrome
            success_tag = true
            break
        end
    end
    return BPResult(success_tag, errored_qubits, iter, sortperm(messageq(ms2q,tanner,mu)))
end

function message_list(mq2s,s,tq2s,ts2q;exampt_qubit = 0)
    return [mq2s[q][findfirst(==(s),tq2s[q])] for q in ts2q[s] if q != exampt_qubit]
end

function messageq(ms2q,tanner,mu)
    [messageq2s(message_list(ms2q,qubit,tanner.s2q,tanner.q2s),mu[qubit])  for qubit in 1:tanner.nq]
end

function error_qubits(ms2q,tanner,mu)
    Mod2.(messageq(ms2q,tanner,mu) .< 0 )
end

function messages2q(mlist,sydrome)
    pro = 1.0
    for m in mlist
        pro *= tanh(m)
    end
    return (-1)^sydrome.x*atanh(pro)
end

function messageq2s(mlist,muq)
    return max(min(sum(mlist) + muq,10),-10)
end


function osd(tanner::SimpleTannerGraph,order::Vector{Int},syndrome::Vector{Mod2})
    H = tanner.H[:,order[1:1]]
    hinv = 0
    qubit_list = [order[1]]
    for i in 1:length(order)
        if check_linear_indepent(Matrix([H  tanner.H[:,order[i:i]]]'))
            H = [H  tanner.H[:,order[i:i]]]
            push!(qubit_list,order[i])
        end
        if size(H,2) == tanner.ns
            break
        end
    end

    hinv = mod2matrix_inverse(H)

    error = hinv * syndrome
    return [(i ∈ qubit_list) ? (error[findfirst(==(i),qubit_list)]) : Mod2(0)  for i in 1:tanner.nq]
end

function mod2matrix_inverse(H::Matrix{Bool})
    bm = SimpleBimatrix(copy(H),Matrix{Mod2}(I, size(H,1), size(H,1)),collect(1:size(H,2)))
    gaussian_elimination!(bm, 1:size(bm.matrix,1), 0, 0;allow_col_operation = false)
    return bm.Q
end

function mod2matrix_inverse(H::Matrix{Mod2})
    return mod2matrix_inverse([a.x for a in H])
end

function _check_linear_indepent(H::Matrix{Bool})
    bm = SimpleBimatrix(H,Matrix{Mod2}(I, size(H,1), size(H,1)),collect(1:size(H,2)))
    gaussian_elimination!(bm, 1:size(bm.matrix,1), 0, 0)
    return bm,!(bm.matrix[end,:] == fill(Mod2(0),size(H,2)))
end

function check_linear_indepent(H::Matrix{Bool})
    _ , res= _check_linear_indepent(H)
    return res
end

function check_linear_indepent(H::Matrix{Mod2})
    return check_linear_indepent([a.x for a in H])
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

function bp_osd(sydrome::Vector{Mod2}, tanner::SimpleTannerGraph, p::Float64;max_iter=100)
    bp_res = belief_propagation(sydrome, tanner, p;max_iter=max_iter)
    if bp_res.success_tag
        return bp_res.error_qubits
    else
        return osd(tanner, bp_res.error_perm, sydrome)
    end
end

function tensor_osd(sydrome::Vector{Mod2}, tanner::SimpleTannerGraph, p::Float64)
    error_p = tensor_infer(tanner, p, sydrome)
    return osd(tanner, sortperm(-error_p), sydrome)
end

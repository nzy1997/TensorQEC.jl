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

struct CSSDecodingResult
    success_tag::Bool
    xerror_qubits::Vector{Mod2}
    zerror_qubits::Vector{Mod2}
end

Base.show(io::IO, ::MIME"text/plain", cdr::CSSDecodingResult) = show(io, cdr)
function Base.show(io::IO, cdr::CSSDecodingResult)
    println(io, cdr.success_tag ? "Success" : "Failure")
    if cdr.success_tag
        println(io, "X error:", findall(v->v.x, cdr.xerror_qubits))
        println(io, "Z error:", findall(v->v.x,cdr.zerror_qubits))
    end
end

"""
    IPDecoder <: AbstractDecoder

An integer programming decoder.
"""
struct IPDecoder <: AbstractDecoder end

"""
    decode(decoder::AbstractDecoder, tanner::AbstractTannerGraph, syndrome::AbstractSyndrome)
    decode(decoder::AbstractDecoder, tanner::AbstractTannerGraph, syndrome::AbstractSyndrome,p_vec::Vector{AbstractErrorModel})
Decode the syndrome with a given decoder.  `p_vec` is the error model vector.
"""
function decode(decoder::IPDecoder, tanner::SimpleTannerGraph, syndrome::Vector{Mod2})
    return decode(decoder, tanner, syndrome, fill(0.05, tanner.nq))
end
function decode(decoder::IPDecoder, tanner::SimpleTannerGraph, syndrome::Vector{Mod2}, p::Float64)
    return decode(decoder, tanner, syndrome, fill(p, tanner.nq))
end
function decode(decoder::IPDecoder, tanner::SimpleTannerGraph, syndrome::Vector{Mod2},p_vec::Vector{Float64};verbose = false)
    H = [a.x for a in tanner.H]
    return DecodingResult(true,_mixed_integer_programming(H, [s.x for s in syndrome], p_vec;verbose))
end

function _mixed_integer_programming(H::Matrix{Bool}, syndrome::Vector{Bool}, p_vec::Vector{Float64};verbose = false)
    m,n = size(H)
    model = Model(HiGHS.Optimizer)
    !verbose && set_silent(model)

    @variable(model, 0 <= z[i = 1:n] <= 1, Int)
    @variable(model, 0 <= k[i = 1:m], Int)
    
    for i in 1:m
        @constraint(model, sum(z[j] for j in 1:n if H[i,j] == 1) == 2 * k[i] + (syndrome[i] ? 1 : 0))
    end

    @objective(model, Max, sum(log(p_vec[j])*z[j] for j in 1:n)+sum(log(1-p_vec[j])*(1-z[j]) for j in 1:n))
    optimize!(model)
    @assert is_solved_and_feasible(model) "The problem is infeasible!"
    return Mod2.(value.(z) .> 0.5)
end

function decode(decoder::IPDecoder, tanner::CSSTannerGraph, syndrome::CSSSyndrome)
    return decode(decoder, tanner, syndrome, fill(DepolarizingError(0.05, 0.05, 0.05), nq(tanner)))
end

function decode(decoder::IPDecoder, tanner::CSSTannerGraph, syndrome::CSSSyndrome,p_vec::Vector{DepolarizingError};verbose = false)
    Hx = [a.x for a in tanner.stgx.H]
    Hz = [a.x for a in tanner.stgz.H]
    mx,n = size(Hx)
    mz = size(Hz, 1)

    model = Model(HiGHS.Optimizer)
    !verbose && set_silent(model)

    @variable(model, 0 <= x[i = 1:n] <= 1, Int)
    @variable(model, 0 <= y[i = 1:n] <= 1, Int)
    @variable(model, 0 <= z[i = 1:n] <= 1, Int)
    @variable(model, 0 <= k[i = 1:mx], Int)
    @variable(model, 0 <= l[i = 1:mz], Int)
    
    for i in 1:mx
        @constraint(model, sum(z[j]+y[j] for j in 1:n if Hx[i,j] == 1) == 2 * k[i] + (syndrome.sx[i].x ? 1 : 0))
    end
    for i in 1:mz
        @constraint(model, sum(x[j]+y[j] for j in 1:n if Hz[i,j] == 1) == 2 * l[i] + (syndrome.sz[i].x ? 1 : 0))
    end

    for i in 1:n
        @constraint(model, x[i] + y[i] +z[i] <= 1)
    end
    @objective(model, Max, sum(log(p_vec[j].px) * x[j] + log(p_vec[j].py) * y[j] + log(p_vec[j].pz) * z[j] +log(1- p_vec[j].px - p_vec[j].py - p_vec[j].pz) * (1 - x[j] - y[j] - z[j]) for j in 1:n))
    optimize!(model)
    @assert is_solved_and_feasible(model) "The problem is infeasible!"
    return CSSDecodingResult(true, Mod2.(value.(x) .> 0.5) .+ Mod2.(value.(y) .> 0.5), Mod2.(value.(z) .> 0.5) .+ Mod2.(value.(y) .> 0.5))
end

struct BPResult
    success_tag::Bool
    error_qubits::Vector{Mod2}
    iter::Int
    error_perm::Vector{Int}
end

function decode(decoder::BPDecoder, tanner::SimpleTannerGraph, syndrome::Vector{Mod2}, p::Float64)
    res = belief_propagation(syndrome, tanner, p;max_iter=decoder.bp_max_iter)
    return DecodingResult(res.success_tag, res.error_qubits)
end

function decode(decoder::BPOSD, tanner::SimpleTannerGraph, syndrome::Vector{Mod2}, p::Float64)
    eqs = bp_osd(syndrome, tanner, p;max_iter=decoder.bp_max_iter)
    return DecodingResult(true, eqs)
end

function belief_propagation(syndrome::Vector{Mod2}, tanner::SimpleTannerGraph, p::Float64;max_iter=100)
    return belief_propagation(syndrome, tanner, fill(p, tanner.nq);max_iter=max_iter)
end

function belief_propagation(syndrome::Vector{Mod2}, tanner::SimpleTannerGraph, p::Vector{Float64};max_iter=100)
    mu = [log((1-p[i])/p[i]) for i in 1:tanner.nq]
    mq2s =[[mu[i] for _ in 1:length(tanner.q2s[i])] for i in 1:tanner.nq]
    ms2q = [[0.0 for _ in 1:length(tanner.s2q[s])] for s in 1:tanner.ns]
    # mq2s =[[rand() for _ in v] for v in tanner.q2s]
    errored_qubits = Mod2[]
    success_tag = false
    iter = 0
    for iter in 1:max_iter
        ms2q = [[messages2q(message_list(mq2s,s,tanner.q2s,tanner.s2q;exampt_qubit = q),syndrome[s]) for q in tanner.s2q[s]] for s in 1:tanner.ns]
        mq2s = [[messageq2s(message_list(ms2q,qubit,tanner.s2q,tanner.q2s;exampt_qubit = s),mu[qubit]) for s in tanner.q2s[qubit]] for qubit in 1:tanner.nq]
        errored_qubits = error_qubits(ms2q,tanner,mu)
        if syndrome_extraction(errored_qubits, tanner) == syndrome
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

function messages2q(mlist,syndrome)
    pro = 1.0
    for m in mlist
        pro *= tanh(m)
    end
    return (-1)^syndrome.x*atanh(pro)
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

function bp_osd(syndrome::Vector{Mod2}, tanner::SimpleTannerGraph, p::Float64;max_iter=100)
    bp_res = belief_propagation(syndrome, tanner, p;max_iter=max_iter)
    if bp_res.success_tag
        return bp_res.error_qubits
    else
        return osd(tanner, bp_res.error_perm, syndrome)
    end
end

function tensor_osd(syndrome::Vector{Mod2}, tanner::SimpleTannerGraph, p::Float64)
    error_p = tensor_infer(tanner, p, syndrome)
    return osd(tanner, sortperm(-error_p), syndrome)
end

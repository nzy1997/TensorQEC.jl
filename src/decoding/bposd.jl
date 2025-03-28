struct BPResult
    success_tag::Bool
    error_qubits::Vector{Mod2}
    error_perm::Vector{Int}
end

Base.@kwdef struct BPDecoder <: AbstractDecoder
    bp_max_iter::Int = 100
end

Base.@kwdef struct BPOSD <: AbstractDecoder
    bp_max_iter::Int =100
end

struct CompiledBP<: CompiledDecoder 
    tanner::SimpleTannerGraph
    bp_max_iter::Int
    mq2s::Dict{Tuple{Int64, Int64}, Float64}
    ms2q::Dict{Tuple{Int64, Int64}, Float64}
    mu::Vector{Float64}
end

function compile(decoder::BPDecoder, problem::SimpleDecodingProblem)
    mu = [log((1-problem.pvec[i])/problem.pvec[i]) for i in 1:problem.tanner.nq]
    mq2s = Dict([(i,j) => mu[i] for i in 1:problem.tanner.nq for j in problem.tanner.q2s[i] ])
    ms2q = Dict([(s,i) => 0.0 for s in 1:problem.tanner.ns for i in problem.tanner.s2q[s] ])
    return CompiledBP(problem.tanner, decoder.bp_max_iter,mq2s,ms2q,mu)
end

function messages2q(mq2s,s2q,s)
    pro = 1.0
    for m in s2q
        pro *= tanh(mq2s[(m,s)])
    end
    return pro
end

function messageq2s(mlist,muq)
    return max(min(sum(mlist) + muq,10),-10)
end

function decode(cb::CompiledBP,syndrome::SimpleSyndrome)
    return belief_propagation(cb,syndrome.s)
end
function belief_propagation(cb::CompiledBP,syndrome::Vector{Mod2})
    mq2s = copy(cb.mq2s)
    q_vec = fill(0.0,cb.tanner.nq)
    for _ in 1:cb.bp_max_iter
        for s in 1:cb.tanner.ns
            pro = messages2q(mq2s,cb.tanner.s2q[s],s)
            for q in cb.tanner.s2q[s]
                cb.ms2q[(s,q)] = (-1)^syndrome[s].x*atanh(pro/tanh(mq2s[(q,s)]))
            end
        end
        for q in 1:cb.tanner.nq
            q_vec[q] = sum(x -> cb.ms2q[(x,q)], cb.tanner.q2s[q]) + cb.mu[q]
            for s in cb.tanner.q2s[q]
                mq2s[(q,s)] = max(min(q_vec[q] - cb.ms2q[(s,q)],10),-10)
            end
        end
        errored_qubits =  Mod2.(q_vec .< 0 )
        @show q_vec
        if syndrome_extraction(errored_qubits, cb.tanner).s == syndrome
            return BPResult(true, errored_qubits, sortperm(messageq(ms2q,tanner,mu)))
            break
        end
    end

    return DecodingResult(res.success_tag, res.error_qubits)
end

struct CompiledBPOSD <: CompiledDecoder 
    problem::SimpleDecodingProblem
    bp_max_iter::Int
end

function compile(decoder::BPOSD, problem::SimpleDecodingProblem)
    return CompiledBPOSD(problem, decoder.bp_max_iter)
end

function decode(cb::CompiledBPOSD,syndrome::SimpleSyndrome)
    res = bp_osd(syndrome.s, cb.problem.tanner, cb.problem.pvec;max_iter=cb.bp_max_iter)
    return DecodingResult(true, res)
end

# function belief_propagation(syndrome::Vector{Mod2}, tanner::SimpleTannerGraph, p::Float64;max_iter=100)
#     return belief_propagation(syndrome, tanner, fill(p, tanner.nq);max_iter=max_iter)
# end

# function belief_propagation(syndrome::Vector{Mod2}, tanner::SimpleTannerGraph, p::Vector{Float64};max_iter=100)
#     mu = [log((1-p[i])/p[i]) for i in 1:tanner.nq]
#     mq2s =[[mu[i] for _ in 1:length(tanner.q2s[i])] for i in 1:tanner.nq]
#     ms2q = [[0.0 for _ in 1:length(tanner.s2q[s])] for s in 1:tanner.ns]
#     errored_qubits = Mod2[]
#     for iter in 1:max_iter
#         ms2q = [[messages2q(message_list(mq2s,s,tanner.q2s,tanner.s2q;exampt_qubit = q),syndrome[s]) for q in tanner.s2q[s]] for s in 1:tanner.ns]
#         mq2s = [[messageq2s(message_list(ms2q,qubit,tanner.s2q,tanner.q2s;exampt_qubit = s),mu[qubit]) for s in tanner.q2s[qubit]] for qubit in 1:tanner.nq]
#         errored_qubits = error_qubits(ms2q,tanner,mu)
#         if syndrome_extraction(errored_qubits, tanner).s == syndrome
#             return BPResult(true, errored_qubits, sortperm(messageq(ms2q,tanner,mu)))
#             break
#         end
#     end
#     return BPResult(false, fill(Mod2(0),tanner.nq), sortperm(messageq(ms2q,tanner,mu)))
# end

# function message_list(mq2s,s,tq2s,ts2q;exampt_qubit = 0)
#     return [mq2s[q][findfirst(==(s),tq2s[q])] for q in ts2q[s] if q != exampt_qubit]
# end

# function messageq(ms2q,tanner,mu)
#     [messageq2s(message_list(ms2q,qubit,tanner.s2q,tanner.q2s),mu[qubit])  for qubit in 1:tanner.nq]
# end

# function error_qubits(ms2q,tanner,mu)
#     Mod2.(messageq(ms2q,tanner,mu) .< 0 )
# end

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


function bp_osd(syndrome::Vector{Mod2}, tanner::SimpleTannerGraph, p::Vector{Float64};max_iter=100)
    bp_res = belief_propagation(syndrome, tanner, p;max_iter=max_iter)
    if bp_res.success_tag
        return bp_res.error_qubits
    else
        return osd(tanner, bp_res.error_perm, syndrome)
    end
end
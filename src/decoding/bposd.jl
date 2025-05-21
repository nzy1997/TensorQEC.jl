struct BPResult
    success_tag::Bool
    error_qubits::Vector{Mod2}
    error_perm::Vector{Int}
end

abstract type AbstractBPDecoder <: AbstractDecoder end
Base.@kwdef struct BPDecoder <: AbstractBPDecoder
    bp_max_iter::Int = 100
    osd::Bool = true
end

struct CompiledBP <: CompiledDecoder
    tanner::SimpleTannerGraph
    bp_max_iter::Int
    mq2s::Dict{Tuple{Int64, Int64}, Float64}
    ms2q::Dict{Tuple{Int64, Int64}, Float64}
    mu::Vector{Float64}
    osd::Bool
end

struct CompiledCSSBP <: CompiledDecoder
   cbx::CompiledBP
   cbz::CompiledBP
end

function compile(decoder::AbstractBPDecoder, problem::CSSDecodingProblem)
    cbx = compile(decoder, SimpleDecodingProblem(problem.tanner.stgx, fill(0.05,problem.tanner.stgx.nq)))
    cbz = compile(decoder, SimpleDecodingProblem(problem.tanner.stgz, fill(0.05,problem.tanner.stgz.nq)))
    return CompiledCSSBP(cbx,cbz)
end
function decode(cb::CompiledCSSBP,syndrome::CSSSyndrome)
    bp_resx = decode(cb.cbz,SimpleSyndrome(syndrome.sz))
    bp_resz = decode(cb.cbx,SimpleSyndrome(syndrome.sx))
    return CSSDecodingResult(bp_resx.success_tag && bp_resz.success_tag, CSSErrorPattern(bp_resx.error_qubits,bp_resz.error_qubits))
end

function compile(decoder::AbstractBPDecoder, problem::SimpleDecodingProblem)
    mu = [log((1-problem.pvec[i])/problem.pvec[i]) for i in 1:problem.tanner.nq]
    mq2s = Dict([(i,j) => mu[i] for i in 1:problem.tanner.nq for j in problem.tanner.q2s[i] ])
    ms2q = Dict([(s,i) => 0.0 for s in 1:problem.tanner.ns for i in problem.tanner.s2q[s] ])
    return CompiledBP(problem.tanner, decoder.bp_max_iter,mq2s,ms2q,mu, decoder.osd)
end

function messages2q(mq2s,s2q,s)
    pro = 1.0
    for m in s2q
        pro *= tanh(mq2s[(m,s)])
    end
    return pro
end

function decode(cb::CompiledBP,syndrome::SimpleSyndrome)
    bp_res = belief_propagation(cb,syndrome.s)
    if bp_res.success_tag || !(cb.osd)
        return DecodingResult(bp_res.success_tag,bp_res.error_qubits)
    else
        return DecodingResult(true,osd(cb.tanner, bp_res.error_perm, syndrome.s))
    end
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
        if syndrome_extraction(errored_qubits, cb.tanner).s == syndrome
            return BPResult(true, errored_qubits, sortperm(q_vec))
        end
    end
    return BPResult(false, fill(Mod2(0),cb.tanner.nq), sortperm(q_vec))
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
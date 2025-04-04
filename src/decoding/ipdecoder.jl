"""
    IPDecoder <: AbstractDecoder

An integer programming decoder.
"""
Base.@kwdef struct IPDecoder <: AbstractDecoder 
    optimizer = SCIP.Optimizer
    verbose::Bool = false
end

struct CompiledIP <: CompiledDecoder
    decoder::IPDecoder
    reduction::AbstractReductionResult
end

struct FlatDecodingProblem
    tanner::SimpleTannerGraph
    code::Vector{Vector{Int}}
    pvec::Vector{Vector{Float64}}
end

struct GeneralDecodingProblemToFlatDecodingProblem <: AbstractReductionResult
    fdp::FlatDecodingProblem
    dict::Dict{Int,Vector{Int}}
    qubit_num::Int
end
get_fdp(g::TensorQEC.GeneralDecodingProblemToFlatDecodingProblem) = g.fdp
struct SimpleDecodingProblemToFlatDecodingProblem <: AbstractReductionResult
    fdp::FlatDecodingProblem
end
get_fdp(cfdp::SimpleDecodingProblemToFlatDecodingProblem) = cfdp.fdp
extract_decoding(re::SimpleDecodingProblemToFlatDecodingProblem, error_qubits::Vector{Mod2}) = DecodingResult(true, error_qubits)

function compile(decoder::IPDecoder, gdp::GeneralDecodingProblem)
    gdp2fdp = flattengdp(gdp)
    return CompiledIP(decoder, gdp2fdp)
end

function compile(decoder::IPDecoder, sdp::SimpleDecodingProblem)
    fdp = FlatDecodingProblem(sdp.tanner, [[i] for i in 1:sdp.tanner.nq], [[p] for p in sdp.pvec])
    return CompiledIP(decoder, SimpleDecodingProblemToFlatDecodingProblem(fdp))
end

function decode(ci::CompiledIP, syndrome::CSSSyndrome)
    return decode(ci,SimpleSyndrome([syndrome.sx...,syndrome.sz...]))
end
function decode(ci::CompiledIP, syndrome::SimpleSyndrome)
    return extract_decoding(ci.reduction,_mixed_integer_programming(ci.decoder,get_fdp(ci.reduction), syndrome.s))
end

struct CSSDecodingProblemToFlatDecodingProblem <: AbstractReductionResult
    fdp::FlatDecodingProblem
    c2g::CSSToGeneralDecodingProblem
    gdp2fdp::GeneralDecodingProblemToFlatDecodingProblem
end
get_fdp(cfdp::CSSDecodingProblemToFlatDecodingProblem) = cfdp.fdp
function compile(decoder::IPDecoder, sdp::CSSDecodingProblem)
    c2g = reduce2general(sdp.tanner,[[p.px,p.py,p.pz] for p in sdp.pvec])
    gdp2fdp = flattengdp(c2g.gdp)
    return CompiledIP(decoder, CSSDecodingProblemToFlatDecodingProblem(gdp2fdp.fdp,c2g,gdp2fdp))
end

function extract_decoding(cfdp::CSSDecodingProblemToFlatDecodingProblem, error_qubits::Vector{Mod2})
    return extract_decoding(cfdp.c2g,extract_decoding(cfdp.gdp2fdp,error_qubits).error_qubits)
end

function _mixed_integer_programming(decoder::IPDecoder, fdp::FlatDecodingProblem, syndrome::Vector{Mod2})
    H = [a.x for a in fdp.tanner.H]
    m,n = size(H)
    model = Model(decoder.optimizer)
    !decoder.verbose && set_silent(model)

    @variable(model, 0 <= z[i = 1:n] <= 1, Int)
    @variable(model, 0 <= k[i = 1:m], Int)
    
    for i in 1:m
        @constraint(model, sum(z[j] for j in 1:n if H[i,j] == 1) == 2 * k[i] + (syndrome[i].x ? 1 : 0))
    end

    obj = 0.0
    for (i,code) in enumerate(fdp.code)
        tensor = fdp.pvec[i]
        zsum = sum([z[i] for i in code])
        @constraint(model, zsum <= 1)
        for j in 1:length(tensor)
            if tensor[j] == 0.0
                @constraint(model, z[code[j]] == 0)
            else
                obj += log(tensor[j]) * z[code[j]]
            end
        end
        obj += log(1-sum(tensor)) * (1-zsum)
    end
    @objective(model, Max, obj)
    optimize!(model)
    @assert is_solved_and_feasible(model) "The problem is infeasible!"
    return Mod2.(value.(z) .> 0.5)
end

function flattengdp(gdp::GeneralDecodingProblem)
    code = deepcopy(gdp.ptn.code.ixs)
    pvec = Vector{Vector{Float64}}(undef, length(code))

    num_error_pos = gdp.tanner.nq
    s2q = deepcopy(gdp.tanner.s2q)

    dict = Dict{Int,Vector{Int}}()
    for (j,tensor) in enumerate(gdp.ptn.tensors)
        nonzero_pos = findall(!iszero, tensor)
        pvec_temp = Vector{Float64}()
        for i in 1:length(code[j])
            push!(pvec_temp, tensor[[ i == t ? 2 : 1 for t in 1:length(code[j])]...])
        end
        for (i,pos) in enumerate(nonzero_pos)
            pos2 = findall(==(2), pos.I)
            if length(pos2) <= 1
                continue
            else
                num_error_pos += 1
                s2qtemp = _setmod2(mapreduce(x->gdp.tanner.q2s[code[j][x]],vcat,pos2))
                dict[num_error_pos] = code[j][pos2]
                for s in s2qtemp
                    push!(s2q[s],num_error_pos)
                end
                push!(code[j],num_error_pos)
                push!(pvec_temp,tensor[pos])
            end
        end
        pvec[j] = pvec_temp
    end
    return GeneralDecodingProblemToFlatDecodingProblem(FlatDecodingProblem(SimpleTannerGraph(num_error_pos,s2q),code,pvec),dict,gdp.tanner.nq)
end


function extract_decoding(gdp2fdp::GeneralDecodingProblemToFlatDecodingProblem, error_qubits::Vector{Mod2})
    num_qubits = gdp2fdp.qubit_num
    ans_error_qubits = error_qubits[1:num_qubits]

    for i in num_qubits+1:length(error_qubits)
        if error_qubits[i].x == 1
            ans_error_qubits[gdp2fdp.dict[i]] = fill(Mod2(1),length(gdp2fdp.dict[i]))
        end
    end

    return DecodingResult(true,ans_error_qubits)
end

function _setmod2(vec::Vector{Int})
    vans = Int[]
    for v in vec
        pos  = findfirst(==(v),vans)
        if isnothing(pos)
            push!(vans,v)
        else
            deleteat!(vans,pos)
        end
    end
    return vans
end

function _mixed_integer_programming_for_one_solution(H, syndrome::Vector{Mod2})
    m,n = size(H)
    model = Model(SCIP.Optimizer)
    set_silent(model)

    @variable(model, 0 <= z[i = 1:n] <= 1, Int)
    @variable(model, 0 <= k[i = 1:m], Int)
    
    for i in 1:m
        @constraint(model, sum(z[j] for j in 1:n if H[i,j] == 1) == 2 * k[i] + (syndrome[i].x ? 1 : 0))
    end
    @objective(model, Max, 1.0)
    optimize!(model)
    @assert is_solved_and_feasible(model) "The problem is infeasible!"
    return Mod2.(value.(z) .> 0.5)
end
"""
    IPDecoder <: AbstractGeneralDecoder

An integer programming based decoder.

### Note
TBW

### Keyword Arguments
- `optimizer = SCIP.Optimizer`: The optimizer to use for solving the integer programming problem
- `verbose::Bool = false`: Whether to print the verbose output
"""
Base.@kwdef struct IPDecoder <: AbstractGeneralDecoder 
    optimizer = SCIP.Optimizer   # The optimizer to use for solving the integer programming problem
    verbose::Bool = false        # Whether to print the verbose output
end

"""
    FlatDecodingProblem

TBW

### Fields
- `tanner::SimpleTannerGraph`: The Tanner graph
- `code::Vector{Vector{Int}}`: TBW
- `pvec::Vector{Vector{Float64}}`: The probability vector
"""
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

struct CompiledIP <: CompiledDecoder
    decoder::IPDecoder
    reduction::GeneralDecodingProblemToFlatDecodingProblem
end

get_fdp(g::TensorQEC.GeneralDecodingProblemToFlatDecodingProblem) = g.fdp
function compile(decoder::IPDecoder, gdp::GeneralDecodingProblem)
    gdp2fdp = flattengdp(gdp)
    return CompiledIP(decoder, gdp2fdp)
end

function decode(ci::CompiledIP, syndrome::SimpleSyndrome)
    return extract_decoding(ci.reduction,_mixed_integer_programming(ci.decoder,get_fdp(ci.reduction), syndrome.s))
end

function _mixed_integer_programming(decoder::IPDecoder, fdp::FlatDecodingProblem, syndrome::Vector{Mod2})
    H = fdp.tanner.H
    m,n = size(H)
    model = Model(decoder.optimizer)
    !decoder.verbose && set_silent(model)

    @variable(model, 0 <= z[i = 1:n] <= 1, Int)
    @variable(model, 0 <= k[i = 1:m], Int)
    
    for i in 1:m
        @constraint(model, sum(z[j] for j in 1:n if H[i,j].x) == 2 * k[i] + (syndrome[i].x ? 1 : 0))
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
        if tensor isa Vector
            pvec[j] = [tensor[2]]
            continue
        end
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

function _mixed_integer_programming_for_one_solution(H::Matrix{Mod2}, syndrome::Vector{Mod2})
    m,n = size(H)
    model = Model(SCIP.Optimizer)
    set_silent(model)

    @variable(model, 0 <= z[i = 1:n] <= 1, Int)
    @variable(model, 0 <= k[i = 1:m], Int)
    
    for i in 1:m
        @constraint(model, sum(z[j] for j in 1:n if H[i,j].x) == 2 * k[i] + (syndrome[i].x ? 1 : 0))
    end

    optimize!(model)
    @assert is_solved_and_feasible(model) "The problem is infeasible!"
    return Mod2.(value.(z) .> 0.5)
end

function _mixed_integer_programming_for_one_solution(tanner::CSSTannerGraph, syndrome::CSSSyndrome)
    return _mixed_integer_programming_for_one_solution(tanner.stgz.H, syndrome.sz), _mixed_integer_programming_for_one_solution(tanner.stgx.H, syndrome.sx)
end
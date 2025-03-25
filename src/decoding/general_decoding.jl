abstract type AbstractReductionResult end

struct CSSToGeneralDecodingProblem <: AbstractReductionResult
    gdp::GeneralDecodingProblem
    qubit_num::Int
end

struct GeneralDecodingProblemToFlatDecodingProblem <: AbstractReductionResult
    fdp::FlatDecodingProblem
    dict::Dict{Int,Vector{Int}}
    qubit_num::Int
end

single_qubit_tensor(p::Float64) = single_qubit_tensor(p,p,p)
function single_qubit_tensor(px::Float64, py::Float64, pz::Float64)
    return [[1-px-py-pz py;px 0.0];;;[pz 0.0;0.0 0.0]]
end

function reduce2general(tanner::CSSTannerGraph, pvec::Vector{Vector{Float64}})
    num_qubits = nq(tanner)
    tn = TensorNetwork(DynamicEinCode([[i,i+num_qubits,i+2*num_qubits] for i in 1:num_qubits],Int[]),[single_qubit_tensor(pvec[j]...) for j in 1:num_qubits])
    return reduce2general(tanner,tn)
end

function reduce2general(tanner::CSSTannerGraph,p::Float64)
    num_qubits = nq(tanner)
    return reduce2general(tanner,fill(fill(p,3),num_qubits))
end

function reduce2general(tanner::CSSTannerGraph,tn::TensorNetwork)
    num_qubits = nq(tanner)
    return CSSToGeneralDecodingProblem(GeneralDecodingProblem(SimpleTannerGraph(3*num_qubits,[[vcat(x.+ num_qubits,x .+ 2* num_qubits) for x in tanner.stgx.s2q]..., [vcat(x,x .+  num_qubits) for x in tanner.stgz.s2q]...]),tn),num_qubits)
end

function extract_decoding(cgdp::CSSToGeneralDecodingProblem, error_qubits::Vector{Mod2})
    num_qubits = cgdp.qubit_num
    return CSSDecodingResult(true, error_qubits[1:num_qubits] .+  error_qubits[(num_qubits+1):2*num_qubits] , error_qubits[(2*num_qubits+1):3*num_qubits] .+  error_qubits[(num_qubits+1):2*num_qubits])
end

function general_syndrome(syn::CSSSyndrome)
    return vcat(syn.sx,syn.sz)
end
_mixed_integer_programming(fdp::FlatDecodingProblem, syndrome::Vector{Mod2};verbose = false) = _mixed_integer_programming(IPDecoder(), fdp, syndrome;verbose = verbose)
function _mixed_integer_programming(decoder::IPDecoder, fdp::FlatDecodingProblem, syndrome::Vector{Mod2};verbose = false)
    H = [a.x for a in fdp.tanner.H]
    m,n = size(H)
    model = Model(decoder.optimizer)
    !verbose && set_silent(model)

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
    code = gdp.ptn.code.ixs
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

    return ans_error_qubits
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

function decode(decoder::IPDecoder, gdp::GeneralDecodingProblem, syndrome::Vector{Mod2})
    gdp2fdp = flattengdp(gdp)
    return extract_decoding(gdp2fdp,_mixed_integer_programming(decoder,gdp2fdp.fdp, syndrome))
end
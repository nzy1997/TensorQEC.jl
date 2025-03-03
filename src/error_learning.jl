function qc_probability(qc::ChainBlock, final_state::Vector{Complex{Float64}})
    optnet = probability_tn(qc, final_state)
    return abs(Yao.contract(optnet)[])
end

function probability_tn(qc::ChainBlock, final_state::Vector{Complex{Float64}})
    qc = copy(qc)
    push!(qc,matblock(final_state*final_state'))
    number_qubits = nqubits(qc)
    tn,_,_ = simulation_tensornetwork(qc,QCInfo(Int[],collect(1:number_qubits),number_qubits))
    optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())
    return optnet
end

function Yao.depolarizing_channel(n::Int, p_vec::AbstractVector)
    return UnitaryChannel(vec(pauli_basis(n)),p_vec)
end

mutable struct TrainingChannel{D} <: AbstractRecoder{D}
    symbol::Int
    uc::UnitaryChannel
    input_indices
    output_indices
    tensor_pos
end

Yao.nqudits(x::TrainingChannel) = x.symbol
TrainingChannel(qudit_num, uc; nlevel=2) = TrainingChannel{nlevel}(qudit_num,uc,nothing,nothing,nothing)

function YaoPlots.draw!(c::YaoPlots.CircuitGrid, p::TrainingChannel, address, controls)
    @assert length(controls) == 0
    YaoPlots._draw!(c, [(getindex.(Ref(address), (1,)), c.gatestyles.g, "TrainingChannel")])
end

function YaoToEinsum.add_gate!(eb::YaoToEinsum.EinBuilder{T}, b::PutBlock{D,C,TrainingChannel{D}}) where {T,D,C}
    locs = collect(b.locs)

    k = length(locs)
    
    nlabels = [YaoToEinsum.newlabel!(eb) for _=1:k]

    b.content.input_indices = eb.slots[locs]
    b.content.output_indices = nlabels

    eb.slots[locs] .= nlabels
    return eb
end

TrainingChannel(; nlevel=2) = TrainingChannel{nlevel}(nothing)
function YaoToEinsum.add_gate!(eb::YaoToEinsum.EinBuilder{T}, b::PutBlock{D,C,ComplexConj{TrainingChannel{D},D}}) where {T,D,C}
    locs = collect(b.locs)
    k = length(locs)
    nlabels = [YaoToEinsum.newlabel!(eb) for _=1:k]

    ops1,ops2,probs = channel2tensor(b.content.content.uc)
    plabel = YaoToEinsum.newlabel!(eb)

    label_vec1 = [b.content.content.output_indices...,b.content.content.input_indices...,plabel]
    label_vec2 = [eb.slots[locs]..., nlabels...,plabel]

    YaoToEinsum.add_tensor!(eb, ops1, label_vec1)
    YaoToEinsum.add_tensor!(eb, ops2, label_vec2)
    YaoToEinsum.add_tensor!(eb, probs, [plabel])

    b.content.content.tensor_pos = length(eb.tensors)
    eb.slots[locs] .= nlabels
    return eb
end
function probability_channel(qc::ChainBlock, final_state::Vector{Complex{Float64}})
    optnet,_ = probability_tn_channel(qc,final_state)
    return real(Yao.contract(optnet)[])
end
function probability_tn_channel(qc::ChainBlock, final_state::Vector{Complex{Float64}})
    qc = copy(qc)
    number_qubits = nqubits(qc)
    push!(qc,put(number_qubits,(1:number_qubits)=>matblock(final_state*final_state')))
    qc= simplify(qc; rules=[to_basictypes, Optimise.eliminate_nested])

    qc2 = chain(number_qubits)
    channel_count = 0
    tc = Vector{TrainingChannel}()
    for gate in qc
        if gate.content isa UnitaryChannel
            channel_count += 1
            push!(tc, TrainingChannel(nqubits(gate.content),gate.content))
            push!(qc2, PutBlock(number_qubits, tc[channel_count],gate.locs))
        else
            push!(qc2, gate)
        end
    end
    qc_info = QCInfo(Int[],collect(1:number_qubits),number_qubits)
    qc2= simplify(qc2; rules=[to_basictypes, Optimise.eliminate_nested])
    qce,srs = ein_circ(qc2,qc_info)
    # return qce
    tn,_,_ = qc2enisum(qce,srs,qc_info)
    # return tn,getfield.(tc,:tensor_pos)
 
    optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())
    return optnet,getfield.(tc,:tensor_pos)
end

function channel2mat(uc::UnitaryChannel)
    return mat(sum([uc.probs[x]*kron(uc.operators[x],uc.operators[x]') for x in 1:length(uc.probs)]))
end

function channel2tensor(uc::UnitaryChannel)
    k = 2*uc.n
    return cat([reshape(mat(x),(fill(2,k)...,1)) for x in uc.operators]...;dims = k+1),cat([reshape(mat(x'),(fill(2,k)...,1)) for x in uc.operators]...;dims = k+1),ComplexF64.(uc.probs)
end

struct TrainningData
    pvec::Vector{Float64}
    states::Vector{Vector{ComplexF64}}
end

function generate_new_tensor(old_tensors::Vector{AbstractArray{ComplexF64}},p_pos::Vector{Int},final_state::Vector{Complex{Float64}},pvec::Vector{Vector{Float64}})
    new_t = copy(old_tensors)
    n = length(old_tensors)
    channel_num = length(p_pos)
    new_t[p_pos] = pvec
    qubit_num = Int(log2(length(final_state)))
    state_tensor = reshape(final_state*final_state',(fill(2,2*qubit_num)...,))
    new_t[end] = conj.(state_tensor)
    new_t[(n-2*qubit_num-3*channel_num) ÷ 2 + 2*qubit_num] = state_tensor
    return new_t
end

function get_grad(code::SlicedEinsum, p::Float64,tensors::Vector{AbstractArray{ComplexF64}},p_pos::Vector{Int})
    p_app,grad = OMEinsum.cost_and_gradient(code,(tensors...,))
    return [2*(real(p_app[])-p).* (grad[x]) for x in p_pos] #,(real(p_app[])-p)^2
end

function get_grad(code::SlicedEinsum,tensors::Vector{AbstractArray{ComplexF64}},p_pos::Vector{Int}, td::TrainningData,pvec::Vector{Vector{Float64}})
    pvec_input = [[1 - sum(x), x...] for x in pvec]
    temp = sum([get_grad(code,td.pvec[x],generate_new_tensor(tensors,p_pos,td.states[x],pvec_input),p_pos) for x in 1:length(td.pvec)])
    return [x[2:end] .- x[1] for x in temp]
end

function loss_function(code::SlicedEinsum,tensors::Vector{AbstractArray{ComplexF64}},p_pos::Vector{Int}, td::TrainningData,pvec::Vector{Vector{Float64}})
    pvec_input = [[1 - sum(x), x...] for x in pvec]
    return sum([(real(code(generate_new_tensor(tensors,p_pos,td.states[x],pvec_input)...)[])-td.pvec[x])^2 for x in 1:length(td.pvec)])
end

function error_learning(model,td::TrainningData,optnet,p_pos::Vector{Int};iter=10)
    model = copy(model)
    train_state = Optimisers.setup(Optimisers.Adam(), model)
    for i in 1:iter
        grad = real.(get_grad(optnet.code,optnet.tensors,p_pos,td, model))
        if norm(grad) < 1e-8
            break
        end
        train_state, model = Optimisers.update(train_state, model,grad)
        @show loss_function(optnet.code,optnet.tensors,p_pos,td,model)
        @show i
        @show norm(grad)
    end
    return model
end
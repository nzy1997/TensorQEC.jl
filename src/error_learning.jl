function qc_probability(qc::ChainBlock, final_state::Vector{Complex{Float64}})
    optnet = probability_tn(qc, final_state)
    return abs(Yao.contract(optnet)[1])
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
end

Yao.nqudits(x::TrainingChannel) = x.symbol
TrainingChannel(qudit_num, uc; nlevel=2) = TrainingChannel{nlevel}(qudit_num,uc,nothing,nothing)

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

    ops,probs = channel2tensor(b.content.content.uc)
    plabel = YaoToEinsum.newlabel!(eb)

    label_vec = [b.content.content.output_indices..., eb.slots[locs]...,b.content.content.input_indices...,nlabels...,plabel]

    YaoToEinsum.add_tensor!(eb, ops, label_vec)
    YaoToEinsum.add_tensor!(eb, probs, [plabel])

    eb.slots[locs] .= nlabels
    return eb
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
    return abs(einsum(tn.code,(tn.tensors...,))[1])
    # optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())
    # return optnet
end

function channel2mat(uc::UnitaryChannel)
    return mat(sum([uc.probs[x]*kron(uc.operators[x],uc.operators[x]') for x in 1:length(uc.probs)]))
end

function channel2tensor(uc::UnitaryChannel)
    k = 4*uc.n
    return cat([reshape(mat(kron(x,x')),(fill(2,k)...,1)) for x in uc.operators]...;dims = k+1),ComplexF64.(uc.probs)
end
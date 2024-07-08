struct ComplexConj{BT<:AbstractBlock,D} <: TagBlock{BT,D}
    content::BT
end
ComplexConj(x::BT) where {D,BT<:AbstractBlock{D}} = ComplexConj{BT,D}(x)
Yao.mat(::Type{T}, blk::ComplexConj) where {T} = conj(mat(T, content(blk)))

Base.conj(x::Union{XGate, ZGate,HGate}) = x
Base.conj(x::AbstractBlock) = ComplexConj(x)
Base.conj(x::ComplexConj) = content(x)
Base.copy(x::ComplexConj) = ComplexConj(copy(content(x)))
YaoBlocks.chsubblocks(blk::ComplexConj, target::AbstractBlock) = ComplexConj(target)

Base.conj(blk::ChainBlock{D}) where {D} =
    ChainBlock(blk.n, AbstractBlock{D}[conj(b) for b in subblocks(blk)])
Base.conj(x::PutBlock) = PutBlock(nqudits(x), conj(content(x)), x.locs)

Base.conj(blk::ControlBlock) =
    ControlBlock(blk.n, blk.ctrl_locs, blk.ctrl_config, conj(blk.content), blk.locs)

function YaoBlocks.map_address(blk::ComplexConj, info::AddressInfo)
    ComplexConj(YaoBlocks.map_address(content(blk), info))
end
YaoBlocks.Optimise.to_basictypes(block::ComplexConj) = ComplexConj(block.content)

function YaoPlots.draw!(c::YaoPlots.CircuitGrid, p::ComplexConj{<:PrimitiveBlock}, address, controls)
    bts = length(controls)>=1 ? YaoPlots.get_cbrush_texts(c, content(p)) : YaoPlots.get_brush_texts(c, content(p))
    YaoPlots._draw!(c, [controls..., (getindex.(Ref(address), occupied_locs(p)), bts[1], "conj of "*bts[2])])
end

abstract type AbstractRecoder{D} <: TrivialGate{D} end

mutable struct IdentityRecorder{D} <: AbstractRecoder{D}
    symbol
end

mutable struct SymbolRecorder{D} <: AbstractRecoder{D}
    symbol
end

IdentityRecorder(; nlevel=2) = IdentityRecorder{nlevel}(nothing)
SymbolRecorder(; nlevel=2) = SymbolRecorder{nlevel}(nothing)
Yao.nqudits(sr::AbstractRecoder) = 1
Yao.print_block(io::IO, sr::AbstractRecoder) = print(io, sr.symbol)

function YaoPlots.draw!(c::YaoPlots.CircuitGrid, p::IdentityRecorder, address, controls)
    @assert length(controls) == 0
    YaoPlots._draw!(c, [(getindex.(Ref(address), (1,)), c.gatestyles.g, "I$(p.symbol)")])
end

function YaoPlots.draw!(c::YaoPlots.CircuitGrid, p::SymbolRecorder, address, controls)
    @assert length(controls) == 0
    YaoPlots._draw!(c, [(getindex.(Ref(address), (1,)), c.gatestyles.g, "$(p.symbol)")])
end

function YaoToEinsum.add_gate!(eb::YaoToEinsum.EinBuilder{T}, b::PutBlock{D,C,SymbolRecorder{D}}) where {T,D,C}
    lj = eb.slots[b.locs[1]]
    b.content.symbol = lj
    return eb
end

function YaoToEinsum.add_gate!(eb::YaoToEinsum.EinBuilder{T}, b::PutBlock{D,C,IdentityRecorder{D}}) where {T,D,C}
    b.content.symbol = length(eb.tensors)+1
    m = T[[1 0]; [0 1]]
    k = 1 
    locs = [b.locs[1]] 
    nlabels = [YaoToEinsum.newlabel!(eb) for _=1:k]
    YaoToEinsum.add_tensor!(eb, reshape(Matrix{T}(m), fill(2, 2k)...), [nlabels..., eb.slots[locs]...])
    eb.slots[locs] .= nlabels
    return eb
end

"""
    QCInfo(data_qubits::Vector{Int},ancilla_qubits::Vector{Int},nq::Int)
    QCInfo(data_qubits::Vector{Int},nq::Int)

A struct to store the qubit information of a quantum circuit.

### Fields
- `data_qubits`: The data qubit indices.
- `ancilla_qubits`: The ancilla qubit indices. If not specified, it is set to the complement of `data_qubits` in `1:nq`
- `nq`: The total number of qubits.
"""
struct QCInfo 
    data_qubits::Vector{Int}
    ancilla_qubits::Vector{Int}
    nq::Int
end
QCInfo(data_qubits::Vector{Int}, nq::Int) = QCInfo(data_qubits, setdiff(1:nq, data_qubits), nq)

function dm_circ(qc::ChainBlock)
    num_qubits = nqubits(qc)
    qcf = chain(2*num_qubits)
    dm_circ!(qcf, qc)
    return qcf
end

function dm_circ!(qcf::ChainBlock, qc::ChainBlock)
    num_qubits = nqubits(qc)
    @assert 2 * num_qubits == nqubits(qcf)
    push!(qcf,subroutine(2*num_qubits, qc, 1:num_qubits))
    push!(qcf,subroutine(2*num_qubits, conj(qc), num_qubits+1:2*num_qubits))
    return qcf
end

function ein_circ(qc::ChainBlock, input_qubits::Vector{Int}, output_qubits::Vector{Int}, num_qubits::Int)
    qc_f = chain(2*num_qubits)
    srs = [SymbolRecorder() for _ in 1:2*(length(input_qubits)+length(output_qubits))]
    [push!(qc_f, put(2*num_qubits, input_qubits[i] => srs[2*i-1]), put(2*num_qubits, num_qubits+input_qubits[i] => srs[2*i])) for i in 1:length(input_qubits)]

    push!(qc_f,qc)
    
    [push!(qc_f, put(2*num_qubits, output_qubits[i] => srs[2*i-1+2*length(input_qubits)]), put(2*num_qubits, num_qubits+output_qubits[i] => srs[2*i+2*length(input_qubits)])) for i in 1:length(output_qubits)]
    return simplify(qc_f; rules=[to_basictypes, Optimise.eliminate_nested]),srs
end

function ein_circ(qc::ChainBlock, input_qubits::Vector{Int}, output_qubits::Vector{Int})
    num_qubits = nqubits(qc)
    return ein_circ(dm_circ(qc), input_qubits, output_qubits, num_qubits)
end

function ein_circ(qc::ChainBlock, qc_info::QCInfo)
    return ein_circ(qc, qc_info.data_qubits, qc_info.data_qubits ∪ qc_info.ancilla_qubits)
end

function qc2enisum(qc::ChainBlock, srs::Vector{SymbolRecorder{D}}, qc_info::QCInfo) where D
    ein_code = yao2einsum(qc;initial_state=Dict(x=>0 for x in qc_info.ancilla_qubits ∪ (qc_info.ancilla_qubits.+qc_info.nq)), optimizer=nothing)
    replace_dict = [srs[4*length(qc_info.data_qubits)+2*i-1].symbol => srs[4*length(qc_info.data_qubits)+2*i].symbol for i in 1:length(qc_info.ancilla_qubits)]
    jointcode = replace(ein_code.code, replace_dict...)
    empty!(jointcode.iy)
    input_indices = [[srs[2*i-1].symbol  for i in 1:length(qc_info.data_qubits)]..., [srs[2*i].symbol  for i in 1:length(qc_info.data_qubits)]...]
    output_indices = [[srs[2*length(qc_info.data_qubits)+2*i-1].symbol  for i in 1:length(qc_info.data_qubits)]..., [srs[2*length(qc_info.data_qubits)+2*i].symbol  for i in 1:length(qc_info.data_qubits)]...]
    push!(jointcode.iy,input_indices...)
    push!(jointcode.iy,output_indices...)
    return TensorNetwork(jointcode, ein_code.tensors), input_indices, output_indices
end

"""
    simulation_tensornetwork(qc::ChainBlock,qc_info::QCInfo)

Generate the tensor network representation of the quantum circuit with the given [`QCInfo`](@ref), where ancilla qubits are initilized at zero state and partial traced after the circuit.

### Arguments
- `qc`: The quantum circuit.
- `qc_info`: The qubit information of the quantum circuit

### Returns
- `tn`: The tensor network representation of the quantum circuit.
- `input_indices`: The input indices of the tensor network.
- `output_indices`: The output indices of the tensor network.
"""
function simulation_tensornetwork(qc::ChainBlock,qc_info::QCInfo)
    qc= simplify(qc; rules=[to_basictypes, Optimise.eliminate_nested])
    qce,srs = ein_circ(qc,qc_info)
    return qc2enisum(qce,srs,qc_info) 
end
"""
    fidelity_tensornetwork(qc::ChainBlock,qc_info::QCInfo)

Generate the tensor network representation of the quantum circuit fidelity with the given [`QCInfo`](@ref), where ancilla qubits are initilized at zero state and partial traced after the circuit. The data qubits are traced out.

### Arguments
- `qc`: The quantum circuit.
- `qc_info`: The qubit information of the quantum circuit

### Returns
- `tn`: The tensor network representation of the quantum circuit.
"""
function fidelity_tensornetwork(qc::ChainBlock,qc_info::QCInfo)
    tn, input_indices,output_indices = simulation_tensornetwork(qc,qc_info)
    jointcode = replace(tn.code, [input_indices[i]=> output_indices[i] for i in 1:length(input_indices)]...)
    empty!(jointcode.iy)
    tn = TensorNetwork(jointcode, tn.tensors)
    tn.tensors[1] = tn.tensors[1]./(4^length(qc_info.data_qubits))
    @show (2^length(qc_info.data_qubits))
    @show tn.tensors[1]
    return tn
end
""" 
    coherent_error_unitary(u::AbstractMatrix{T}, error_rate::Real; cache::Union{Vector, Nothing} = nothing) where T

Generate the error unitary near the given error rate.

### Arguments
- `u`: The original unitary.
- `error_rate`: The error rate.
- `cache`: The vector to store the error rate.

### Returns
- `q`: The errored unitary.
"""
function coherent_error_unitary(u::AbstractMatrix{T}, error_rate::Real; cache::Union{Vector, Nothing} = nothing) where T
    appI = randn(T,size(u))*error_rate + I
    q2 , _ = qr(appI)
    q = u * q2
    cache === nothing || push!(cache, 1 - abs(tr(q'*u)/size(u,1)))
    return Matrix(q)
end

toput(gate::ControlBlock{XGate,1,1}) = put(nqudits(gate), (gate.ctrl_locs..., gate.locs...)=>ConstGate.CNOT)
toput(gate::ControlBlock{ZGate,1,1}) = put(nqudits(gate), (gate.ctrl_locs..., gate.locs...)=>ConstGate.CZ)

toput(gate::ControlBlock{XGate,2,1}) = put(nqudits(gate), (gate.ctrl_locs..., gate.locs...)=>ConstGate.Toffoli)
toput(gate::ControlBlock{ZGate,2,1}) = put(nqudits(gate), (gate.ctrl_locs..., gate.locs...)=>CCZ)
toput(gate::AbstractBlock) = gate


function error_quantum_circuit(qc::ChainBlock, error_rate::T ) where {T <: Real}
    qc= simplify(qc; rules=[to_basictypes, Optimise.eliminate_nested])
    nq = nqubits(qc)
    eqc = chain(nq)
    for _gate in qc
        gate = toput(_gate)
        eu = coherent_error_unitary(mat(gate.content),error_rate)
        push!(eqc,put(nq,gate.locs=>matblock(eu)))
    end
    return eqc
end

"""
    error_quantum_circuit_pair_replace(qc::ChainBlock, error_rate::T ) where {T <: Real}
    error_quantum_circuit_pair_replace(qc::ChainBlock, pairs)

Generate the error quantum circuit for the given error rate.

### Arguments
- `qc`: The quantum circuit.
- `error_rate`: The error rate.
- `pairs`: The error gates used to replace the original gates.

### Returns
- `qcf`: The error quantum circuit.
- `vec`: The vector to store the error rate to each gate.
"""
function error_quantum_circuit_pair_replace(qc::ChainBlock, error_rate::T ) where {T <: Real}
    pairs,vec = error_pairs(error_rate) 
    qcf = error_quantum_circuit_pair_replace(qc,pairs)
    return qcf, vec
end

function error_quantum_circuit_pair_replace(qc::ChainBlock, pairs)
    qcf = replace_block(x->toput(x), qc)
    for pa in pairs
        qcf = replace_block(pa, qcf)
    end
    return qcf
end
"""
    error_pairs(error_rate::T; gates = nothing) where {T <: Real}

Generate the error pairs for the given error rate.

### Arguments
- `error_rate`: The error rate.
- `gates`: The gates to be errored. If not specified, it is set to `[X,Y,Z,H,CCZ,ConstGate.Toffoli,ConstGate.CNOT,ConstGate.CZ]`

### Returns
- `pairs`: The error pairs.
- `vec`: The vector to store the error rate to each gate.
"""
function error_pairs(error_rate::T; gates = nothing) where {T <: Real}
    vec = Vector{T}()
    if gates === nothing
        pairs = [x => matblock(coherent_error_unitary(mat(x),error_rate;cache = vec);tag = "errored $x") for x in [X,Y,Z,H,CCZ,ConstGate.Toffoli,ConstGate.CNOT,ConstGate.CZ]]
    else
        pairs = [x => matblock(coherent_error_unitary(mat(x),error_rate;cache = vec);tag = "errored $x") for x in gates]
    end
    return pairs, vec
end

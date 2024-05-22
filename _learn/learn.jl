using YaoToEinsum
using TensorQEC.Yao
using YaoToEinsum.OMEinsum

mutable struct SymbolRecorder{D} <: TrivialGate{D}
    symbol
end
SymbolRecorder(; nlevel=2) = SymbolRecorder{nlevel}(nothing)
Yao.nqudits(sr::SymbolRecorder) = 1
Yao.print_block(io::IO, sr::SymbolRecorder) = print(io, sr.symbol)

srs = [SymbolRecorder() for i in 1:4]
c = chain(3, put(3, 2=>X), put(3, 1=>Y), control(3, 1, 3=>Y), put(3, 2=>srs[1]))

function YaoToEinsum.add_gate!(eb::YaoToEinsum.EinBuilder{T}, b::PutBlock{D,C,SymbolRecorder{D}}) where {T,D,C}
    lj = eb.slots[b.locs[1]]
    b.content.symbol = lj
    return eb
end

using Test
using TensorQEC

function toric_code_cir(codesize::Int)
    t = TensorQEC.ToricCode(codesize, codesize)
    st = TensorQEC.stabilizers(t)
    qcen, data_qubits, code = TensorQEC.encode_stabilizers(st)
    data_qubit_num = size(code.matrix, 2) ÷ 2
    st_me = TensorQEC.stabilizers(t,linearly_independent = false)
    qcm,st_pos, num_qubits = measure_circuit_fault_tol(st_me)

    bimat = TensorQEC.stabilizers2bimatrix(st_me)
    table = make_table(bimat.matrix, 1)
    qccr = correct_circuit(table, st_pos, num_qubits, data_qubit_num, data_qubit_num)

    qcf = chain(num_qubits)
    push!(qcf, subroutine(num_qubits, qcen, 1:data_qubit_num))
    for i in 1:codesize
        push!(qcf, put(num_qubits, i => X))
    end 
    push!(qcf,qcm)
    push!(qcf,qccr)
    push!(qcf, subroutine(num_qubits, qcen', 1:data_qubit_num))
    return qcf, data_qubits, num_qubits
end

function ein_cir(qc::ChainBlock, data_qubits::Vector{Int}, num_qubits::Int)
    data_qubit_num = length(data_qubits)
    qc_f = chain(2*num_qubits)
    srs = [SymbolRecorder() for i in 1:(2*data_qubit_num+2*num_qubits)]
    srs_num = 1
    for i in data_qubits
        push!(qc_f, put(2*num_qubits, i => srs[srs_num]))
        srs_num += 1
    end

    for i in data_qubits
        push!(qc_f, put(2*num_qubits, num_qubits+i => srs[srs_num]))
        srs_num += 1
    end

    push!(qc_f,subroutine(2*num_qubits, qc, 1:num_qubits))
    push!(qc_f,subroutine(2*num_qubits, qc, num_qubits+1:2*num_qubits))
    for i in data_qubits
        push!(qc_f, put(2*num_qubits, i => srs[srs_num]))
        srs_num += 1
    end

    for i in data_qubits
        push!(qc_f, put(2*num_qubits, num_qubits+i => srs[srs_num]))
        srs_num += 1
    end
    for i in setdiff(1:num_qubits, data_qubits)
        push!(qc_f, put(2*num_qubits, i => srs[srs_num]))
        srs_num += 1
    end
    for i in setdiff(1:num_qubits, data_qubits)
        push!(qc_f, put(2*num_qubits, num_qubits+i => srs[srs_num]))
        srs_num += 1
    end
    return qc_f,srs
end

mapr(a::SymbolRecorder, b::SymbolRecorder) = a.symbol => b.symbol
function toric_code_enisum(qc::ChainBlock,srs::Vector{SymbolRecorder{D}},data_qubits::Vector{Int},num_qubits::Int) where D
    ein_code = yao2einsum(qc;initial_state=Dict(x=>0 for x in setdiff(setdiff(1:2*num_qubits, data_qubits), num_qubits .+ data_qubits)), optimizer=nothing)
    data_qubit_num = length(data_qubits)

    ds1 = 1:data_qubit_num
    ds2 = 2*data_qubit_num+1:3*data_qubit_num
    ds3 = data_qubit_num+1:2*data_qubit_num
    ds4 = 3*data_qubit_num+1:4*data_qubit_num
    anc1 =  4*data_qubit_num+1:num_qubits-data_qubit_num+4*data_qubit_num 
    anc2 = num_qubits+3*data_qubit_num+1:2*num_qubits+2*data_qubit_num
    @show ds1,ds2,ds3,ds4,anc1,anc2

    jointcode = replace(ein_code.code, 
        mapr.(srs[ds1], srs[ds2])..., 
        mapr.(srs[ds3], srs[ds4])..., 
        mapr.(srs[anc1], srs[anc2])...)
    empty!(jointcode.iy) 
    return TensorNetwork(jointcode, ein_code.tensors)
end
qc, data_qubits,num_qubits = toric_code_cir(3)
qcf,srs = ein_cir(qc,data_qubits,num_qubits)
tn = toric_code_enisum(qcf,srs,data_qubits,num_qubits)
optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())
contract(optnet)/2^8

vizcircuit(qcf; filename = "_learn/toric_code.svg")
function YaoPlots.draw!(c::YaoPlots.CircuitGrid, p::SymbolRecorder, address, controls)
    @assert length(controls) == 0
    YaoPlots._draw!(c, [(getindex.(Ref(address), (1,)), c.gatestyles.g, "$(p.symbol)")])
end


using Random
Random.seed!(0)
toyqc = chain(4, put(4, (1,2) => GeneralMatrixBlock(rand_unitary(4); nlevel=2, tag="randu1")), put(4, (1,2,3,4) => GeneralMatrixBlock(rand_unitary(16); nlevel=2, tag="randu2")))
qcf,srs = ein_cir(toyqc,[1,2],4)
tn = toric_code_enisum(qcf,srs,[1,2],4)
optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())
contract(optnet)
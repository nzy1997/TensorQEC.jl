using TensorInference
using TensorInference: Factor
using Test
using Yao
using Base.Iterators: product
using LinearAlgebra

struct Gate
	unitary::Matrix{Complex{Float64}}
	qubits::Vector{Int}
end

struct QuantumCircuit
	n_qubits::Int
	gates::Vector{Gate}
end

const CNOT = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
const S = ComplexF64[1 0; 0 im]

@testset "qc" begin
	qc = QuantumCircuit(3, [Gate(CNOT, [1, 2]), Gate(CNOT, [2, 3])])
	@test qc isa QuantumCircuit
end

function pauli_basis(nqubits::Int)
	paulis = Matrix.([I2, X, Y, Z])
	return [(length(pauli) == 1 ? pauli[1] : kron(pauli...)) for pauli in product(fill(paulis, nqubits)...)]
end

@testset "pauli_basis" begin
	@test pauli_basis(1) == [Matrix(I2), Matrix(X), Matrix(Y), Matrix(Z)]
end

function pauli_decomposition(m::AbstractMatrix)
	nqubits = Int(log2(size(m, 1)))
	return [tr(pauli * m) for pauli in pauli_basis(nqubits)] / (2^nqubits)
end

@testset "pauli_decomposition" begin
	@test pauli_decomposition(Matrix(H)) == [0, 1, 0, 1] / sqrt(2)
	@test pauli_decomposition(Matrix(kron(X, X))) == [0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0]
end

function matrix2vals(m::AbstractMatrix)
	nqubits = Int(log2(size(m, 1)))
	return reshape(reduce(hcat,[pauli_decomposition(m * pauli * m') for pauli in pauli_basis(nqubits)]),fill(4, 2*nqubits)...)
end

@testset "matrix2vals" begin
	@test matrix2vals(Matrix(X)) == [1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
    p1=pauli_basis(1)
    for i in 1:4
        for j in 1:4
            for k in 1:4
                for l in 1:4
                    @test tr(CNOT * kron(p1[i], p1[j]) * CNOT * kron(p1[k], p1[l]))/4 == matrix2vals(CNOT)[k,l,i,j]
					SS=kron(S,S)
					@test tr(SS * kron(p1[i], p1[j]) * SS' * kron(p1[k], p1[l]))/4 == matrix2vals(SS)[k,l,i,j]
                end
            end
        end
    end
end

function pauli_mapping(m::AbstractMatrix)
	nqubits = Int(log2(size(m, 1)))
	paulis = pauli_basis(nqubits)
	return [Float64(tr(pj * m * pi * m')/size(m, 1)) for pi in paulis, pj in paulis]
end

function matrix2factor(g::Gate, input_vars::Vector{Int}, pins::Vector{Int})
	vals = pauli_mapping(g.unitary)
	vars = (input_vars..., pins...)
	return Factor(vars, vals)
end

@testset "matrix2factor" begin
	g = Gate(CNOT, [1, 2])
	f = matrix2factor(g, [1, 2], [3, 4])
	@test f.vars == (1, 2, 3, 4)
end



#every qubit must be encoded,othervise this code may not work.
#syn is a vector of 0,1,2,3. 0 means |0>, 1 means |1>, 2 means dataqubit.


struct CliffordNetwork{T<:Real}
	factors::Vector{Factor{T}}
	physical_qubits::Vector{Int}
	mapped_qubits::Vector{Int}
	function CliffordNetwork(factors::Vector{Factor{T}}, physical_qubits::Vector{Int}, mapped_qubits::Vector{Int}) where T
		@assert length(physical_qubits) == length(mapped_qubits)
		new{T}(factors, physical_qubits, mapped_qubits)
	end
end
Yao.nqubits(cl::CliffordNetwork) = length(cl.physical_qubits)

function newgate!(pins::Vector{Int}, gate::Gate, nvars::Int)
	input_gate_vars = pins[gate.qubits]
	output_gate_vars = collect(nvars+1:nvars+length(gate.qubits))
	pins[gate.qubits] .= output_gate_vars
	gate_tensor = matrix2factor(gate, input_gate_vars, output_gate_vars)
	return gate_tensor
end
function clifford_network(qc::QuantumCircuit)
nvars = qc.n_qubits
	pins = collect(1:nvars)
	factors = Vector{Factor{Float64}}()
	# add gates
	for gate in qc.gates
		push!(factors, newgate!(pins, gate, nvars))
		nvars += length(gate.qubits)
	end
	return CliffordNetwork(factors, pins, collect(1:qc.n_qubits))
end

function _generate_tensor_network(cl::CliffordNetwork, ps::Vector{Vector{Float64}}, qs::Dict{Int, Vector{Float64}})
	factors = copy(cl.factors)
	# add prior distributions of Pauli errors on physical qubits
	for (k, i) in enumerate(cl.physical_qubits)
		push!(factors, Factor{Float64,1}((i,), ps[k]))
	end
	for (k, v) in qs
		push!(factors, Factor{Float64,1}((cl.mapped_qubits[k],), v))
	end
	openvars = setdiff(cl.mapped_qubits, keys(qs))
	return TensorNetworkModel(
		1:length(factors),
		fill(4, length(factors)),
		factors,
		openvars = openvars
	)
end

function generate_tensor_network(cl::CliffordNetwork, p::AbstractVector{<:Real}, syndromes::Dict{Int, Bool})
	_generate_tensor_network(cl, fill(p, nqubits(cl)), Dict(i => vector_syndrome(s) for (i, s) in syndromes))
end

vector_syndrome(measure_outcome) = iszero(measure_outcome) ? Bool[1,0,0,1] : Bool[0,1,1,0]
function projector(v::AbstractVector{Bool})
	locs = findall(!iszero, v)
	n = length(v)
	return Matrix{Bool}(I, n, n)[locs, :]
end

@testset "projector" begin
	@test projector(Bool[1,0,0,1]) == Bool[1 0 0 0; 0 0 0 1]
	@test projector(Bool[0,1,1,0]) == Bool[0 1 0 0; 0 0 1 0]
end

function circuit2tensornetworks(qc::QuantumCircuit, p::AbstractVector{<:Real})
	_circuit2tensornetworks(qc, fill(p, qc.n_qubits))
end

function _circuit2tensornetworks(qc::QuantumCircuit, ps)
	cl = clifford_network(qc)
	_generate_tensor_network(cl, ps, Dict{Int,Vector{Float64}}())
end

@testset "circuit2tensornetworks" begin
	qc = QuantumCircuit(3, [Gate(CNOT, [1, 2]), Gate(CNOT, [2, 3])])
	p=Float64[1,0,0,0]
	tn = circuit2tensornetworks(qc,p)
	@test tn isa TensorNetworkModel
end

# |ψ> ---.------- p
#        |
# |0> ---X---.--- p
#            |
# |0> -------X--- p
@testset "tensor network mapping" begin
	# create a circuit and convert it to the pauli basis
	yaoqc = chain(cnot(3, 3, 2),cnot(3, 2, 1))
	yaopauli = pauli_mapping(mat(ComplexF64, yaoqc))

	# tensor network mapping of a quantum circuit
	qc = QuantumCircuit(3, [Gate(CNOT, [1, 2]), Gate(CNOT, [2, 3])])
	for ci in CartesianIndices((fill(4, 3)...,))
		ps = [Yao.BitBasis._onehot(Float64, 4, ci.I[i]) for i in 1:3]
		tn = _circuit2tensornetworks(qc, ps)
		p1 = probability(tn)
		p2 = yaopauli[:,:,:,ci.I...]
		@test p1 ≈ p2
	end
end

@testset "most_probable_config" begin
	qc = QuantumCircuit(3, [Gate(CNOT, [1, 2]), Gate(CNOT, [2, 3])])
	p=Float64[1,0,0,0]
	syn=[0,0,3]
	tn = _circuit2tensornetworks(qc, fill(p, qc.n_qubits); syn=syn)
	cfg = probability(tn)
	@test cfg==[1,0,0,0]
end

#syn is a vector of 0,1,2,3. 0 means |0>, 1 means |1>, 2 means dataqubit, 3 means open.
function syndrome_inference(qc::QuantumCircuit, syn::Vector{Int64}, p::Vector{Vector{Float64}})
	syn_inf = fill(0,qc.n_qubits)
	for i in 1:qc.n_qubits
		if syn[i] == 0 || syn[i] == 1
			temp=syn[i]
			syn[i] = 2
			tn = _circuit2tensornetworks(qc,p; syn=syn)
			logp,cfg=most_probable_config(tn)

			syn[i] = temp
		elseif syn[i] == 2
			continue;
		else
			error("Invalid syndrome")
		end
	end

	return syn_inf
end

@testset "syndrome_inference" begin
	qc = QuantumCircuit(3, [Gate(mat(ComplexF64,I4), [1, 2])])
	p=Float64[0,0.3,0.6,0]
	syn=fill(1,3)
	syn_inf=syndrome_inference(qc,syn,fill(p,qc.n_qubits))
	@test syn_inf == [0,0,0]
end



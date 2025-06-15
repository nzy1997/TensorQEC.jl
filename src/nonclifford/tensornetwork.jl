struct CliffordNetwork{T<:Real}
	factors::Vector{Factor{T}}
	physical_qubits::Vector{Int}
	mapped_qubits::Vector{Int}
	nvars::Int
function CliffordNetwork(factors::Vector{Factor{T}}, physical_qubits::Vector{Int}, mapped_qubits::Vector{Int}, nvars::Int) where T
		@assert length(physical_qubits) == length(mapped_qubits) "got: $(length(physical_qubits)) != $(length(mapped_qubits))"
		new{T}(factors, physical_qubits, mapped_qubits, nvars)
	end
end
Yao.nqubits(cl::CliffordNetwork) = length(cl.physical_qubits)

# TODO: limit the gates to be only Clifford gates
function newgate!(pins::Vector{Int}, gate::PutBlock, nvars::Int)
    # @assert is_clifford(gate)
    locs = gate.locs
	input_gate_vars = [pins[i] for i in locs]
	output_gate_vars = collect(nvars+1:nvars+length(locs))
	pins[collect(locs)] .= output_gate_vars
	gate_tensor = matrix2factor(content(gate), input_gate_vars, output_gate_vars)
	return gate_tensor, nvars+length(locs)
end
function tensor_prepend!(prepins::Vector{Int}, tensor::AbstractMatrix, loc::Integer, nvars::Integer)
	nvars += 1
	prevar = prepins[loc]
	prepins[loc] = nvars
	return Factor((nvars, prevar), tensor), nvars
end
function tensor_prepend!(prepins::Vector{Int}, tensor::AbstractVector, loc::Integer, nvars::Integer)
	return Factor((prepins[loc],), tensor), nvars
end
convert_to_put(g::PutBlock) = g
function convert_to_put(g::ControlBlock)
    nc, ng = length(g.ctrl_locs), length(g.locs)
    locs = (g.ctrl_locs..., g.locs...)
    return put(nqubits(g), locs =>ControlBlock(length(locs), (1:nc...,), g.ctrl_config, content(g), (nc+1:nc+ng...,)))
end

# convert a gate to a Factor
function matrix2factor(g::AbstractBlock{D}, input_vars::Vector{Int}, pins::Vector{Int}) where D
	vals = reshape(pauli_repr(mat(ComplexF64, g)), ntuple(_ -> D^2, 2 * nqudits(g)))
	vars = (pins..., input_vars...)
	return Factor(vars, vals)
end

"""
	clifford_network(qc::ChainBlock)

Generate a Clifford network from a quantum circuit.
"""
function clifford_network(qc::ChainBlock)
	nvars = nqubits(qc)
	pins = collect(1:nvars)
	factors = Vector{Factor{Float64}}()
	# add gates
	for gate in qc
    	_gate = convert_to_put(gate)
		factor, nvars = newgate!(pins, _gate, nvars)
		push!(factors, factor)
	end
	return CliffordNetwork(factors, pins, collect(1:length(pins)), nvars)
end

@enum ExtraTensor UNITY4 PXY PIZ PXY_UNITY2 PIZ_UNITY2

struct BoundarySpec{T}
	pauli_coeffs::NTuple{4, T}
	attach_unity::Bool
end

# generate a tensor network from a CliffordNetwork instance
# `p` is the prior distribution of Pauli errors on physical qubits
# `syndromes` is a dictionary of syndromes, where the key is the index of the syndrome and the value is the prior distribution of the syndrome
function generate_tensor_network(cl::CliffordNetwork{T}, ps::Dict{Int, BoundarySpec{T}}, qs::Dict{Int, BoundarySpec{T}}) where T
	# TODO: assert qs locations are valid
	nvars=cl.nvars
	factors = copy(cl.factors)
	cards = fill(4, nvars)
	mars = Vector{Vector{Int}}()
	# add prior distributions of Pauli errors on physical qubits
	for (k, v) in ps
		nvars = _add_boundary!(cl.physical_qubits, v, k, factors, cards, mars, nvars)
	end
	for (k, v) in qs
		nvars = _add_boundary!(cl.mapped_qubits, v, k, factors, cards, mars, nvars)
	end
	return TensorNetworkModel(
		UAIModel(nvars, cards, factors);
		# openvars are open indices in the tensor network
		openvars = cl.mapped_qubits[setdiff(1:nqubits(cl), keys(qs))] ∪ cl.physical_qubits[setdiff(1:nqubits(cl), keys(ps))],
		unity_tensors_labels = mars
	)
end
function _add_boundary!(openvars::AbstractVector, v::BoundarySpec{T}, k::Int, factors::Vector{Factor{T}}, cards::Vector{Int}, mars::Vector{Vector{Int}}, nvars::Int) where T
	if v.attach_unity
		# boundary tensor is a projector
		pmat = projector(T, collect(v.pauli_coeffs))
		factor, nvars = tensor_prepend!(openvars, pmat, k, nvars)
		push!(cards, size(pmat, 1))
		push!(factors, factor)
		push!(mars, [nvars])
	else
		# boundary tensor is a vector
		factor, nvars = tensor_prepend!(openvars, collect(v.pauli_coeffs), k, nvars)
		push!(factors, factor)
	end
	return nvars
end

vector_syndrome(measure_outcome) = iszero(measure_outcome) ? Bool[1,0,0,1] : Bool[0,1,1,0]
function projector(::Type{T}, v::AbstractVector) where T
	locs = findall(!iszero, v)
	return Diagonal(T.(v))[locs, :]
end

function circuit2tensornetworks(qc::ChainBlock, ps::Dict, qs::Dict)
	cl = clifford_network(qc)
	generate_tensor_network(cl, ps, qs)
end
function simple_circuit2tensornetworks(qc::ChainBlock, ps::AbstractVector{Vector{T}}) where T<:Real
	circuit2tensornetworks(qc,
		Dict([i=>BoundarySpec((ps[i]...,), false) for i in 1:nqubits(qc)]),
		Dict{Int, BoundarySpec{T}}()
	)
end

function Yao.expect(operator::PauliString, cl::CliffordNetwork{T}, rho::PauliString) where T
	n = nqubits(cl)
	# construct the tensor network
	ps = Dict([i=>BoundarySpec((Yao.BitBasis._onehot(T, 4, rho.operators[i].id + 1)...,), false) for i in 1:n])
	qs = Dict([i=>BoundarySpec((Yao.BitBasis._onehot(T, 4, operator.operators[i].id + 1)...,), false) for i in 1:n])
	# TODO: avoid repeated optimization of contraction order
	tn = generate_tensor_network(cl, ps, qs)
	return probability(tn)*2^n
end
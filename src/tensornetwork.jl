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
function matrix2factor(g::AbstractBlock, input_vars::Vector{Int}, pins::Vector{Int})
	vals = pauli_mapping(mat(ComplexF64, g))
	vars = (pins..., input_vars...)
	return Factor(vars, vals)
end

# Yao circuit to a CliffordNetwork instance.
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
	mars = Int[]
	# add prior distributions of Pauli errors on physical qubits
	for (k, v) in ps
		nvars = _add_boundary!(cl.physical_qubits, v, k, factors, cards, mars, nvars)
	end
	for (k, v) in qs
		nvars = _add_boundary!(cl.mapped_qubits, v, k, factors, cards, mars, nvars)
	end
	return TensorNetworkModel(
		1:nvars,
		cards,
		factors;
		# openvars are open indices in the tensor network
		openvars = cl.mapped_qubits[setdiff(1:nqubits(cl), keys(qs))] ∪ cl.physical_qubits[setdiff(1:nqubits(cl), keys(ps))],
		mars = [[l] for l in [cl.physical_qubits; cl.mapped_qubits]]
	)
end
function _add_boundary!(openvars::AbstractVector, v::BoundarySpec{T}, k::Int, factors::Vector{Factor{T}}, cards::Vector{Int}, mars::Vector{Int}, nvars::Int) where T
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
	n = length(v)
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

# rho0.state is the density matrix.
function Yao.expect(operator::SumOfPaulis, cl::CliffordNetwork, rho0::SumOfPaulis)
	# step 2: use the pauli decomposition as the input at the physical qubits
	for (c, p) in rho0.items
		for (d, q) in operator.items
			result += c * d * expect(PauliString(q), cl, PauliString(p))
		end
	end
end

function expect(p::PauliString, cl::CliffordNetwork, rho0::PauliString)
	# construct the tensor network
	ps = Dict([i=>BoundarySpec((p.ids[i]...,), false) for i in 1:nqubits(cl)])
	tn = generate_tensor_network(cl, ps, qs)
end

function hh()
	push!(cl.factors, Factor((cl.mapped_qubits...,), rho0))
	# step 3: fix the mapped qubits to a specific pauli string.
	for (loc, gate) in operator
		idx = findfirst(==(gate), [I2, X, Y, Z])
		@assert idx !== nothing "gate error, got: $gate"
		factor, nvars = tensor_prepend!(cl.physical_qubits, pauli_decomposition(mat(gate)), loc[1], cl.nvars)
		push!(cl.factors,factor)
	end
	# step 4: calculate the expectation value by contracting the tensor network
	tn=TensorNetworkModel(
		1:nvars,
		fill(4, nvars),
		cl.factors;
		# openvars are open indices in the tensor network
		mars = []
	)
	return probability(tn)
end
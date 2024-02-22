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
	return gate_tensor
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
		push!(factors, newgate!(pins, _gate, nvars))
		nvars += length(_gate.locs)
	end
	return CliffordNetwork(factors, pins, collect(1:length(pins)), nvars)
end

function _generate_tensor_network(cl::CliffordNetwork, ps::Vector{Vector{Float64}}, qs::Dict{Int, Vector{Float64}})
	factors = copy(cl.factors)
	nvars=cl.nvars
	# add prior distributions of Pauli errors on physical qubits
	for (k, i) in enumerate(cl.physical_qubits)
		push!(factors, Factor{Float64,1}((i,), ps[k]))
	end
	openvars = setdiff(cl.mapped_qubits, keys(qs))
	for (k, v) in qs
		nvars=nvars+1
		push!(factors, Factor{Float64,1}((cl.mapped_qubits[k],nvars), v))
		push!(openvars, nvars)
	end
	return TensorNetworkModel(
		1:nvars,
		fill(4, cl.nvars),
		factors,
		openvars = openvars
	)
end

# generate a tensor network from a CliffordNetwork instance
# `p` is the prior distribution of Pauli errors on physical qubits
# `syndromes` is a dictionary of syndromes, where the key is the index of the syndrome and the value is the prior distribution of the syndrome
function generate_tensor_network(cl::CliffordNetwork, p::AbstractVector{<:Real}, syndromes::Dict{Int, Bool})
	_generate_tensor_network(cl, fill(p, nqubits(cl)), Dict(i => projector(vector_syndrome(s)) for (i, s) in syndromes))
end

vector_syndrome(measure_outcome) = iszero(measure_outcome) ? Bool[1,0,0,1] : Bool[0,1,1,0]
function projector(v::AbstractVector{Bool})
	locs = findall(!iszero, v)
	n = length(v)
	return Matrix{Bool}(I, n, n)[locs, :]
end

function circuit2tensornetworks(qc::ChainBlock, ps)
	cl = clifford_network(qc)
	_generate_tensor_network(cl, ps, Dict{Int,Vector{Float64}}())
end

#syn is a vector of 0,1,2,3. 0 means |0>, 1 means |1>, 2 means dataqubit, 3 means open.
function syndrome_inference(qc::ChainBlock, syn::Vector{Int64}, p::Vector{Vector{Float64}})
    nvars = nqubits(qc)
	syn_inf = fill(0,nvars)
	for i in 1:nvars
		if syn[i] == 0 || syn[i] == 1
			temp=syn[i]
			syn[i] = 2
			tn = _circuit2tensornetworks(qc,p; syn=syn)
			logp, cfg=most_probable_config(tn)

			syn[i] = temp
		elseif syn[i] == 2
			continue;
		else
			error("Invalid syndrome")
		end
	end

	return syn_inf
end


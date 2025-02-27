"""
	measure_syndrome!(reg::AbstractRegister, stabilizers::AbstractVector{PauliString{N}}) where N

Measure the given stabilizers.

### Arguments

- `reg`: The quantum register.
- `stabilizers`: The vector of pauli strings, composing the generator of stabilizer group.

### Returns
- `measure_outcome`: The measurement outcome of the stabilizers, which is either 1 or -1.
"""
function measure_syndrome!(reg::AbstractRegister, stabilizers::AbstractVector{PauliString{N}}) where N
	measure_oprators = [Yao.YaoBlocks.Optimise.to_basictypes(ps) for ps in stabilizers]
	return [round(Int, real.(measure!(op,reg))) for op in measure_oprators]
end

function syndrome_transform(bimat::CSSBimatrix,syn::Vector{Int})
	return bimat.Q * (syn .≈ -1)
end

function generate_syndrome_dict(bimat::CSSBimatrix, syn::Vector{Mod2})
	return Dict([bimat.ordering[i]=>syn[i].x for i in 1:size(bimat.Q,2)])
end

"""
	transformed_syndrome_dict(measure_outcome::Vector{Int}, code::CSSBimatrix)

Generate the syndrome dictionary on the transformed stabilizers from the measurement outcome. 

### Arguments
- `measure_outcome`: The measurement outcome of the stabilizers, which is either 1 or -1.
- `code`: The structure storing the encoding information.

### Returns
- `syn_dict`: The syndrome dictionary on the transformed stabilizers. 1 is transformed to 0, -1 is transformed to 1. 
"""
function transformed_syndrome_dict(measure_outcome::Vector{Int}, code::CSSBimatrix)
	return generate_syndrome_dict(code, syndrome_transform(code, measure_outcome))
end

"""
	syndrome_inference(cl::CliffordNetwork{T}, syn::Dict{Int,Bool}, p::Vector{Vector{Float64}}) where T

Infer the error probability of each qubit from the measurement outcome of the stabilizers.

### Arguments
- `cl`: The Clifford network.
- `syn`: The syndrome dictionary.
- `p`: The prior error probability of each physical qubit.

### Returns
- `pinf`: The inferred error probability of each physical qubit in coding space. For errored stabilizers, there are two possibilities: X or Y error. For unerrored stabilizers, there are also two possibilities: no error or Z error. For unmeasured qubits, there are four possibilities: no error, X error, Y error, Z error. Therefore the length of each vector in `pinf` may be 2 or 4.
"""
function syndrome_inference(cl::CliffordNetwork{T}, syn::Dict{Int,Bool}, p::Vector{Vector{Float64}})where T
	n = length(p)
	ps = Dict([i=>BoundarySpec((p[i]...,), false) for i in 1:n])
	qs = Dict([i=>BoundarySpec((ones(T,4)...,), true) for i in 1:n])
	for (k, v) in syn
		qs[k] = BoundarySpec((v ?  (0.0,1.0,1.0,0.0) : (1.0,0.0,0.0,1.0) ),true)
	end
	tn = generate_tensor_network(cl, ps, qs)
	mp = marginals(tn)
	return Dict([k => mp[[cl.mapped_qubits[k]]] for k in 1:n])
end

"""
	correction_pauli_string(qubit_num::Int, syn::Dict{Int, Bool}, prob::Dict{Int, Vector{Float64}})

Generate the error Pauli string in the coding space. To correct the error, we still need to transform it to the physical space.

### Arguments
- `qubit_num`: The number of qubits.
- `syn`: The syndrome dictionary.
- `prob`: The inferred error probability of each physical qubit in coding space.

### Returns
- `ps`: The error Pauli string in the coding space.
"""
function correction_pauli_string(qubit_num::Int, syn::Dict{Int, Bool}, prob::Dict{Int, Vector{Float64}})
	ps = ones(Int, qubit_num)
	for (k, v) in prob
		if k ∈ keys(syn)
			if syn[k] #syn[k] is true, measure outcome is -1, use X or Y
				ps[k] = (findmax(v)[2]) == 1 ? 2 : 3
			elseif findmax(v)[2] == 2
				ps[k] = 4
			end
		else
            ps[k] = [1,2,3,4][findmax(v)[2]]
		end
	end
	return PauliString(ps[1:end]...)
end

"""
	inference(measure_outcome::Vector{Int}, code::CSSBimatrix, qc::ChainBlock, p::Vector{Vector{Float64}})

Infer the error probability of each qubit from the measurement outcome of the stabilizers.

### Arguments
- `measure_outcome`: The measurement outcome of the stabilizers, which is either 1 or -1.
- `code`: The structure storing the encoding information.
- `qc`: The encoding circuit.
- `p`: The prior error probability of each physical qubit.

### Returns
- `ps_ec_phy`: The error Pauli string for error correction.
"""
function inference(measure_outcome::Vector{Int}, code::CSSBimatrix, qc::ChainBlock, p::Vector{Vector{Float64}}) 
	syn_dict = generate_syndrome_dict(code, syndrome_transform(code, measure_outcome))
	cl = clifford_network(qc)
	pinf = syndrome_inference(cl, syn_dict, p)
	ps_ec_phy = pauli_string_map_iter(correction_pauli_string(nqubits(qc), syn_dict, pinf), qc)
	return ps_ec_phy
end
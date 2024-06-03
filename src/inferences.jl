function measure_syndrome!(reg::AbstractRegister, stabilizers::AbstractVector{PauliString{N}}) where N
	measure_oprators = [Yao.YaoBlocks.Optimise.to_basictypes(ps) for ps in stabilizers]
	return [round(Int, real.(measure!(op,reg))) for op in measure_oprators]
end

function syndrome_transform(bimat::Bimatrix,syn::Vector{Int})
	return bimat.Q * (syn .≈ -1)
end

function generate_syndrome_dict(bimat::Bimatrix, syn::Vector{Mod2})
	return Dict([bimat.ordering[i]=>syn[i].x for i in 1:size(bimat.Q,2)])
end

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

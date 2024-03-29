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



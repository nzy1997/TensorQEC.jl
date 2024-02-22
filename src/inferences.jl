

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

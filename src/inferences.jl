
function syndrome_inference(cl::CliffordNetwork{T}, syn::Dict{Int,Bool}, p::Vector{Vector{Float64}})where T
	n = length(p)
	ps = Dict([i=>BoundarySpec((p[i]...,), false) for i in 1:n])
	qs = Dict([i=>BoundarySpec((ones(T,4)...,), false) for i in 1:n])
	for (k, v) in syn
		qs[k] = BoundarySpec((v ? (0.0,1.0,1.0,0.0) : (1.0,0.0,0.0,1.0)),true)
	end
	tn = generate_tensor_network(cl, ps, qs)
	return marginals(tn)
end

struct SpinGlassSA{T, VI<:AbstractVector{Int}, MM<:AbstractMatrix{Mod2}, MT<:AbstractMatrix{T}} <: CompiledDecoder
	s2qx::VI
	s2q_ptrx::VI
	s2qz::VI
	s2q_ptrz::VI
	lx::MM
	lz::MM
	xlogical_qubits::VI
	xlogical_qubits_ptr::VI
	zlogical_qubits::VI
	zlogical_qubits_ptr::VI
	logpx_diff::MT
	logpz_diff::MT
	betas::Vector{T}
	num_trials::Int
	tanner::CSSTannerGraph
end


function generate_spin_glass_sa(tanner::CSSTannerGraph, ide::IndependentDepolarizingError{T}, betas::Vector{T}, num_trials::Int) where {T}
	lx,lz = logical_operator(tanner)
	xlogical_qubits = [findall(i->i.x,row) for row in eachrow(lx)]
	zlogical_qubits = [findall(i->i.x,row) for row in eachrow(lz)]
	logp_xerror = log.(ide.px)
	logp_yerror = log.(ide.py)
	logp_zerror = log.(ide.pz)
	logp_noerror = log.(one(T) .- ide.px .- ide.py .- ide.pz)
	logp_i2x = logp_noerror - logp_xerror
	logp_i2z = logp_noerror - logp_zerror
	logp_x2y = logp_xerror - logp_yerror
	logp_z2y = logp_zerror - logp_yerror
	logpx_diff = [logp_i2x;; -logp_i2x;; logp_z2y;; -logp_z2y]
	logpz_diff = [logp_i2z;; logp_x2y;; -logp_i2z;; -logp_x2y]
	s2qx, s2q_ptrx = _vecvec2vecptr(tanner.stgx.s2q)
	s2qz, s2q_ptrz = _vecvec2vecptr(tanner.stgz.s2q)
	xlogical_qubits,xlogical_qubits_ptr = _vecvec2vecptr(xlogical_qubits)
	zlogical_qubits,zlogical_qubits_ptr = _vecvec2vecptr(zlogical_qubits)
	return SpinGlassSA(s2qx, s2q_ptrx, s2qz, s2q_ptrz, lx, lz, xlogical_qubits, xlogical_qubits_ptr, zlogical_qubits, zlogical_qubits_ptr, logpx_diff, logpz_diff, betas, num_trials, tanner)
end

"""
	anneal_run!(config, sap, betas::Vector{Float64}, num_sweep::Int)

Perform Simulated Annealing using Metropolis updates for the run.

# Arguments
* `config`: configuration that can be updated.
* `sap`: problem with `energy`, `flip!` and `random_config` interfaces.
* `betas`: inverse temperature scales, i.e. it can be [1.0, 0.5, 0.0]

# Keyword arguments
* `num_sweep`: the number of sweeps.
* `num_sweep_thermalize`: the number of sweeps for thermalization.
* `ptemp`: the probability of proposing a new beta.
* `rng`: the random number generator.
* `eta_lr`: the learning rate of eta.

Returns a named tuple: (; optimal_cost, optimal_configuration, p1, mostlikely_configuration, acceptance_rate, beta_acceptance_rate, valid_samples, etas).
"""
function anneal_run!(config::CSSErrorPattern, sap::SpinGlassSA{T,VI,MM,MT}; rng = Random.Xoshiro()) where {T,VI,MM,MT}
	betas = sap.betas
	num_trials = sap.num_trials
	logical_num = length(sap.xlogical_qubits_ptr) - 1
	logical_count = fill(0.0,fill(4,logical_num)...)

	for _ in 1:num_trials
		for beta in betas
			try_flip!(config,config.xerror, sap.s2qx, sap.s2q_ptrx, rng, beta,sap.logpx_diff,T)
			try_flip!(config,config.zerror, sap.s2qz, sap.s2q_ptrz, rng, beta,sap.logpz_diff,T)
			try_flip!(config,config.xerror, sap.xlogical_qubits, sap.xlogical_qubits_ptr, rng, beta,sap.logpx_diff,T)
			try_flip!(config,config.zerror, sap.zlogical_qubits, sap.zlogical_qubits_ptr, rng, beta,sap.logpz_diff,T)
		end

		logical_count[[(sum(config.xerror[sap.zlogical_qubits[sap.zlogical_qubits_ptr[i]:sap.zlogical_qubits_ptr[i+1]-1]]).x ? 2 : 1) + (sum(config.zerror[sap.xlogical_qubits[sap.xlogical_qubits_ptr[i]:sap.xlogical_qubits_ptr[i+1]-1]]).x ? 2 : 0) for i in 1:logical_num]...] += 1.0
	end
	return logical_count./num_trials
end

"""
	flip!(config::AnnealingConfig, ispin::Proposal, ap::AnnealingProblem) -> SpinConfig

Apply the change to the configuration.
"""
function flip!(config,fliplocs)
	for q in fliplocs
		@inbounds config[q] = config[q] .+ Mod2(1)
	end
	return config
end
struct SimulatedAnnealing{T} <: AbstractDecoder
	betas::Vector{T}
	num_trials::Int
	use_cuda::Bool
end

function compile(decoder::SimulatedAnnealing, problem::CSSDecodingProblem)
	prob = generate_spin_glass_sa(problem.tanner, problem.pvec, decoder.betas, decoder.num_trials)
	return decoder.use_cuda ? togpu(prob) : prob
end
togpu(prob) = error("CUDA extension not loaded, try `using CUDA`")

function decode(cm::SpinGlassSA, syndrome::CSSSyndrome)
	config = CSSErrorPattern(TensorQEC._mixed_integer_programming_for_one_solution(cm.tanner, syndrome)...)
    res = anneal_run!(config, cm)
	_, pos = findmax(res)
	lx = cm.lx
	lz = cm.lz
	# 1:I, 2:X, 3:Z, 4:Y
	for i in axes(lx,1)
		(sum(lz[i,:].* config.xerror).x && (!iszero(pos.I[i]%2))) && (config.xerror .+= lx[i,:])
		(sum(lx[i,:].* config.zerror).x && (pos.I[i] < 3)) && (config.zerror .+= lz[i,:])
    end
	return CSSDecodingResult(true,config)
end

function _vecvec2vecptr(vecvec::Vector{Vector{T}}) where T
	vec = vcat(vecvec...)
    ptr = cumsum([1, length.(vecvec)...])
	return vec, ptr
end

function try_flip!(config,xerror, vec,ptr, rng, beta,logpx_diff,T)
	for index in 1:length(ptr)-1
		fliplocs = view(vec, ptr[index]:ptr[index+1]-1)
		ΔE = zero(T)
		@inbounds for i in fliplocs
			old_pos = (config.xerror[i].x ? 2 : 1) + (config.zerror[i].x ? 2 : 0)
			ΔE += logpx_diff[i,old_pos]
		end

		if ΔE <= 0
			flip!(xerror, fliplocs)
		else
			prob = @fastmath exp(-beta * ΔE)
			if rand(rng) < prob
				flip!(xerror, fliplocs)
			end
		end
	end
end
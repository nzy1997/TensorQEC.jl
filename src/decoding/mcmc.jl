struct SpinGlassSA{T}
	tanner::CSSTannerGraph
	xlogical_qubits::Vector{Vector{Int}}
	zlogical_qubits::Vector{Vector{Int}}
	logp_i2x::Vector{T}
	logp_i2z::Vector{T}
	logp_x2y::Vector{T}
	logp_z2y::Vector{T}
end

function generate_spin_glass_sa(tanner::CSSTannerGraph, ide::IndependentDepolarizingError{T}) where {T}
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

	return SpinGlassSA(tanner, xlogical_qubits, zlogical_qubits, logp_i2x, logp_i2z, logp_x2y, logp_z2y)
end

"""
	anneal_singlerun!(config::AnnealingConfig, prob, betas::Vector{Float64}, num_sweep::Int)

Perform Simulated Annealing using Metropolis updates for the single run.

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
function anneal_singlerun!(config::CSSErrorPattern, sap::SpinGlassSA{T}, betas::Vector{T}, num_trials::Int; rng = Random.Xoshiro()) where {T}
	logical_num = length(sap.xlogical_qubits)
	logical_count = fill(0,fill(4,logical_num)...)
	xtry = [sap.xlogical_qubits..., sap.tanner.stgx.s2q...]
	ztry = [sap.zlogical_qubits..., sap.tanner.stgz.s2q...]
	for id in 1:num_trials
		for beta in betas
			#!!!Not dry
			for fliplocs in xtry
				ΔE = zero(T)
				@inbounds for i in fliplocs
					if config.xerror[i].x && config.zerror[i].x # y -> z
						ΔE -= sap.logp_z2y[i]
					elseif config.xerror[i].x && !config.zerror[i].x # x -> i
						ΔE -= sap.logp_i2x[i]
					elseif !config.xerror[i].x && config.zerror[i].x # z -> y
						ΔE += sap.logp_z2y[i]
					else # i -> x
						ΔE += sap.logp_i2x[i]
					end
				end

                if ΔE <= 0
					flip!(config.xerror, fliplocs)
				else
					prob = @fastmath exp(-beta * ΔE)
					if rand(rng) < prob
						flip!(config.xerror, fliplocs)
					end
				end
			end

			for fliplocs in ztry
				ΔE = zero(T)
				@inbounds for i in fliplocs
					if config.xerror[i].x && config.zerror[i].x # y -> x
						ΔE -= sap.logp_x2y[i]
					elseif config.xerror[i].x && !config.zerror[i].x # x -> y
						ΔE += sap.logp_x2y[i]
					elseif !config.xerror[i].x && config.zerror[i].x # z -> i
						ΔE -= sap.logp_i2z[i]
					else # i -> z
						ΔE += sap.logp_i2z[i]
					end
				end

                if ΔE <= 0
					flip!(config.zerror, fliplocs)
				else
					prob = @fastmath exp(-beta * ΔE)
					if rand(rng) < prob
						flip!(config.zerror, fliplocs)
					end
				end
			end
		end
		logical_count[[(sum(config.xerror[sap.zlogical_qubits[i]]).x ? 2 : 1) + (sum(config.zerror[sap.xlogical_qubits[i]]).x ? 2 : 0) for i in 1:logical_num]...] += 1
		# for i in 1:logical_num
		# 	sum(config.xerror[sap.zlogical_qubits[i]]).x && (logicalx_count[i] += 1)
		# 	sum(config.zerror[sap.xlogical_qubits[i]]).x && (logicalz_count[i] += 1)
		# end
		# @show id, logical_count
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
struct MCMC{T} <: AbstractDecoder
	betas::Vector{T}
	etas::Vector{T}
	num_update_each_temp::Int
	ptemp::Float64
end

struct CompiledMCMC{T} <: CompiledDecoder
	sap::SpinGlassSA{T}
	decoder::MCMC{T}
end

# function compile(decoder::MCMC, problem::SimpleDecodingProblem)

#     return CompiledMCMC(SpinGlassSA(problem.tanner.s2q, problem.tanner.logical_qubits, problem.pvec, problem.tanner.logical_qubits_check), decoder)
# end



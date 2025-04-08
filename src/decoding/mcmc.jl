struct SpinGlassSA{T}
	s2q::Vector{Vector{Int}}
	logical_qubits::Vector{Int}
	logp_vector_error::Vector{T}
	logp_vector_noerror::Vector{T}
	logical_qubits_check::Vector{Int} # logicals to check
end

function SpinGlassSA(s2q::AbstractVector{Vector{Int}}, logical_qubits::AbstractVector{Int}, p_vector::AbstractVector{T}, logical_qubits2::AbstractVector{Int}) where T
	@assert all(p -> 0 <= p <= 1, p_vector) "`p_vector` should be in [0,1], got: $p_vector"
	# TODO: boundary check
	return SpinGlassSA(s2q, logical_qubits, log.(p_vector), log.(one(T) .- p_vector), logical_qubits2)
end

struct SpinConfig
	config::Vector{Mod2}
end

function beta_update(::Type{T}, ibeta, nbeta, rng) where T
	if ibeta == 1
		new_ibeta = 2
	elseif ibeta == nbeta
		new_ibeta = nbeta - 1
	else
		new_ibeta = rand(rng) > 0.5 ? ibeta - 1 : ibeta + 1 # propose a new beta
	end
	poldtonew = ibeta == 1 || ibeta == nbeta ? T(1.0) : T(0.5)
	pnewtoold = new_ibeta == 1 || new_ibeta == nbeta ? T(1.0) : T(0.5)
	return new_ibeta, poldtonew / pnewtoold
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
function anneal_singlerun!(config, sap::SpinGlassSA{T}, betas::Vector{T}; rng = Random.Xoshiro(), num_trials) where T
	zero_config, one_config = get01configs(config, sap.logical_qubits, sap.logical_qubits_check)
	one_count = 0
	for _ in 1:num_trials
		for beta in betas
			for ispin in 1:length(sap.s2q)+1
				ΔE = zero(T)
				fliplocs = ispin == length(sap.s2q) + 1 ? sap.logical_qubits : sap.s2q[ispin]
				@inbounds for i in fliplocs
					ΔE += config.config[i].x ? sap.logp_vector_error[i] - sap.logp_vector_noerror[i] : sap.logp_vector_noerror[i] - sap.logp_vector_error[i]
				end
                prob = ΔE <= 0 ? 1.0 : @fastmath exp(-beta * ΔE)
                if rand(rng) < prob  #accept
                    flip!(config, ispin, sap)
                end
			end
		end
		sum(i -> config.config[i], sap.logical_qubits_check).x && (one_count += 1)
	end
	return (; mostlikely = one_count / num_trials > 0.5 ? one_config : zero_config, p1 = one_count / num_trials)
end

function get01configs(config, logical_qubits::Vector{Int}, logical_qubits2::Vector{Int})
	zero_config = deepcopy(config)
	one_config = deepcopy(config)
	one_config.config[logical_qubits] .= one_config.config[logical_qubits] .+ Mod2(1)

	if sum(config.config[logical_qubits2]).x
		zero_config, one_config = one_config, zero_config
	end
	return zero_config, one_config
end

"""
	energy(config::AnnealingConfig, ap::AnnealingProblem) -> Real

Get the cost of specific configuration.
"""
function energy(config::SpinConfig, sap::SpinGlassSA{T}) where T
	return -sum(i -> config.config[i].x ? sap.logp_vector_error[i] : sap.logp_vector_noerror[i], 1:(length(config.config)))
end

"""
	propose(rng::Random.AbstractRNG, config::AnnealingConfig, ap::AnnealingProblem) -> (Proposal, Real)

Propose a change, as well as the energy change.
"""
@inline function propose(rng::Random.AbstractRNG, config::SpinConfig, sap::SpinGlassSA{T}) where T  # ommit the name of argument, since not used.
	max_ispin = length(sap.s2q) + 1
	ispin = rand(rng, 1:max_ispin)
	ΔE = zero(T)
	# TODO: allow tuning the logical qubits flip probability
	fliplocs = ispin == length(sap.s2q) + 1 ? sap.logical_qubits : sap.s2q[ispin]
	@inbounds for i in fliplocs
		ΔE += config.config[i].x ? sap.logp_vector_error[i] - sap.logp_vector_noerror[i] : sap.logp_vector_noerror[i] - sap.logp_vector_error[i]
	end
	return ispin, ΔE
end

"""
	flip!(config::AnnealingConfig, ispin::Proposal, ap::AnnealingProblem) -> SpinConfig

Apply the change to the configuration.
"""
function flip!(config::SpinConfig, ispin::Int, sap::SpinGlassSA)
	fliplocs = ispin == length(sap.s2q) + 1 ? sap.logical_qubits : sap.s2q[ispin]
	for q in fliplocs
		@inbounds config.config[q] = config.config[q] .+ Mod2(1)
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



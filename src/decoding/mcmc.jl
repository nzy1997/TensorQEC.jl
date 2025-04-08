struct SpinGlassSA{T}
    s2q::Vector{Vector{Int}}
    logp_vector_error::Vector{T}
    logp_vector_noerror::Vector{T}
end

function SpinGlassSA(s2q::AbstractVector{Vector{Int}}, p_vector::AbstractVector{T}) where T
    @assert all(p-> 0 <= p <= 1, p_vector) "`p_vector` should be in [0,1], got: $p_vector"
    # TODO: boundary check
    return SpinGlassSA(s2q, log.(p_vector), log.(one(T) .- p_vector))
end

struct SpinConfig
    config::Vector{Mod2}
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
function anneal_singlerun!(config, sap::SpinGlassSA{T}; num_sweep::Int=100000, rng=Random.Xoshiro()) where T
    n = length(config.config)
    # thermalizing stage, update eta
    cost = energy(config, sap)

    optimal_cost = cost
    optimal_config = deepcopy(config)
    mean_p = 0.0
    for i = 1:num_sweep * n  # single instruction multiple data, see julia performance tips.
        proposal, ΔE = propose(rng, config, sap)
        # TODO: implement a faster exp: https://deathandthepenguinblog.wordpress.com/2015/04/13/writing-a-faster-exp/
        prob = ΔE <= 0 ? 1.0 : @fastmath exp(-ΔE)
        if rand(rng) < prob  #accept
            flip!(config, proposal, sap)
            cost += ΔE
            if cost < optimal_cost
                optimal_cost = cost
                optimal_config = deepcopy(config)
            end
        end
        mean_p += exp(-cost)
    end
    return (; optimal_cost, optimal_config, mean_p = (mean_p/num_sweep/n))
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
    ispin = rand(rng, 1:length(sap.s2q))
    ΔE = zero(T)
    @inbounds for i in sap.s2q[ispin]
        ΔE += config.config[i].x ? sap.logp_vector_error[i] - sap.logp_vector_noerror[i] : sap.logp_vector_noerror[i] - sap.logp_vector_error[i]
    end
    return ispin, ΔE
end

"""
    flip!(config::AnnealingConfig, ispin::Proposal, ap::AnnealingProblem) -> SpinConfig

Apply the change to the configuration.
"""
function flip!(config::SpinConfig, ispin::Int, sap::SpinGlassSA)
    for q in sap.s2q[ispin]
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



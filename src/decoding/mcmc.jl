struct SpinGlassSA{T}
    s2q::Vector{Vector{Int}}
    syndrome::Vector{Mod2}
    logical_qubits::Vector{Int}
    logp_vector_error::Vector{T}
    logp_vector_noerror::Vector{T}
    logical_qubits2::Vector{Int} # logicals to check
end

function SpinGlassSA(s2q::AbstractVector{Vector{Int}}, syndrome::AbstractVector{Mod2}, logical_qubits::AbstractVector{Int}, p_vector::AbstractVector{T}, logical_qubits2::AbstractVector{Int}) where T
    @assert all(p-> 0 <= p <= 1, p_vector) "`p_vector` should be in [0,1], got: $p_vector"
    # TODO: boundary check
    return SpinGlassSA(s2q, syndrome, logical_qubits, log.(p_vector), log.(one(T) .- p_vector), logical_qubits2)
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
function anneal_singlerun!(config, sap::SpinGlassSA{T}, betas::Vector{T}; num_sweep::Int=100000, num_sweep_thermalize::Int=100, ptemp=T(0.1), rng=Random.Xoshiro(), eta_lr=T(0.2)) where T
    @assert isone(betas[1]) "betas[1] must be 1.0, since this is the temperature to sample the logical qubits!"
    n = length(config.config)
    zero_config, one_config = get01configs(config, sap.logical_qubits, sap.logical_qubits2)
    # thermalizing stage, update eta
    ibeta = 1; beta = betas[ibeta]
    etas = fill(T(0), length(betas))
    cost = energy(config, sap)
    for _ = 1:num_sweep_thermalize * n
        if length(betas) > 1 && rand(rng) < 0.5
            new_ibeta = rand(rng, 1:length(betas)-1); new_ibeta >= ibeta && (new_ibeta += 1)  # propose a new beta
            lbeta, hbeta = minmax(ibeta, new_ibeta)
            etas[hbeta] = (1-eta_lr) * etas[hbeta] + eta_lr * (betas[hbeta] - betas[lbeta])*cost  # update low beta
            prob = exp(-(betas[new_ibeta] - beta)*cost + (etas[new_ibeta] - etas[ibeta]))
            if rand(rng) < prob
                beta = betas[new_ibeta]
                ibeta = new_ibeta
            end
        end
        proposal, ΔE = propose(rng, config, sap)
        if rand(rng) < exp(-beta*ΔE)
            flip!(config, proposal, sap)
            cost += ΔE
        end
    end

    propose_count = accept_count = 0  # for all temperature
    valid_count = one_count = 0       # for beta = 1.0
    beta_propose_count = beta_update_count = 0  # for beta-swap
    optimal_cost = cost
    optimal_config = deepcopy(config)
    for _ = 1:num_sweep * n  # single instruction multiple data, see julia performance tips.
        if length(betas) > 1 && rand(rng) < ptemp
            beta_propose_count += 1
            new_ibeta = rand(rng, 1:length(betas)-1); new_ibeta >= ibeta && (new_ibeta += 1)  # propose a new beta
            prob = exp(-(betas[new_ibeta] - beta)*cost + etas[new_ibeta] - etas[ibeta])
            if rand(rng) < prob
                beta = betas[new_ibeta]
                ibeta = new_ibeta
                beta_update_count += 1
            end
        else
            propose_count += 1
            proposal, ΔE = propose(rng, config, sap)
            # TODO: implement a faster exp: https://deathandthepenguinblog.wordpress.com/2015/04/13/writing-a-faster-exp/
            prob = ΔE <= 0 ? 1.0 : @fastmath exp(-beta*ΔE)
            if rand(rng) < prob  #accept
                flip!(config, proposal, sap)
                cost += ΔE
                if cost < optimal_cost
                    optimal_cost = cost
                    optimal_config = deepcopy(config)
                end
                accept_count += 1
            end
            if isone(beta)
                valid_count += 1
                sum(i->config.config[i], sap.logical_qubits2).x && (one_count += 1)
            end
        end
    end
    
    return (; optimal_cost,
            optimal_config,
            p1 = one_count/valid_count,
            mostlikely = one_count/valid_count * 2 > 1 ? one_config : zero_config,
            accept_rate = accept_count/propose_count,
            beta_accpet_rate = beta_update_count/beta_propose_count,
            valid_samples = propose_count,
            etas = etas
        )
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
    fliplocs = ispin == length(sap.s2q)+1 ? sap.logical_qubits : sap.s2q[ispin]
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
    fliplocs = ispin == length(sap.s2q)+1 ? sap.logical_qubits : sap.s2q[ispin]
    for q in fliplocs
        @inbounds config.config[q] = config.config[q] .+ Mod2(1)
    end
    return config
end

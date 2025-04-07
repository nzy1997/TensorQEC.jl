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
    anneal_singlerun!(config::AnnealingConfig, prob, betas::Vector{Float64}, num_update_each_temp::Int)

Perform Simulated Annealing using Metropolis updates for the single run.

    * `config`: configuration that can be updated.
    * `sap`: problem with `energy`, `flip!` and `random_config` interfaces.
    * `betas`: inverse temperature scales.
    * `num_update_each_temp`: the number of update in each temprature scale.

Returns (minimum cost, optimal configuration).
"""
function anneal_singlerun!(config, sap::SpinGlassSA{T}, betas::Vector{T}, etas::Vector{T}, num_update_each_temp::Int; ptemp=0.1, rng=Random.Xoshiro()) where T
    total_count = logical_count = 0

    cost = energy(config, sap)
    optimal_cost = cost
    optimal_config = deepcopy(config)

    zero_config = deepcopy(config)
    one_config = deepcopy(config)
    one_config.config[sap.logical_qubits] .= one_config.config[sap.logical_qubits] .+ Mod2(1)

    if sum(config.config[sap.logical_qubits2]).x
        zero_config, one_config = one_config, zero_config
    end

    ibeta = 1; beta = betas[ibeta]
    accept_count = 0
    for i = 1:num_update_each_temp  # single instruction multiple data, see julia performance tips.
        if length(betas) > 1 && rand(rng) < ptemp
            new_ibeta = rand(rng, 1:length(betas)-1); new_ibeta >= ibeta && (new_ibeta += 1)  # propose a new beta
            prob = exp(-(betas[new_ibeta] - beta)*cost + etas[new_ibeta] - etas[ibeta])
            if rand(rng) < prob
                beta = betas[new_ibeta]
                ibeta = new_ibeta
                accept_count += 1
            end
        else
            proposal, ΔE = propose(rng, config, sap)
            # TODO: implement a faster exp: https://deathandthepenguinblog.wordpress.com/2015/04/13/writing-a-faster-exp/
            prob = ΔE <= 0 ? 1 : @fastmath exp(-beta*ΔE)
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
                total_count += 1
                sum(i->config.config[i], sap.logical_qubits2).x && (logical_count += 1)
            end
        end
    end
    return (; optimal_cost,
            optimal_config,
            p1 = logical_count/total_count,
            mostlikely = logical_count/total_count * 2 > 1 ? one_config : zero_config,
            accept_rate = accept_count/num_update_each_temp
        )
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

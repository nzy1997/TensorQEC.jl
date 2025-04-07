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
    anneal_singlerun!(config::AnnealingConfig, prob, tempscales::Vector{Float64}, num_update_each_temp::Int)

Perform Simulated Annealing using Metropolis updates for the single run.

    * `config`: configuration that can be updated.
    * `sap`: problem with `energy`, `flip!` and `random_config` interfaces.
    * `tempscales`: temperature scales, which should be a decreasing array.
    * `num_update_each_temp`: the number of update in each temprature scale.

Returns (minimum cost, optimal configuration).
"""
function anneal_singlerun!(config, sap::SpinGlassSA{T}, tempscales::Vector{T}, num_update_each_temp::Int, rng=Random.Xoshiro()) where T
    logical_count = 0

    cost = energy(config, sap)
    optimal_cost = cost
    optimal_config = deepcopy(config)

    zero_config = deepcopy(config)
    one_config = deepcopy(config)
    one_config.config[sap.logical_qubits] .= one_config.config[sap.logical_qubits] .+ Mod2(1)

    if sum(config.config[sap.logical_qubits2]).x
        zero_config, one_config = one_config, zero_config
    end

    # xsqure = 0
    # xmean = 0
    for beta = 1 ./ tempscales
        for i = 1:num_update_each_temp  # single instruction multiple data, see julia performance tips.
            proposal, ΔE = propose(rng, config, sap)
            # TODO: implement a faster exp: https://deathandthepenguinblog.wordpress.com/2015/04/13/writing-a-faster-exp/
            prob = ΔE <= 0 ? 1 : @fastmath exp(-beta*ΔE)
            if prob > rand(rng)  #accept
                flip!(config, proposal, sap)
                cost += ΔE
                if cost < optimal_cost
                    optimal_cost = cost
                    optimal_config = deepcopy(config)
                end
            end
            sum(i->config.config[i], sap.logical_qubits2).x && (logical_count += 1)
            # xmean = logical_count/i
            # i > 10000 && (xmean - xmean^2)/sqrt(i) < 1e-3 * abs(0.5 - xmean) && (return xmean > 0.5 ? one_config : zero_config)
            #@show logical_count/i, (xmean - xmean^2)/sqrt(i)
            # if i > 10000
            #     num = i - 10000
            #     xmean = logical_count/num
            #     # @info (logical_count/num - (logical_count/(num))^2)/sqrt(num)
            #     (logical_count/num - (logical_count/(num))^2)/sqrt(num) < 0.25 * abs(0.5 - logical_count/num) && (return logical_count/num > 0.5 ? one_config : zero_config)
            # end
            # (i>10000 && sum(config.config[prob.logical_qubits2]).x) && (logical_count += 1)
        end
    end
    return (; optimal_cost,
            optimal_config,
            p1 = logical_count/num_update_each_temp,
            mostlikely = logical_count/num_update_each_temp * 2 > 1 ? one_config : zero_config
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

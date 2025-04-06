struct SpinGlassSA
    s2q::Vector{Vector{Int}}
    syndrome::Vector{Mod2}
    logical_qubits::Vector{Int}
    p_vector::Vector{Float64}
    logical_qubits2::Vector{Int} # logicals to check
end

struct SpinConfig
    config::Vector{Mod2}
end

"""
    anneal_singlerun!(config::AnnealingConfig, prob, tempscales::Vector{Float64}, num_update_each_temp::Int)

Perform Simulated Annealing using Metropolis updates for the single run.

    * configuration that can be updated.
    * prob: problem with `energy`, `flip!` and `random_config` interfaces.
    * tempscales: temperature scales, which should be a decreasing array.
    * num_update_each_temp: the number of update in each temprature scale.

Returns (minimum cost, optimal configuration).
"""
function anneal_singlerun!(config, prob, tempscales::Vector{Float64}, num_update_each_temp::Int)
    logical_count = 0

    cost = energy(config, prob)
    optimal_cost = cost
    optimal_config = deepcopy(config)

    zero_config = deepcopy(config)
    one_config = deepcopy(config)
    one_config.config[prob.logical_qubits] .= one_config.config[prob.logical_qubits] .+ Mod2(1)

    if sum(config.config[prob.logical_qubits2]).x
        zero_config, one_config = one_config, zero_config
    end

    for beta = 1 ./ tempscales
        for i = 1:num_update_each_temp  # single instruction multiple data, see julia performance tips.
            proposal, ΔE = propose(config, prob)
            if exp(-beta*ΔE) > rand()  #accept
                flip!(config, proposal, prob)
                cost += ΔE
                if cost < optimal_cost
                    optimal_cost = cost
                    optimal_config = deepcopy(config)
                end
            end
            sum(config.config[prob.logical_qubits2]).x && (logical_count += 1)
            @show logical_count/(i)
            if i > 10000
                num = i - 10000

                @info (logical_count/num - (logical_count/(num))^2)/sqrt(num)
                (logical_count/num - (logical_count/(num))^2)/sqrt(num) < abs(0.5 - logical_count/num) && (return logical_count/num > 0.5 ? one_config : zero_config)
            end

            # (i>10000 && sum(config.config[prob.logical_qubits2]).x) && (logical_count += 1)
        end
    end
    @show optimal_cost optimal_config
    logical_count
end
 
"""
    energy(config::AnnealingConfig, ap::AnnealingProblem) -> Real

Get the cost of specific configuration.
"""
function energy(config::SpinConfig, sap::SpinGlassSA)
    return -sum(i -> config.config[i].x ? log(sap.p_vector[i]) : log(1.0 - sap.p_vector[i]), 1:(length(config.config)))
end

"""
    propose(config::AnnealingConfig, ap::AnnealingProblem) -> (Proposal, Real)

Propose a change, as well as the energy change.
"""
@inline function propose(config::SpinConfig, sap::SpinGlassSA)  # ommit the name of argument, since not used.
    max_ispin = length(sap.s2q) + 1
    ispin = rand(1:max_ispin)
    if ispin == length(sap.s2q)+1
        ΔE = sum(i -> config.config[i].x ? log(sap.p_vector[i]/((1.0 - sap.p_vector[i]))) : log((1.0 - sap.p_vector[i])/(sap.p_vector[i])), sap.logical_qubits)
        return ispin, ΔE
    else
        ΔE = sum(i -> config.config[i].x ? log(sap.p_vector[i]/((1.0 - sap.p_vector[i]))) : log((1.0 - sap.p_vector[i])/(sap.p_vector[i])), sap.s2q[ispin])
        return ispin, ΔE
    end
end

"""
    flip!(config::AnnealingConfig, ispin::Proposal, ap::AnnealingProblem) -> SpinConfig

Apply the change to the configuration.
"""
function flip!(config::SpinConfig, ispin::Int, sap::SpinGlassSA)
    if ispin == length(sap.s2q)+1
        config.config[sap.logical_qubits] .= config.config[sap.logical_qubits] .+ Mod2(1)
    else
        config.config[sap.s2q[ispin]] .= config.config[sap.s2q[ispin]] .+ Mod2(1)
    end
    config
end

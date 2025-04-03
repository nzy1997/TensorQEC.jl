"""
    RBM(num_visible::Int, num_hidden::Int)

A Restricted Boltzmann Machine with the specified number of visible and hidden nodes.
"""
struct RBM
    W::Matrix{Float64}  # weights
    v_bias::Vector{Float64}  # bias for visible layer
    h_bias::Vector{Float64}  # bias for hidden layer
    num_visible::Int
    num_hidden::Int
end

"""
    RBM(num_visible::Int, num_hidden::Int)

Create a new RBM with random weights and biases.
"""
function RBM(num_visible::Int, num_hidden::Int)
    W = randn(num_hidden, num_visible) * 1e-2
    v_bias = zeros(num_visible)
    h_bias = randn(num_hidden) * 1e-2
    return RBM(W, v_bias, h_bias, num_visible, num_hidden)
end

function sigmoid(x::Float64)
    return 1.0 / (1.0 + exp(-x))
end

"""
    v_to_h(rbm::RBM, v::Vector{Float64})

Forward pass p(h|v) from visible to hidden layer.
"""
function v_to_h(rbm::RBM, v::Vector{Float64})
    return sigmoid.(rbm.W * v + rbm.h_bias)
end

"""
    h_to_v(rbm::RBM, h::Vector{Float64})

Backward pass p(v|h) from hidden to visible layer.
"""
function h_to_v(rbm::RBM, h::Vector{Float64})
    return sigmoid.(rbm.W' * h + rbm.v_bias)
end

"""
    sample_from_prob(prob_list::Vector{Float64})

Sample binary values from probabilities.
"""
function sample_from_prob(prob_list::Vector{Float64})
    return Float64.(rand(length(prob_list)) .< prob_list)
end

"""
    contrastive_divergence(rbm::RBM, v::Vector{Float64}, k::Int)

Perform k-step contrastive divergence sampling.
"""
function contrastive_divergence(rbm::RBM, v::Vector{Float64}, k::Int)
    prob_h = v_to_h(rbm, v)
    h = sample_from_prob(prob_h)
    
    for _ in 1:k
        prob_v = h_to_v(rbm, h)
        v = sample_from_prob(prob_v)
        prob_h = v_to_h(rbm, v)
        h = sample_from_prob(prob_h)
    end
    
    return v
end

"""
    free_energy(rbm::RBM, v::Vector{Float64})

Calculate the free energy of the RBM for a given visible state.
"""
function free_energy(rbm::RBM, v::Vector{Float64})
    vbias_term = v' * rbm.v_bias
    wx_b = rbm.W * v + rbm.h_bias
    hidden_term = sum(log.(1 .+ exp.(wx_b)))
    return -hidden_term - vbias_term
end
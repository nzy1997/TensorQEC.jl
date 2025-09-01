struct ErrorModel
    num_bits::Int
    pair::Dict{Vector{Int},Array{Float64}}
end

abstract type AbstractSampler end
struct IndependentSampler <: AbstractSampler end

function random_error_pattern(em::ErrorModel, num_samples::Int, sampler::IndependentSampler) end

# decoding problem
struct DecodingProblem
    error_model::ErrorModel
    check_matrix::Matrix{Bool}
    logical_matrix::Matrix{Bool}
end

function simple_decoding_problem(c::code,error_rate::Float64) end

function syndrome_extraction(check_matrix::Matrix{Bool}, error_patterns::Matrix{Bool}) end

abstract type AbstractDecoder end

function decode(problem,syndrome,decoder::AbstractDecoder) end

function check_logical_error(error_pattern::Matrix{Bool}, logical_matrix::Matrix{Bool}) end
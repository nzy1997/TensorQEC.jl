# Error Model
abstract type AbstractErrorModel end

function random_error_pattern(em::AbstractErrorModel, num_samples::Int) end

struct IndependentFlipError<: AbstractErrorModel
    p::Vector{Float64}
end

struct TensorNetworkError<: AbstractErrorModel
    ixs::Vector{Vector{Int}}
    ps::Vector{AbstractArray{Float64}}
    num_bits::Int
end

# decoding problem
struct DecodingProblem{ET<:AbstractErrorModel}
    error_model::ET
    check_matrix::Matrix{Bool}
    logical_matrix::Matrix{Bool}
end
# ?? bool or mod2

function code_capacity(c::code) end
function circuit_level(qc::ChainBlock) end

function syndrome_extraction(check_matrix::Matrix{Bool}, syndrome_sample) end

# decoder
abstract type AbstractDecoder end
abstract type AbstractCompliedDecoder end
function compile(decoder::AbstractDecoder, problem::DecodingProblem) end
function decode(compiled_decoder::AbstractCompliedDecoder, syndrome_sample) end

# QuantumClifford.jl
struct TableDecoder <: AbstractSyndromeDecoder
    """Stabilizer tableau defining the code"""
    H
    """Faults matrix corresponding to the code"""
    faults_matrix
    """The number of qubits in the code"""
    n::Int
    """The depth of the code"""
    s::Int
    """The number of encoded qubits"""
    k::Int
    """The lookup table corresponding to the code, slow to create"""
    lookup_table::Dict{Vector{Bool},Vector{Bool}}
    lookup_buffer::Vector{Bool}
    TableDecoder(H, faults_matrix, n, s, k, lookup_table) = new(H, faults_matrix, n, s, k, lookup_table, fill(false, s))
end

function TableDecoder(c::Code)
    H = parity_checks(c)
    s, n = size(H)
    _, _, r = canonicalize!(Base.copy(H), ranks=true)
    k = n - r
    lookup_table = create_lookup_table(H)
    fm = faults_matrix(H)
    return TableDecoder(H, fm, n, s, k, lookup_table)
end

function decode(d::TableDecoder, syndrome_sample) end

# batch decoding
function evaluate_decoder(d::AbstractSyndromeDecoder, setup::AbstractECCSetup, nsamples::Int) end
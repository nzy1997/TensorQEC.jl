# Error Model
struct ErrorModel
    # ixs::Vector{Vector{Int}}
    # ps::Vector{AbstractArray{Float64}}
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


function simple_decoding_problem(c::code,error_rate::Float64) end # QECCore

function circuit_level() end # TensorQEC
# abstract type AbstractDecodingSetup end

# struct DepolarizingSetup <: AbstractDecodingSetup
#     p::Float64
# end

# struct CircuitLevelSetup <: AbstractDecodingSetup
#     two_qubit_error_rate_p::Float64
#     measure_p::Float64
# end

# function decoding_problem(c::code,setup::DepolarizingSetup) end #QECCore
# function circuit_level(qc::ChainBlock) end

function syndrome_extraction(check_matrix::Matrix{Bool}, error_patterns::Matrix{Bool}) end

# decoder
abstract type AbstractDecoder end
# abstract type AbstractCompliedDecoder end
# function compile(decoder::AbstractDecoder, problem::DecodingProblem) end
function decode(decoder::TableDecoder, syndrome_sample::Matrix{Bool}) end

# QuantumClifford.jl
struct TableDecoder <: AbstractDecoder
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

function decode(d::TableDecoder, syndrome_sample::Matrix{Bool}) end
# return matrix{Bool}

# batch decoding
# function logical_error_rate(d::AbstractSyndromeDecoder, setup::AbstractECCSetup, nsamples::Int) 


# end

decoder = TableDecoder(code,)

function decode(problem,syndrome,TablerDecoder())

end
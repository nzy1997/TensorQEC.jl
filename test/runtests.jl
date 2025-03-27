using TensorQEC
using Test

@testset "mod2" begin
    include("codes/mod2.jl")
end

@testset "paulistring" begin
    include("paulibasis/paulistring.jl")
end

@testset "clifford" begin
    include("paulibasis/cliffordgroup.jl")
end

@testset "pauli basis" begin
    include("paulibasis/paulibasis.jl")
end

@testset "tensor network" begin
    include("tensornetwork/tensornetwork.jl")
end

@testset "inferences" begin
    include("tensornetwork/inferences.jl")
end

@testset "codes" begin
    include("codes/codes.jl")
end

@testset "encoder" begin
    include("codes/encoder.jl")
end

@testset "measurement" begin
    include("codes/measurement.jl")
end

@testset "table make" begin
    include("codes/tablemake.jl")
end

@testset "simulation" begin
    include("tensornetwork/simulation.jl") 
end

@testset "ldpc" begin
    include("codes/ldpc.jl")
end

@testset "tableau" begin
    include("paulibasis/tableaux.jl")
end

@testset "error_model" begin
    include("decoding/error_model.jl")
end

@testset "code_distance" begin
    include("codes/code_distance.jl")
end

# @testset "threshold" begin
#     include("decoding/threshold.jl")
# end

@testset "decoder" begin
    include("decoding/decoder.jl")
end

@testset "multiprocessing" begin
    include("multiprocessing.jl")
end

@testset "error_learning" begin
    include("decoding/error_learning.jl")
end

@testset "matching" begin
    include("decoding/matching.jl")
end

@testset "ipdecoder" begin
    include("decoding/ipdecoder.jl")
end

@testset "bpdecoder" begin
    include("decoding/bpdecoder.jl")
end

@testset "decoding interfaces" begin
    include("decoding/interfaces.jl")
end

@testset "tndecoder" begin
    include("decoding/tndecoder.jl")
end
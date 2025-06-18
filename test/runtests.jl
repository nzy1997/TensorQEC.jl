using TensorQEC
using Test
using Documenter

@testset "mod2" begin
    include("codes/mod2.jl")
end

@testset "paulistring" begin
    include("clifford/paulistring.jl")
end

@testset "clifford" begin
    include("clifford/cliffordgroup.jl")
end

@testset "pauli basis" begin
    include("clifford/paulibasis.jl")
end

@testset "tensor network" begin
    include("nonclifford/tensornetwork.jl")
end

@testset "inferences" begin
    include("decoding/inferenceswithencoder.jl")
end

@testset "codes" begin
    include("codes/codes.jl")
end

@testset "encoder" begin
    include("codes/encoder.jl")
end

@testset "measurement" begin
    include("decoding/measurement.jl")
end

@testset "truthtable" begin
    include("decoding/truthtable.jl")
end

@testset "simulation" begin
    include("nonclifford/simulation.jl") 
end

@testset "ldpc" begin
    include("codes/ldpc.jl")
end

@testset "tableau" begin
    include("clifford/tableaux.jl")
end

@testset "error_model" begin
    include("decoding/error_model.jl")
end

@testset "code_distance" begin
    include("codes/code_distance.jl")
end

@testset "threshold" begin
    include("decoding/threshold.jl")
end

@testset "decoder" begin
    include("decoding/general_decoding.jl")
end

@testset "multiprocessing" begin
    include("multiprocessing.jl")
end

@testset "error_learning" begin
    include("nonclifford/error_learning.jl")
end

@testset "matching" begin
    include("decoding/matching.jl")
end

@testset "ipdecoder" begin
    include("decoding/ipdecoder.jl")
end

@testset "bpdecoder" begin
    include("decoding/bposd.jl")
end

@testset "decoding interfaces" begin
    include("decoding/interfaces.jl")
end

@testset "tndecoder" begin
    include("decoding/tndecoder.jl")
end

@testset "correction" begin
    include("nonclifford/correction.jl")
end

@testset "deprecate" begin
    include("deprecate.jl")
end

DocMeta.setdocmeta!(TensorQEC, :DocTestSetup, :(using TensorQEC); recursive=true)
Documenter.doctest(TensorQEC; manual=false, fix=false)
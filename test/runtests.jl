using TensorQEC
using Test

@testset "paulistring" begin
    include("paulistring.jl")
end

@testset "pauli basis" begin
    include("paulibasis.jl")
end

@testset "tensor network" begin
    include("tensornetwork.jl")
end

@testset "inferences" begin
    include("inferences.jl")
end

@testset "encoder" begin
    include("encoder.jl")
end
using TensorQEC
using Test

@testset "pauli basis" begin
    include("paulibasis.jl")
end

@testset "tensor network" begin
    include("tensornetwork.jl")
end
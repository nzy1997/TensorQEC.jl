using TensorQEC
using Test

@testset "mod2" begin
    include("mod2.jl")
end

@testset "paulistring" begin
    include("paulistring.jl")
end

@testset "clifford" begin
    include("cliffordgroup.jl")
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

@testset "measurement" begin
    include("measurement.jl")
end

@testset "tablemake" begin
    include("tablemake.jl")
end

@testset "qc2ein.jl" begin
    include("qc2ein.jl") 
 end
 
 @testset "coerror.jl" begin
    include("coerror.jl")
 end
using Test
using TensorQEC
using Random

@testset "FlipError" begin
    em = FlipError(0.1)
    @test random_error_qubits(10, em) isa Vector{Mod2}
end

@testset "DepolarizingError" begin
    em = DepolarizingError(0.05, 0.06, 0.1)
    ex,ez = random_error_qubits(100000, em)
    @test ex isa Vector{Mod2}
    @test ez isa Vector{Mod2}
    @test count(v->v.x,ex)/100000 ≈ 0.11 atol=0.05
    @test count(v->v.x,ez)/100000 ≈ 0.16 atol=0.05
end


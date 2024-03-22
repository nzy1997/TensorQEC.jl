using TensorQEC, Test, TensorQEC.Yao
using TensorQEC: pauli_repr

@testset "perm repr" begin
    m = pauli_repr(H)
    pm = TensorQEC.to_perm_matrix(Int8, Int, m)
    @test Matrix(pm) ≈ m

    m = pauli_repr(ConstGate.CNOT)
    pm = TensorQEC.to_perm_matrix(Int8, Int, m)
    @test Matrix(pm) ≈ m
end

@testset "generate group" begin
    @test TensorQEC.generate_group([1im]) |> length == 4
end

@testset "pauli group" begin
    @test size(pauli_group(1)) == (4, 4)
    @test all(getfield.(pauli_group(1)[1,:], :first) .== 0)
    @test all(getfield.(pauli_group(1)[2,:], :first) .== 1)
    @test size(pauli_group(2)) == (4, 4, 4)
end

@testset "clifford group" begin
    csize(n::Int) = prod(k->2 * 4^k * (4^k - 1), 1:n)
    @test length(clifford_group(1)) == csize(1)
    @test length(clifford_group(2)) == csize(2)
end

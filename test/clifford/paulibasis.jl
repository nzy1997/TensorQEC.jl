using Test, TensorQEC, TensorQEC.Yao, TensorQEC.LinearAlgebra

@testset "pauli_c2l and pauli_l2c" begin
    N = 3
	@test TensorQEC.pauli_l2c(Val(N), TensorQEC.pauli_c2l(Val(N), [1, 2, 3])) == (1, 2, 3)
    @test TensorQEC.pauli_c2l(Val(N), TensorQEC.pauli_l2c(Val(N), 12)) == 12
end

@testset "pauli_basis" begin
	i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
	@test pauli_basis(1) == PauliString.([i, x, y, z])
end

@testset "pauli_decomposition" begin
	@test pauli_decomposition(Matrix(H)) == [0, 1, 0, 1] / sqrt(2)
	@test pauli_decomposition(Matrix{Float64}(H)) == [0, 1, 0, 1] / sqrt(2)
	@test pauli_decomposition(Matrix(kron(X, X))) == [0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0]
end

@testset "density matrix" begin
	reg = rand_state(6)
	dm = density_matrix(reg, 1:3)
	sp = SumOfPaulis(dm)
	@test mat(yaoblock(sp)) ≈ dm.state
end

@testset "arrayreg to sumofpaulis" begin 
	reg = rand_state(3)
	dm = density_matrix(reg)
	sp1 = SumOfPaulis(reg)
	@test mat(yaoblock(sp1)) ≈ dm.state

	reg = ghz_state(3)
	dm = density_matrix(reg)
	sp2 = SumOfPaulis(reg)
	@test mat(yaoblock(sp2)) ≈ dm.state
end

@testset "map_pauli_string" begin
	i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
	ps = PauliString(x, y, z, y, x, i)
	@test TensorQEC.map_pauli_string(ps, pauli_repr(mat(ComplexF64,cnot(2,1,2))), [5,6]) == PauliString(x, y, z, y, x, x)
end

@testset "pauli_string_map_iter" begin
	i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
	ps = PauliString(i, y, z, x, i)
	qc = chain(cnot(5,4,5), put(5, 3=>H))
	@test TensorQEC.pauli_string_map_iter(ps, qc) == PauliString(i, y, x, x, x)

	ps = PauliString(i, x, x)
	qc = chain(cnot(3,3,2), cnot(3,3,1))
	@test TensorQEC.pauli_string_map_iter(ps, qc) == PauliString(x, i, x)
end

@testset "yao interfaces" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    @test PauliString(5, (2, 3)=>x) == PauliString(i, x, x, i, i)
    @test PauliString(5, (2, 3)=>x, 4=>y) == PauliString(i, x, x, y, i)
    @test pauli_decomposition(X) ≈ [0, 1, 0, 0]
    @test pauli_repr(X) ≈ Diagonal([1, 1, -1, -1])
end
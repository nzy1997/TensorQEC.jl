using Test, TensorQEC, TensorQEC.Yao, TensorQEC.LinearAlgebra

@testset "pauli_basis" begin
	@test pauli_basis(1) == PauliString.(1:4)
end

@testset "pauli_decomposition" begin
	@test pauli_decomposition(Matrix(H)) == [0, 1, 0, 1] / sqrt(2)
	@test pauli_decomposition(Matrix(kron(X, X))) == [0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0]
end

@testset "density matrix" begin
	reg = rand_state(6)
	dm = density_matrix(reg, 1:3)
	sp = densitymatrix2sumofpaulis(dm)
	@test mat(sp) ≈ dm.state
end

@testset "arrayreg2sumofpaulis" begin 
	reg = rand_state(3)
	dm = density_matrix(reg)
	sp2 = arrayreg2sumofpaulis(reg)
	@test mat(sp2) ≈ dm.state

	reg = ghz_state(4)
	dm = density_matrix(reg)
	sp2 = arrayreg2sumofpaulis(reg)
	@test mat(sp2) ≈ dm.state
end

@testset "pauli_string_map" begin
	ps = PauliString(2,3,4,3,2,1)
	@test TensorQEC.pauli_string_map(ps, pauli_mapping(mat(ComplexF64,cnot(2,1,2))), [5,6]).ids ==(2,3,4,3,2,2)
end

@testset "pauli_string_map_iter" begin
	ps = PauliString(1,3,4,2,1)
	qc = chain(cnot(5,4,5), put(5, 3=>H))
	@test TensorQEC.pauli_string_map_iter(ps, qc).ids ==(1,3,2,2,2)

	ps = PauliString(1,2,2)
	qc = chain(cnot(3,3,2), cnot(3,3,1))
	@test TensorQEC.pauli_string_map_iter(ps, qc).ids == (2,1,2)
end

@testset "yao interfaces" begin
    @test paulistring(5, X, (2, 3)) == PauliString(I2, X, X, I2, I2)
    @test pauli_decomposition(X) ≈ [0, 1, 0, 0]
    @test pauli_mapping(X) ≈ Diagonal([1, 1, -1, -1])
end
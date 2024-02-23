using Test, TensorQEC, TensorQEC.Yao

@testset "pauli_basis" begin
	@test pauli_basis(1) == [Matrix(I2), Matrix(X), Matrix(Y), Matrix(Z)]
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
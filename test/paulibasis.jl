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

@testset "pauli string" begin
	reg = rand_state(4)
	g = PauliString((1, 2, 3, 4))
	@test nqubits(g) == 4
	r1 = apply(reg, g)
	r2 = apply(reg, kron(I2, X, Y, Z))
	@test r1 ≈ r2
	@test mat(g) ≈ mat(kron(I2, X, Y, Z))
end
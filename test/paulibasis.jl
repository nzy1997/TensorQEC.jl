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
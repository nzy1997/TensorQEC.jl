using Test, TensorQEC, TensorQEC.Yao, TensorQEC.LinearAlgebra, TensorQEC.Yao
using Random

@testset "bimatrix2stabilizers" begin
	t = ToricCode(3, 3)
	result = stabilizers(t)
	code = TensorQEC.stabilizers2bimatrix(result)
	st = TensorQEC.bimatrix2stabilizers(code)
	@test st == result
end

@testset "gaussian_elimination" begin
	t = ToricCode(3, 3)
	result = stabilizers(t)
	code = TensorQEC.stabilizers2bimatrix(result)
	code2 = TensorQEC.gaussian_elimination!(copy(code))
	@test code2.matrix[1:8, 1:8] == [
		1  0  0  0  0  0  0  0;
		0  1  0  0  0  0  0  0;
		0  0  1  0  0  0  0  0;
		0  0  0  1  0  0  0  0;
		0  0  0  0  1  0  0  0;
		0  0  0  0  0  1  0  0;
		0  0  0  0  0  0  1  0;
		0  0  0  0  0  0  0  1
	]
	@test code2.matrix[9:16, 27:34] == [
		1  0  0  0  0  0  0  0;
		0  1  0  0  0  0  0  0;
		0  0  1  0  0  0  0  0;
		0  0  0  1  0  0  0  0;
		0  0  0  0  1  0  0  0;
		0  0  0  0  0  1  0  0;
		0  0  0  0  0  0  1  0;
		0  0  0  0  0  0  0  1
	]
	@test sort(code.ordering) == collect(1:18)
	# using Q to check the gaussian elimination
	m1, m2 = code2.Q * Mod2.(code.matrix), Mod2.(code2.matrix)
	@test m1[:, vcat(code2.ordering, code2.ordering .+ nqubits(code2))] == m2
end

@testset "quantum chain block" begin
	t = ToricCode(2, 2)
	result = stabilizers(t)
	code = TensorQEC.stabilizers2bimatrix(result)
	TensorQEC.gaussian_elimination!(code)
	st = TensorQEC.bimatrix2stabilizers(code)
	qc = TensorQEC.encode_circuit(code)
	u = mat(ComplexF64, qc)
	for i in 1:size(code.matrix, 1)
		@test u * mat(ComplexF64, put(8, code.ordering[i] => Z)) * u' ≈ mat(ComplexF64, st[i])
	end
end

@testset "encode_stabilizers" begin
	result = stabilizers(SurfaceCode(3,3))
	qc, data_qubits, bimat = TensorQEC.encode_stabilizers(result)
	u = mat(ComplexF64, qc)
	st = TensorQEC.bimatrix2stabilizers(bimat)
	for i in 1:size(bimat.matrix, 1)
		@test u * mat(ComplexF64, put(9, bimat.ordering[i] => Z)) * u' ≈ mat(ComplexF64, st[i])
	end
end

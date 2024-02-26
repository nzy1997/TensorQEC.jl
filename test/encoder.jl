using Test, TensorQEC, TensorQEC.Yao, TensorQEC.LinearAlgebra, TensorQEC.Yao

@testset "toric code" begin
	t = TensorQEC.ToricCode(2, 3)
	result = TensorQEC.stabilizers(t)
	expected_result =
		PauliString.([
			(2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1),
			(1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1),
			(1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1),
			(1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1),
			(2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2),
			(4, 4, 1, 1, 1, 1, 4, 1, 1, 1, 4, 1),
			(4, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 4),
			(1, 1, 4, 4, 1, 1, 4, 1, 4, 1, 1, 1),
			(1, 1, 4, 4, 1, 1, 1, 4, 1, 4, 1, 1),
			(1, 1, 1, 1, 4, 4, 1, 1, 4, 1, 4, 1),
		])
	@test result == expected_result
end

@testset "toric_code" begin
	t = TensorQEC.ToricCode(2, 2)
	result = TensorQEC.stabilizers(t)
	code = TensorQEC.stabilizers2bimatrix(result)
	@test code.xcodenum == 3
	@test code.ordering == collect(1:8)
	@test code.matrix == [
		1  0  1  0  1  1  0  0  0  0  0  0  0  0  0  0;
		0  1  0  1  1  1  0  0  0  0  0  0  0  0  0  0;
		1  0  1  0  0  0  1  1  0  0  0  0  0  0  0  0;
		0  0  0  0  0  0  0  0  1  1  0  0  1  0  1  0;
		0  0  0  0  0  0  0  0  1  1  0  0  0  1  0  1;
		0  0  0  0  0  0  0  0  0  0  1  1  1  0  1  0
	]
end

@testset "bimatrix2stabilizers" begin
	t = TensorQEC.ToricCode(3, 3)
	result = TensorQEC.stabilizers(t)
	code = TensorQEC.stabilizers2bimatrix(result)
	stabilizers = TensorQEC.bimatrix2stabilizers(code)
	@test stabilizers == result
end

@testset "gaussian_elimination" begin
	t = TensorQEC.ToricCode(3, 3)
	result = TensorQEC.stabilizers(t)
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
	t = TensorQEC.ToricCode(2, 2)
	result = TensorQEC.stabilizers(t)
	code = TensorQEC.stabilizers2bimatrix(result)
	TensorQEC.gaussian_elimination!(code)
	stabilizers = TensorQEC.bimatrix2stabilizers(code)
	qc = TensorQEC.encode_circuit(code)
	# display(vizcircuit(qc))
	u = mat(ComplexF64, qc)
	for i in 1:size(code.matrix, 1)
		@test u * mat(ComplexF64, put(8, code.ordering[i] => Z)) * u' ≈ mat(ComplexF64, stabilizers[i])
	end
end


@testset "nine_qubit_surfacecode" begin
	result = TensorQEC.stabilizers(SurfaceCode{3}())
	code = TensorQEC.stabilizers2bimatrix(result)
	TensorQEC.gaussian_elimination!(code)
	stabilizers = TensorQEC.bimatrix2stabilizers(code)
	qc = TensorQEC.encode_circuit(code)
	# display(vizcircuit(qc))
	u = mat(ComplexF64, qc)
	for i in 1:size(code.matrix, 1)
		@test u * mat(ComplexF64, put(9, code.ordering[i] => Z)) * u' ≈ mat(ComplexF64, stabilizers[i])
	end
end


@testset "encode_stabilizers" begin
	result = TensorQEC.stabilizers(SurfaceCode{3}())
	qc, data_qubits, bimat = TensorQEC.encode_stabilizers(result)
	u = mat(ComplexF64, qc)
	stabilizers = TensorQEC.bimatrix2stabilizers(bimat)
	for i in 1:size(bimat.matrix, 1)
		@test u * mat(ComplexF64, put(9, bimat.ordering[i] => Z)) * u' ≈ mat(ComplexF64, stabilizers[i])
	end
end
using Test, TensorQEC, TensorQEC.Yao, TensorQEC.LinearAlgebra

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

@testset "guassian_elimination!" begin
	t = TensorQEC.ToricCode(3, 3)
	result = TensorQEC.stabilizers(t)
	code = TensorQEC.stabilizers2bimatrix(result)
	TensorQEC.guassian_elimination!(code)
	@test code.matrix[1:8, 1:8] == [
		1  0  0  0  0  0  0  0;
		0  1  0  0  0  0  0  0;
		0  0  1  0  0  0  0  0;
		0  0  0  1  0  0  0  0;
		0  0  0  0  1  0  0  0;
		0  0  0  0  0  1  0  0;
		0  0  0  0  0  0  1  0;
		0  0  0  0  0  0  0  1
	]
	@test code.matrix[9:16, 27:34] == [
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
end

@testset "quantum chain block" begin
	t = TensorQEC.ToricCode(2, 2)
	result = TensorQEC.stabilizers(t)
	code = TensorQEC.stabilizers2bimatrix(result)
	TensorQEC.guassian_elimination!(code)
	stabilizers = TensorQEC.bimatrix2stabilizers(code)
	qc = TensorQEC.encode_circuit(code)
	# display(vizcircuit(qc))
	u = mat(ComplexF64, qc)
	for i in 1:size(code.matrix, 1)
		@test u * mat(ComplexF64, put(8, code.ordering[i] => Z)) * u' ≈ mat(ComplexF64, stabilizers[i])
	end
end

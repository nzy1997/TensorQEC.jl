using Test, TensorQEC, TensorQEC.Yao, TensorQEC.LinearAlgebra, TensorQEC.Yao
using Random

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

@testset "ShorCode" begin
	st = stabilizers(ShorCode())
	qcen, data_qubits, code = encode_stabilizers(st)
	qcen = chain(9, put(9, 9 => H), qcen)

	# |0> and |1> state
	reg0 = zero_state(9)
	apply!(reg0, qcen)
	qc = chain(3, put(3, 1 => H), control(3, 1, 2 => X), control(3, 1, 3 => X))
	plusreg = apply!(zero_state(3),qc)
	@test reg0.state == join(plusreg, plusreg,plusreg).state

	reg1 = zero_state(9)
	apply!(reg1, put(9, 9 => X))
	apply!(reg1, qcen)
	qc = chain(3,put(3, 1 => X), put(3, 1 => H), control(3, 1, 2 => X), control(3, 1, 3 => X))
	minusreg = apply!(zero_state(3),qc)
	@test reg1.state == join(minusreg, minusreg,minusreg).state

	# encode random state
	Random.seed!(43565)
	regrs = rand_state(1)
	reg = join(regrs, zero_state(8))
	apply!(reg, qcen)

	@test (reg0.state'*reg.state)[1] ≈ regrs.state[1]
	@test (reg1.state'*reg.state)[1] ≈ regrs.state[2]
end

@testset "ShorCode transversal cnot" begin
	st = stabilizers(ShorCode())
	qcen, data_qubits, code = encode_stabilizers(st)
	qcen = chain(9, put(9, 9 => H), qcen)

	regrs = rand_state(1)
	# Z error, X stabilizer, logical |+>
	reg = join(zero_state(9),regrs, zero_state(8))	

	qc = chain(18,subroutine(18,qcen,1:9),put(18,18=>H),subroutine(18,qcen,10:18))
	regen = apply(reg,qc)
	[push!(qc,control(18,9+i,i=>X)) for i in 1:9]
	# push!(qc,subroutine(18,qcen',1:9))
	# display(vizcircuit(qc))

	apply!(reg,qc)
	# focus!(reg,9)
	@show fidelity(reg,regen)

	# X error, Z stabilizer, logical |0>
	reg = join(zero_state(9),regrs, zero_state(8))	

	qc = chain(18,subroutine(18,qcen,1:9),subroutine(18,qcen,10:18))
	[push!(qc,control(18,i,9+i=>X)) for i in 1:9]
	push!(qc,subroutine(18,qcen',1:9))
	# display(vizcircuit(qc))

	apply!(reg,qc)
	focus!(reg,9)
	@test fidelity(reg,regrs) ≈ 1
end

@testset "ShorCode transversal cnot" begin
	st = stabilizers(ShorCode())
	qcen, data_qubits, code = encode_stabilizers(st)
	
	regrs = rand_state(1)
	# regrs = zero_state(1)
	reg1 = join(zero_state(9),regrs, zero_state(8))

	focus!(reg1,1:9)
	apply!(reg1,qcen)
	relax!(reg1)

	apply!(reg1,put(18,18=>H))

	focus!(reg1,10:18)
	apply!(reg1,qcen)
	relax!(reg1)


	reg2 = join(zero_state(9),regrs, zero_state(8))
	apply!(reg2,subroutine(18,qcen,1:9))
	apply!(reg2,put(18,18=>H))
	apply!(reg2,subroutine(18,qcen,10:18))	
	@show fidelity(reg2,reg1)
end

@testset "subroutine and focus" begin
	reg = rand_state(2)
	reg1 = copy(reg)
	focus!(reg1,2)
	u = rand_unitary(2)
	apply!(reg1,matblock(u))
	relax!(reg1)

	reg2 = copy(reg)
	apply!(reg2,subroutine(2,matblock(u),2))
	@show fidelity(reg2,reg1)

	reg3 = copy(reg)
	apply!(reg3,put(2,2=>matblock(u)))
	@show fidelity(reg1,reg3)
end
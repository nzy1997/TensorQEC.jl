using Test, TensorQEC, TensorQEC.Yao, TensorQEC.LinearAlgebra
using Random

@testset "toric code" begin
	t = ToricCode(2, 3)
	result = stabilizers(t)
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
	t = ToricCode(2, 2)
	result = stabilizers(t)
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

@testset "surfacecode" begin
	result = stabilizers(SurfaceCode(3,3))
	code = TensorQEC.stabilizers2bimatrix(result)
	TensorQEC.gaussian_elimination!(code)
	st = TensorQEC.bimatrix2stabilizers(code)
	qc = TensorQEC.encode_circuit(code)
	u = mat(ComplexF64, qc)
	for i in 1:size(code.matrix, 1)
		@test u * mat(ComplexF64, put(9, code.ordering[i] => Z)) * u' ≈ mat(ComplexF64, st[i])
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

	apply!(reg,qc)
	@test fidelity(reg,regen) ≈ 1

	# X error, Z stabilizer, logical |0>
	reg = join(zero_state(9),regrs, zero_state(8))	

	qc = chain(18,subroutine(18,qcen,1:9),subroutine(18,qcen,10:18))
	[push!(qc,control(18,i,9+i=>X)) for i in 1:9]
	push!(qc,subroutine(18,qcen',1:9))
	apply!(reg,qc)
	focus!(reg,9)
	@test fidelity(reg,regrs) ≈ 1
end

@testset "code832" begin
    st = stabilizers(Code832())
    qcen, data_qubits, code = encode_stabilizers(st)
    regrs = rand_state(3)
    reg = place_qubits(regrs, data_qubits, nqubits(qcen))
end

@testset "code422" begin
	st = stabilizers(Code422())
	@test st[1] == PauliString((2,2,2,2))
	@test st[2] == PauliString((4,4,4,4))
end
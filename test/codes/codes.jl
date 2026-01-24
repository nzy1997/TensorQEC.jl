using Test, TensorQEC, TensorQEC.Yao, TensorQEC.LinearAlgebra
using Random
using QECCore

@testset "Code type hierarchy" begin
    @test SurfaceCode(3, 3) isa AbstractCSSCode
    @test ShorCode() isa AbstractCSSCode
    @test SteaneCode() isa AbstractCSSCode
    @test Code832() isa AbstractCSSCode
    @test Code422() isa AbstractCSSCode
    @test Code1573() isa AbstractCSSCode
    @test Code513() isa AbstractQECC
    @test !(Code513() isa AbstractCSSCode)
    @test Color488(3) isa AbstractCSSCode
    @test Color666(3) isa AbstractCSSCode
    @test ToricCode(3,3) isa AbstractCSSCode
end

@testset "toric code" begin
	t = ToricCode(2, 3)
	result = stabilizers(t)
	expected_result =
		PauliString.(map(x->Pauli.(x .- 1), [
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
		]))
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

	@test code_distance(CSSTannerGraph(result)) == 2
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

	@test code_distance(CSSTannerGraph(result)) == 3

	tanner = CSSTannerGraph(SurfaceCode(6, 6))
	@test code_distance(tanner) == 6
end


@testset "ShorCode" begin
	st = stabilizers(ShorCode())

	@test code_distance(CSSTannerGraph(st)) == 3
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

	@test code_distance(CSSTannerGraph(st)) == 2
end

@testset "code422" begin
	st = stabilizers(Code422())
	@test st[1] == PauliString(ntuple(i->Pauli(1), 4)...)
	@test st[2] == PauliString(ntuple(i->Pauli(3), 4)...)

	@test code_distance(CSSTannerGraph(st)) == 2
end

@testset "code1573" begin
	st = stabilizers(Code1573())
	@test code_distance(CSSTannerGraph(st)) == 3
end

@testset "ShorCode" begin
	st = stabilizers(ShorCode())
	@test code_distance(CSSTannerGraph(st)) == 3
end

@testset "SteaneCode" begin
	st = stabilizers(SteaneCode())
	@test code_distance(CSSTannerGraph(st)) == 3
end

@testset "Code513" begin
	st = stabilizers(Code513())
end

@testset "[[98,6,12]] BivariateBicycleCode" begin
	st = stabilizers(BivariateBicycleCode(7,7, ((1,0),(0,3),(0,4)), ((0,1),(3,0),(4,0))))
	@test length(st) == 92
	# @test code_distance(CSSTannerGraph(st)) == 12
end

@testset "[[144,12,12]] BivariateBicycleCode" begin
	st = stabilizers(BivariateBicycleCode(6,12, ((3,0),(0,1),(0,2)), ((0,3),(1,0),(2,0))))
	@test length(st) == 132
	# @test code_distance(CSSTannerGraph(st)) == 12
	tanner = CSSTannerGraph(st)
	lx,lz = logical_operator(tanner)
	for i in 1:11
		push!(st,PauliString(144, findall(i->i.x,lz[i,:]) => Pauli(3)))
	end
	tanner = CSSTannerGraph(st)
	# @test code_distance(tanner) == 12
	lx,lz = logical_operator(tanner)
	@test size(lx,1) == 1
	@test size(lz,1) == 1
end

@testset "Color488" begin
	d = 5
	st = stabilizers(Color488(d))
	@test length(st) == 16
	@test code_distance(CSSTannerGraph(st)) == d
end

@testset "Color666" begin
	d = 5
	st = stabilizers(Color666(d))
	@test length(st) == 18
	@test code_distance(CSSTannerGraph(st)) == d
end

@testset "QECCore" begin
	st = stabilizers(Cleve8())
	@test st isa Vector{PauliString{8}}
	@test length(st) == 5

	c = Toric(2,2)
	st = stabilizers(c)
	tanner = CSSTannerGraph(st)
	@test code_distance(tanner) == 2
	@test tanner.stgx.H == Mod2[1  0  1  0  1  1  0  0;
	0  1  0  1  1  1  0  0;
	1  0  1  0  0  0  1  1]
	@test tanner.stgz.H == Mod2[1  1  0  0  1  0  1  0;
	1  1  0  0  0  1  0  1;
	0  0  1  1  1  0  1  0]
end

@testset "QECCore interface" begin
	using QECCore: code_n, code_s, code_k, parity_matrix, parity_matrix_x, parity_matrix_z

	@testset "SurfaceCode(3,3)" begin
		sc = SurfaceCode(3, 3)
		@test code_n(sc) == 9
		@test code_s(sc) == 8
		@test code_k(sc) == 1
		pm = parity_matrix(sc)
		@test size(pm) == (8, 18)
		pmx = parity_matrix_x(sc)
		pmz = parity_matrix_z(sc)
		@test size(pmx, 2) == 9
		@test size(pmz, 2) == 9
		@test size(pmx, 1) + size(pmz, 1) == 8
	end

	@testset "SteaneCode" begin
		steane = SteaneCode()
		@test code_n(steane) == 7
		@test code_s(steane) == 6
		@test code_k(steane) == 1
		pm = parity_matrix(steane)
		@test size(pm) == (6, 14)
		pmx = parity_matrix_x(steane)
		pmz = parity_matrix_z(steane)
		@test size(pmx) == (3, 7)
		@test size(pmz) == (3, 7)
	end

	@testset "Code513" begin
		c = Code513()
		@test code_n(c) == 5
		@test code_s(c) == 4
		@test code_k(c) == 1
		pm = parity_matrix(c)
		@test size(pm) == (4, 10)
		# Code513 stabilizers have both X and Z entries (on different positions per row)
		@test any(pm[:, 1:5]) && any(pm[:, 6:10])
	end

	@testset "ToricCode" begin
		tc = ToricCode(3, 3)
		@test code_n(tc) == 18
		@test code_k(tc) == 2
		pm = parity_matrix(tc)
		@test size(pm, 2) == 36
		pmx = parity_matrix_x(tc)
		pmz = parity_matrix_z(tc)
		@test size(pmx, 2) == 18
		@test size(pmz, 2) == 18
	end

	@testset "ShorCode" begin
		sc = ShorCode()
		@test code_n(sc) == 9
		@test code_s(sc) == 8
		@test code_k(sc) == 1
		pm = parity_matrix(sc)
		@test size(pm) == (8, 18)
		pmx = parity_matrix_x(sc)
		pmz = parity_matrix_z(sc)
		@test size(pmx, 1) == 2  # 2 X stabilizers
		@test size(pmz, 1) == 6  # 6 Z stabilizers
	end

	@testset "Code832" begin
		c = Code832()
		@test code_n(c) == 8
		@test code_s(c) == 5
		@test code_k(c) == 3
		pm = parity_matrix(c)
		@test size(pm) == (5, 16)
	end

	@testset "Code422" begin
		c = Code422()
		@test code_n(c) == 4
		@test code_s(c) == 2
		@test code_k(c) == 2
	end

	@testset "Code1573" begin
		c = Code1573()
		@test code_n(c) == 15
		@test code_s(c) == 8
		@test code_k(c) == 7
	end

	@testset "Color488" begin
		c = Color488(3)
		@test code_n(c) == 7
		@test code_k(c) == 1
		pmx = parity_matrix_x(c)
		pmz = parity_matrix_z(c)
		@test size(pmx, 1) == size(pmz, 1)
	end

	@testset "Color666" begin
		c = Color666(3)
		@test code_n(c) == 7
		@test code_k(c) == 1
	end
end
using TensorQEC, Test, TensorQEC.Yao
using TensorQEC: pauli_repr

@testset "perm repr" begin
	m = pauli_repr(H)
	pm = TensorQEC.to_perm_matrix(Int8, Int, m)
	@test Matrix(pm) ≈ m
	pm = TensorQEC.to_perm_matrix(H)
	@test Matrix(pm) ≈ m

	m = pauli_repr(ConstGate.CNOT)
	pm = TensorQEC.to_perm_matrix(Int8, Int, m)
	@test Matrix(pm) ≈ m
end

@testset "generate group" begin
	@test TensorQEC.generate_group([1im]) |> length == 4
end

@testset "pauli group" begin
	@test size(pauli_group(1)) == (4, 4)
	@test all(getfield.(pauli_group(1)[1, :], :first) .== 0)
	@test all(getfield.(pauli_group(1)[2, :], :first) .== 1)
	@test size(pauli_group(2)) == (4, 4, 4)
end

@testset "clifford group" begin
	csize(n::Int) = prod(k -> 2 * 4^k * (4^k - 1), 1:n)
	@test length(clifford_group(1)) == csize(1)
	@test length(clifford_group(2)) == csize(2)
end

@testset "perm_of_paulistring" begin
	pm = TensorQEC.to_perm_matrix(Int8, Int, pauli_repr(H))
	ps = PauliString((1, 2))
	ps2, val = TensorQEC.perm_of_paulistring(ps, [2]=>pm)
	@test ps2 == PauliString((1, 4))

	pmcn = TensorQEC.to_perm_matrix(Int8, Int, TensorQEC.pauli_repr(ConstGate.CNOT))
	ps = PauliString((2, 1, 3, 2))
	ps2, val = TensorQEC.perm_of_paulistring(ps, [4, 2]=>pmcn)
	@test ps2.ids == (2, 2, 3, 2)

	ps = PauliString((2, 4, 3, 2))
	ps2, val = TensorQEC.perm_of_paulistring(ps, [3, 2]=>pmcn)
	@test ps2.ids == (2, 3, 2, 2)
end

@testset "perm_of_pauligroup" begin
	ps = PauliString(2,3,4,3,2,1)
	pg = PauliGroup(1, ps)
	pm = TensorQEC.to_perm_matrix(Int8, Int, pauli_repr(ConstGate.CNOT))

	pg2 = perm_of_pauligroup(pg, [2, 3]=>pm)
	ps2, val = perm_of_paulistring(ps, [2, 3]=>pm)
	@test pg2 == PauliGroup(1, ps2)
end

@testset "clifford_simulate" begin
	qc = chain(put(5, 1 => H), control(5, 1, 2 => Z), control(5, 3, 4 => X), control(5, 5, 3 => X), put(5, 1 => X))
	ps = PauliString((4, 3, 1, 3, 2))
	
	res = clifford_simulate(ps, qc)
	ps2 = res.output
	val = res.phase
	@test val * mat(qc) * mat(ps) * mat(qc)' ≈ mat(ps2)
end

@testset "annotate_history" begin
	st = stabilizers(SteaneCode())
	table = make_table(st, 1)
	qcen, data_qubits, code = encode_stabilizers(st) 
    qcm ,st_pos  = measure_circuit_steane(data_qubits[1],st)
	ps0 = paulistring(27,3,6) 
	res = clifford_simulate(ps0, qcm)
	annotate_history(res)
	@test res.output.ids == (1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 4, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2)

	annotate_circuit_pics(res)
end


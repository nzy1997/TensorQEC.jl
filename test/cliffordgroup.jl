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

@testset "clifford_simulate" begin
	qc = chain(put(5, 1 => H), control(5, 1, 2 => Z), control(5, 3, 4 => X), control(5, 5, 3 => X), put(5, 1 => X))
	ps = PauliString((4, 3, 1, 3, 2))
	ps2, val = TensorQEC.clifford_simulate(ps, qc)
	@test val * mat(qc) * mat(ps) * mat(qc)' ≈ mat(ps2)
end

@testset "annotate_history" begin
	st = stabilizers(SteaneCode())
	table = make_table(st, 1)
	qcen, data_qubits, code = encode_stabilizers(st) 
    qcm ,st_pos, num_qubits = measure_circuit_steane(data_qubits[1],st,3)
	ps0 = paulistring(27,3,6) 
	res = clifford_simulate(ps0, qcm)
	annotate_history(res)
end

@testset "annotate_history" begin
	YaoPlots.darktheme!()
	st = stabilizers(SteaneCode())
	table = make_table(st, 1)
	qcen, data_qubits, code = encode_stabilizers(st) 
    qcm ,st_pos, num_qubits = measure_circuit_steane_single_type(data_qubits[1],st[4:6],false)
	ps0 = paulistring(17,2,6) 
	res = clifford_simulate(ps0, qcm)
	qcf,pos = TensorQEC.generate_annotate_circuit(res)

	qcx = paulistring_annotate(paulistring(17,2,(6,13,16,17)))
	qci = paulistring_annotate(paulistring(17,2,(13,16,17)))
	push!(qci, put(17, 6 => line_annotation("I";color = "red")))
	qccr = chain(17,
	control(17,(15,-16,-17),1=>X),
	qcx,
	control(17,(-15,16,-17),2=>X),
	qcx,
	control(17,(15,16,-17),3=>X),
	qcx,
	control(17,(-15,-16,17),4=>X),
	qcx,
	control(17,(15,-16,17),5=>X),
	qcx,
	control(17,(-15,16,17),6=>X),
	qci,
	control(17,(15,16,17),7=>X),
	qci
	)

	push!(qcf,qccr)
	TensorQEC.annotate_circuit(qcf;filename="test.png")
end
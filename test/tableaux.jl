using Test, TensorQEC, TensorQEC.Yao, TensorQEC.LinearAlgebra

@testset "new_tableau" begin
	tab2 = new_tableau(2)
	@test tab2.tabx == [paulistring(2,2,1), paulistring(2,2,2)]
	@test tab2.tabz == [paulistring(2,4,1), paulistring(2,4,2)]

	tab3 = new_tableau(3)
	@test tab3.tabx == [paulistring(3,2,1), paulistring(3,2,2), paulistring(3,2,3)]
	@test tab3.tabz == [paulistring(3,4,1), paulistring(3,4,2), paulistring(3,4,3)]
end

@testset "tableau_simulate" begin
	tab = new_tableau(2)
	tab = tableau_simulate(tab, [1]=>to_perm_matrix(Int8, Int, TensorQEC.pauli_repr(H)))
	@test tab.tabx == [paulistring(2,4,1), paulistring(2,2,2)]
	@test tab.tabz == [paulistring(2,2,1), paulistring(2,4,2)]

	tab = new_tableau(3)
	tab = tableau_simulate(tab, [1,3]=>to_perm_matrix(Int8, Int, TensorQEC.pauli_repr(ConstGate.CNOT)))
	@test tab.tabx == [paulistring(3,2,(1,3)), paulistring(3,2,2), paulistring(3,2,3)]
	@test tab.tabz == [paulistring(3,4,(1)), paulistring(3,4,2), paulistring(3,4,(1,3))]

	qc = chain(put(5, 1 => H), control(5, 1, 2 => Z), control(5, 3, 4 => X), control(5, 5, 3 => X), put(5, 1 => X))
	ps = PauliString((4, 3, 1, 3, 2))
	
	res = clifford_simulate(ps, qc)
	res2 = tableau_simulate(ps, qc)
end
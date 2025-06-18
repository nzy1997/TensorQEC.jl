using Test, TensorQEC, TensorQEC.Yao, TensorQEC.LinearAlgebra

@testset "new_tableau" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
	tab2 = new_tableau(2)
	@test getfield.(tab2.tabx,:ps) == [PauliString(2, 1=>x), PauliString(2, 2=>x)]
	@test getfield.(tab2.tabz,:ps) == [PauliString(2, 1=>z), PauliString(2, 2=>z)]

	tab3 = new_tableau(3)
	@test getfield.(tab3.tabx,:ps) == [PauliString(3, 1=>x), PauliString(3, 2=>x), PauliString(3, 3=>x)]
	@test getfield.(tab3.tabz,:ps) == [PauliString(3, 1=>z), PauliString(3, 2=>z), PauliString(3, 3=>z)]
end

@testset "tableau_simulate" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
	tab = new_tableau(2)
	tab2 = tableau_simulate(tab, (1,)=>pauli_repr(Clifford(H)))
	@test getfield.(tab2.tabx,:ps) == [PauliString(2, 1=>z), PauliString(2, 2=>x)]
	@test getfield.(tab2.tabz,:ps) == [PauliString(2, 1=>x), PauliString(2, 2=>z)]

	tab = new_tableau(3)
	tab2 = tableau_simulate(tab, (1,3)=>pauli_repr(Clifford(ConstGate.CNOT)))
	@test getfield.(tab2.tabx,:ps) == [PauliString(3, (1,3)=>x), PauliString(3, 2=>x), PauliString(3, 3=>x)]
	@test getfield.(tab2.tabz,:ps) == [PauliString(3, 1=>z), PauliString(3, 2=>z), PauliString(3, (1,3)=>z)]

	qc = chain(put(5, 1 => H), control(5, 1, 2 => Z), control(5, 3, 4 => X), control(5, 5, 3 => X), put(5, 1 => X))
	ps = PauliString((z, y, i, y, x))
	res = clifford_simulate(ps, qc)
	res2 = tableau_simulate(ps, qc)
	@test res.output == res2.ps
	@test res.phase == im^res2.coeff
end
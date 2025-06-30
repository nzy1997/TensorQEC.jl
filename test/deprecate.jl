using TensorQEC, Test, Yao

@testset "deprecate" begin
    @test PauliGroup(P"XX") isa PauliGroupElement
    @test densitymatrix2sumofpaulis(rand_density_matrix(2)) isa SumOfPaulis
    @test arrayreg2sumofpaulis(rand_state(2)) isa SumOfPaulis
    @test PauliString(1, 2, 3) == PauliString((Pauli(0), Pauli(1), Pauli(2)))
    @test paulistring(2, 1, 2) == PauliString(2, 2 => Pauli(0))
    @test apply!(ArrayReg(bit"00"), PauliString(2, 2 => Pauli(0))) == ArrayReg(bit"00")
    gate = rand_unitary(4)
    @test pauli_mapping(gate) == reshape(pauli_repr(gate), fill(4, 2*log2i(size(gate, 1)))...)
    @test to_perm_matrix(H) == to_perm_matrix(pauli_repr(H))
    @test perm_of_pauligroup(P"XXX", (1, 2)=>CliffordGate(ConstGate.CNOT)) == PauliGroupElement(P"XIX")
end
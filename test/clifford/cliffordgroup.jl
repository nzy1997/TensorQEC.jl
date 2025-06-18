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
    @test all(getfield.(pauli_group(1)[1, :], :coeff) .== 0)
    @test all(getfield.(pauli_group(1)[2, :], :coeff) .== 1)
    @test size(pauli_group(2)) == (4, 16)
end

@testset "clifford group" begin
    csize(n::Int) = prod(k -> 2 * 4^k * (4^k - 1), 1:n)
    @test length(clifford_group(1)) == csize(1)
    @test length(clifford_group(2)) == csize(2)
end

@testset "perm_of_paulistring" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    pm = TensorQEC.to_perm_matrix(Int8, Int, pauli_repr(H))
    ps = PauliString((i, x))
    ps2, val = TensorQEC.perm_of_paulistring(ps, (2,) => pm)
    @test ps2 == PauliString((i, z))

    pmcn = TensorQEC.to_perm_matrix(Int8, Int, TensorQEC.pauli_repr(ConstGate.CNOT))
    ps = PauliString((x, i, y, x))
    ps2, val = TensorQEC.perm_of_paulistring(ps, (4, 2) => pmcn)
    @test ps2.operators == (x, x, y, x)

    ps = PauliString((x, z, y, x))
    ps2, val = TensorQEC.perm_of_paulistring(ps, (3, 2) => pmcn)
    @test ps2.operators == (x, y, x, x)

    # asymmetric case
    pmasym = TensorQEC.to_perm_matrix(Int8, Int, TensorQEC.pauli_repr(ConstGate.S))
    @test pmasym == [1 0 0 0; 0 0 -1 0; 0 1 0 0; 0 0 0 1]
    ps = P"X"
    @test mat(ConstGate.S * X * ConstGate.S') == [0 -im; im 0]
    ps2, val = TensorQEC.perm_of_paulistring(ps, (1,) => pmasym)
    @test ps2 == P"Y"
    @test val == 1
    ps = P"Y"
    @test mat(ConstGate.S * Y * ConstGate.S') == [0 -1; -1 0]
    ps2, val = TensorQEC.perm_of_paulistring(ps, (1,) => pmasym)
    @test ps2 == P"X"
    @test val == -1
end

@testset "perm_of_pauligroup" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    ps = PauliString((x, y, z, y, x, i))
    pg = PauliGroupElement(1, ps)
    pm = TensorQEC.to_perm_matrix(Int8, Int, pauli_repr(ConstGate.CNOT))

    pg2 = perm_of_pauligroup(pg, (2, 3) => pm)
    ps2, val = perm_of_paulistring(ps, (2, 3) => pm)
    @test pg2 == PauliGroupElement(1, ps2)
end

@testset "clifford_simulate" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    qc = chain(put(5, 1 => H), control(5, 1, 2 => Z), control(5, 3, 4 => X), control(5, 5, 3 => X), put(5, 1 => X))
    ps = PauliString((z, y, i, y, x))

    res = clifford_simulate(ps, qc)
    ps2 = res.output
    val = res.phase
    @test val * mat(qc) * mat(ps) * mat(qc)' ≈ mat(ps2)
end

@testset "annotate_history" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    st = stabilizers(SteaneCode())
    qcen, data_qubits, code = encode_stabilizers(st)
    qcm, st_pos = measure_circuit_steane(data_qubits[1], st)
    ps0 = PauliString(27, 6 => y)
    res = clifford_simulate(ps0, qcm)
    annotate_history(res)
    @test getfield.(res.output.operators, :id) .+ 1 == (1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 4, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2)

    annotate_circuit_pics(res)
end


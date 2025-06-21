using TensorQEC, Test, TensorQEC.Yao
using TensorQEC: clifford_group, pauli_group, PermMatrixCSC

@testset "perm repr" begin
    m = pauli_repr(H)
    pm = CliffordGate(H)
    @test nqubits(pm) == 1
    @test mat(pm) ≈ m
    @test string(pm) == "CliffordGate(nqubits = 1)\n I → I\n X → Z\n Y → -Y\n Z → X"

    m = pauli_repr(ConstGate.CNOT)
    pm = CliffordGate(ConstGate.CNOT)
    @test mat(pm) ≈ m
    @test string(pm) == "CliffordGate(nqubits = 2)\n II → II\n XI → XX\n YI → YX\n ZI → ZI\n IX → IX\n XX → XI\n YX → YI\n ZX → ZX\n IY → ZY\n XY → YZ\n YY → -XZ\n ZY → IY\n IZ → ZZ\n XZ → -YY\n YZ → XY\n ZZ → IZ"
    @test string(CliffordGate(ConstGate.S)) == "CliffordGate(nqubits = 1)\n I → I\n X → Y\n Y → -X\n Z → Z"
end

@testset "generate group" begin
    @test TensorQEC.generate_group([1im]) |> length == 4
end

@testset "pauli group" begin
    @test size(pauli_group(1)) == (4, 4)
    @test all(getfield.(pauli_group(1)[1, :], :phase) .== 0)
    @test all(getfield.(pauli_group(1)[2, :], :phase) .== 1)
    @test size(pauli_group(2)) == (4, 16)
end

@testset "clifford group" begin
    csize(n::Int) = prod(k -> 2 * 4^k * (4^k - 1), 1:n)
    @test length(clifford_group(1)) == csize(1)
    @test length(clifford_group(2)) == csize(2)
end

@testset "apply clifford gate" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    pm = CliffordGate(H)
    @test pm isa CliffordGate
    ps = PauliString((i, x))
    elem = pm(ps, (2,))
    @test elem.ps == PauliString((i, z))
    @test elem.phase == 0

    pmcn = CliffordGate(ConstGate.CNOT)
    @test pmcn isa CliffordGate
    ps = PauliString((x, i, y, x))
    elem = pmcn(ps, (4, 2))
    @test elem.ps.operators == (x, x, y, x)
    @test elem.phase == 0

    ps = PauliString((x, z, y, x))
    elem = pmcn(ps, (3, 2))
    @test elem.ps.operators == (x, y, x, x)
    @test elem.phase == 0

    # asymmetric case
    pmasym = CliffordGate(ConstGate.S)
    @test pmasym isa CliffordGate
    @test mat(pmasym) == [1 0 0 0; 0 0 -1 0; 0 1 0 0; 0 0 0 1]
    ps = P"X"
    @test mat(ConstGate.S * X * ConstGate.S') == [0 -im; im 0]
    elem = pmasym(ps, (1,))
    @test elem.ps == P"Y"
    @test elem.phase == 0
    ps = P"Y"
    @test mat(ConstGate.S * Y * ConstGate.S') == [0 -1; -1 0]
    elem = pmasym(ps, (1,))
    @test elem.ps == P"X"
    @test elem.phase == 2
end

@testset "apply pauli group element" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    ps = PauliString((x, y, z, y, x, i))
    pg = PauliGroupElement(1, ps)
    pm = CliffordGate(ConstGate.CNOT)
    @test pm isa CliffordGate

    pg2 = pm(pg, (2, 3))
    pgs = pm(ps, (2, 3))
    @test pg2 == PauliGroupElement(1, pgs.ps)
end

@testset "compile_clifford_circuit" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    qc = chain(put(5, 1 => H), control(5, 1, 2 => Z), control(5, 3, 4 => X), put(5, 2 => H), control(5, 5, 3 => X), put(5, 1 => X))
    cl = TensorQEC.compile_clifford_circuit(qc)
    pg = PauliGroupElement(1, PauliString((z, y, i, y, x)))
    pg2 = cl(pg)
    @test mat(qc) * mat(pg) * mat(qc)' ≈ mat(pg2)
end

@testset "clifford_simulate" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    qc = chain(put(5, 1 => H), control(5, 1, 2 => Z), control(5, 3, 4 => X), put(5, 2 => H), control(5, 5, 3 => X), put(5, 1 => X))
    ps = PauliString((z, y, i, y, x))

    res = clifford_simulate(ps, qc)
    pg2 = res.output
    @test mat(qc) * mat(ps) * mat(qc)' ≈ mat(pg2)
end

@testset "annotate_history" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    st = stabilizers(SteaneCode())
    qcen, data_qubits, code = encode_stabilizers(st)
    qcm, st_pos = measure_circuit_steane(data_qubits[1], st)
    ps0 = PauliString(27, 6 => y)
    res = clifford_simulate(ps0, qcm)
    TensorQEC.annotate_history(res)
    @test getfield.(res.output.ps.operators, :id) .+ 1 == (1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 4, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2)

    TensorQEC.annotate_circuit_pics(res)
end


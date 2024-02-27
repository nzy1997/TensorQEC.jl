using Test, TensorQEC, TensorQEC.Yao


# simple example: Z1Z3 Z2Z3 stabilizers on 3 qubits
@testset "simple example inference and error correct" begin
    qubit_num = 3
    st = [PauliString((1,4,4)),PauliString((4,1,4))]
    qc, data_qubits, code = TensorQEC.encode_stabilizers(st)

    reg = join(rand_state(1), zero_state(2))  # join(qubit3, qubit2, qubit1)
    regcopy = copy(reg)
    apply!(reg, qc)
    apply!(reg, put(qubit_num, 3=>X))

    measure_outcome=measure_syndrome!(reg, st)
    syn_dict=TensorQEC.generate_syndrome_dict(code,syndrome_transform(code, measure_outcome))

    cl = clifford_network(qc)
    p = fill([0.85,0.05,0.05,0.05],qubit_num)
    pinf = syndrome_inference(cl, syn_dict, p)
    ps_ec_phy = TensorQEC.pauli_string_map_iter(correction_pauli_string(qubit_num, syn_dict, pinf), qc)
    @show ps_ec_phy
    apply!(reg, Yao.YaoBlocks.Optimise.to_basictypes(ps_ec_phy))

    @test measure_syndrome!(reg, st) == [1,1]
    apply!(reg, qc')
    @test fidelity(density_matrix(reg, [data_qubits...]), density_matrix(regcopy, [data_qubits...])) ≈ 1.0
end

@testset "syndrome inference and error correct for SurfaceCode3" begin
    qubit_num = 9
    st = stabilizers(SurfaceCode{3}())
    qc, data_qubits, code = TensorQEC.encode_stabilizers(st)

    reg = join(zero_state(8),rand_state(1))
    regcopy = copy(reg)
    apply!(reg, qc)
    apply!(reg, put(9, 3=>X))
    apply!(reg, put(9, 2=>X))

    measure_outcome=measure_syndrome!(reg, st)
    syn_dict=TensorQEC.generate_syndrome_dict(code,syndrome_transform(code, measure_outcome))

    cl = clifford_network(qc)
    p = fill([0.85,0.05,0.05,0.05],qubit_num)
    pinf = syndrome_inference(cl, syn_dict, p)

    ps_ec_phy = TensorQEC.pauli_string_map_iter(correction_pauli_string(qubit_num, syn_dict, pinf), qc)
    @show ps_ec_phy
    apply!(reg, Yao.YaoBlocks.Optimise.to_basictypes(ps_ec_phy))

    @test measure_syndrome!(reg, st) == [1,1,1,1,1,1,1,1]
    apply!(reg, qc')
    @test fidelity(density_matrix(reg, [9]),density_matrix(regcopy, [9])) ≈ 1.0
end
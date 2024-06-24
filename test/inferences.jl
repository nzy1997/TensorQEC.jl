using Test, TensorQEC, TensorQEC.Yao
@testset "measure_syndrome!" begin
    result=stabilizers(SurfaceCode{3,3}())
    code = TensorQEC.stabilizers2bimatrix(result)
    TensorQEC.gaussian_elimination!(code)
    qc = TensorQEC.encode_circuit(code)
    #display(vizcircuit(qc))

    # test X error
    reg = join(rand_state(1), zero_state(8))
    apply!(reg, qc)
    apply!(reg, put(9, 9=>X))
    measure_outcome=measure_syndrome!(reg, result)
    @test measure_outcome==[1,1,1,1,1,1,1,-1]
    @test syndrome_transform(code, measure_outcome) == Mod2[0,0,0,0,0,1,0,1]

    # test Z error
    reg = join(rand_state(1), zero_state(8))
    apply!(reg, qc)
    apply!(reg, put(9, 3=>Z))
    measure_outcome=measure_syndrome!(reg, result)
    @test measure_outcome==[1,1,-1,1,1,1,1,1]
    @test syndrome_transform(code, measure_outcome) == Mod2[0,0,1,0,0,0,0,0]
end

@testset "generate_syndrome_dict" begin
    result=stabilizers(SurfaceCode{3,3}())
    code = TensorQEC.stabilizers2bimatrix(result)
    TensorQEC.gaussian_elimination!(code)
    transformed_error = Mod2[0,1,0,0,0,0,0,0]
    @test TensorQEC.generate_syndrome_dict(code, transformed_error) == Dict([code.ordering[i] =>transformed_error[i].x for i in 1:8])
end

@testset "most_probable_config" begin
	qc = chain(put(3,1=>X),put(3,2=>X))
    cl = clifford_network(qc)
    p=fill([0.5,0.5,0,0],3)
    syn=Dict([1=>true, 2=>false])
    pinf=syndrome_inference(cl, syn, p)
    @show pinf
    @test length(pinf) == 3
end

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
    st = stabilizers(SurfaceCode{3,3}())
    qc, data_qubits, code = TensorQEC.encode_stabilizers(st)

    reg = join(rand_state(1), zero_state(8))
    regcopy = copy(reg)
    apply!(reg, qc)
    apply!(reg, put(9, 4=>X))

    measure_outcome=measure_syndrome!(reg, st)
    @show measure_outcome
    syn_dict=TensorQEC.generate_syndrome_dict(code,syndrome_transform(code, measure_outcome))

    cl = clifford_network(qc)
    p = fill([0.85,0.05,0.05,0.05],qubit_num)
    pinf = syndrome_inference(cl, syn_dict, p)

    ps_ec_phy = TensorQEC.pauli_string_map_iter(correction_pauli_string(qubit_num, syn_dict, pinf), qc)
    @show ps_ec_phy
    apply!(reg, ps_ec_phy)

    @test measure_syndrome!(reg, st) == [1,1,1,1,1,1,1,1]
    apply!(reg, qc')
    @test fidelity(density_matrix(reg, [9]),density_matrix(regcopy, [9])) ≈ 1.0
end


@testset "syndrome inference and error correct for [[7,1,3]]" begin
    qubit_num = 7
    st = stabilizers(SteaneCode())
    qc, data_qubits, code = encode_stabilizers(st)
    reg = join(zero_state(1),rand_state(1), zero_state(5))
    regcopy = copy(reg)
    apply!(reg, qc)
    apply!(reg, put(qubit_num, 4=>X))

    measure_outcome=measure_syndrome!(reg, st)
    syn_dict=TensorQEC.generate_syndrome_dict(code,syndrome_transform(code, measure_outcome))

    cl = clifford_network(qc)
    p = fill([0.85,0.05,0.05,0.05],qubit_num)
    pinf = syndrome_inference(cl, syn_dict, p)

    ps_ec_phy = TensorQEC.pauli_string_map_iter(correction_pauli_string(qubit_num, syn_dict, pinf), qc)
    @show ps_ec_phy
    apply!(reg, ps_ec_phy)

    @test measure_syndrome!(reg, st) == [1,1,1,1,1,1]
    apply!(reg, qc')
    @test fidelity(density_matrix(reg, [qubit_num]),density_matrix(regcopy, [qubit_num])) ≈ 1.0
end

@testset "inference!" begin
    qubit_num = 7
    st = stabilizers(SteaneCode())
    qc, data_qubits, code = encode_stabilizers(st)
    reg = join(zero_state(1),rand_state(1), zero_state(5))
    regcopy = copy(reg)
    apply!(reg, qc)
    apply!(reg, put(qubit_num, 4=>X))

    p = fill([0.85,0.05,0.05,0.05],qubit_num)
    ps_ec_phy = inference!(reg, code, st, qc, p)
    @show ps_ec_phy
    apply!(reg, ps_ec_phy)

    @test measure_syndrome!(reg, st) == [1,1,1,1,1,1]
    apply!(reg, qc')
    @test fidelity(density_matrix(reg, [qubit_num]),density_matrix(regcopy, [qubit_num])) ≈ 1.0
end
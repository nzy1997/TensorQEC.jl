using Test, TensorQEC, TensorQEC.Yao
@testset "measure_syndrome!" begin
    result=stabilizers(SurfaceCode{3}())
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
    @test measure_outcome==[1,-1,1,1,1,1,1,1]
    @test syndrome_transform(code, measure_outcome) == Mod2[0,1,0,0,0,0,0,0]
end

@testset "generate_syndrome_dict" begin
    result=stabilizers(SurfaceCode{3}())
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
    @test length(pinf) == 2
end

# simple example: Z1Z2 Z2Z3 stabilizers on 3 qubits
@testset "simple_example_inference" begin
    stabilizers = [PauliString((1,4,4)),PauliString((4,4,1))]
    qc, data_qubits, code = TensorQEC.encode_stabilizers(stabilizers)
    reg = join(rand_state(1), zero_state(2))
    apply!(reg, put(3, 3=>X))
    @show reg.state
    regcopy = copy(reg)
    @show qc 
    @show data_qubits
    @show code.matrix

    apply!(reg, qc)
    @show reg.state
    #apply!(reg, put(3, 2=>Y))
    apply!(reg, put(3, 3=>X))
    measure_outcome=measure_syndrome!(reg, stabilizers)
    @show measure_outcome
    syn=syndrome_transform(code, measure_outcome)
    syn_dict=TensorQEC.generate_syndrome_dict(code,syn)

    cl = clifford_network(qc)
    p = fill([0.85,0.05,0.05,0.05],3)
    pinf = syndrome_inference(cl, syn_dict, p)
    @show pinf
    ps_ec = correction_pauli_string(3, syn_dict, pinf)
    ps_ec_phy = TensorQEC.pauli_string_map_iter(ps_ec, qc)
    @show ps_ec,ps_ec_phy
    @show reg.state
    apply!(reg, Yao.YaoBlocks.Optimise.to_basictypes(ps_ec_phy))
    @show reg.state
    @test measure_syndrome!(reg, stabilizers) == [1,1]
    @test fidelity(density_matrix(reg, [3]), density_matrix(regcopy, [3])) ≈ 1.0
end

@testset "syndrome inference for SurfaceCode3" begin
    result = stabilizers(SurfaceCode{3}())
    qc, data_qubits, code = TensorQEC.encode_stabilizers(result)

    reg = join(zero_state(8),rand_state(1))
    regcopy = copy(reg)
    apply!(reg, qc)
    apply!(reg, put(9, 5=>Y))
    apply!(reg, put(9, 3=>Z))
    measure_outcome=measure_syndrome!(reg, result)
    @show measure_outcome
    syn=syndrome_transform(code, measure_outcome)
    syn_dict=TensorQEC.generate_syndrome_dict(code,syn)

    cl = clifford_network(qc)
    p = fill([0.94,0.02,0.02,0.02],9)
    pinf = syndrome_inference(cl, syn_dict, p)
    @show pinf
    ps_ec = correction_pauli_string(9, syn_dict, pinf)
    ps_ec_phy = TensorQEC.pauli_string_map_iter(ps_ec, qc)
    @show 
    apply!(reg, Yao.YaoBlocks.Optimise.to_basictypes(ps_ec_phy))
    @test measure_syndrome!(reg, result) == [1,1,1,1,1,1,1,1]
    apply!(reg, qc')
    #fidelity
    @test fidelity(density_matrix(reg, [9]),density_matrix(regcopy, [9])) ≈ 1.0
end
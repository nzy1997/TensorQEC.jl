using Test, TensorQEC, TensorQEC.Yao
@testset "measure_syndrome!" begin
    result=stabilizers(SurfaceCode{3}())
    code = TensorQEC.stabilizers2bimatrix(result)
    TensorQEC.gaussian_elimination!(code)
    st = TensorQEC.bimatrix2stabilizers(code)
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

@testset "syndrome inference for SurfaceCode3" begin
    result=stabilizers(SurfaceCode{3}())
    code = TensorQEC.stabilizers2bimatrix(result)
    TensorQEC.gaussian_elimination!(code)
    st = TensorQEC.bimatrix2stabilizers(code)
    qc = TensorQEC.encode_circuit(code)

    reg = join(rand_state(1), zero_state(8))
    regcopy=copy(reg)
    apply!(reg, qc)
    apply!(reg, put(9, 9=>X))
    measure_outcome=measure_syndrome!(reg, result)
    syn=syndrome_transform(code, measure_outcome)
    syn_dict=TensorQEC.generate_syndrome_dict(code,syn)

    cl = clifford_network(qc)
    p=fill([0.85,0.05,0.05,0.05],9)
    pinf=syndrome_inference(cl, syn_dict, p)
    ps_ec=correction_pauli_string(9, syn_dict, pinf)
    ps_ec_phy=TensorQEC.pauli_string_map_iter(ps_ec, qc)
    apply!(reg, Yao.YaoBlocks.Optimise.to_basictypes(ps_ec_phy))
    @test measure_syndrome!(reg, result)==[1,1,1,1,1,1,1,1]

end
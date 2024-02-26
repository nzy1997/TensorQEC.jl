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
    @test length(pinf) == 3
end


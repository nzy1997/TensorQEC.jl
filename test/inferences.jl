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

@testset "most_probable_config" begin
	qc = chain(put(3,1=>X),put(3,2=>X))
    cl = clifford_network(qc)
    p=fill([0.5,0.5,0,0],3)
    syn=Dict([1=>true, 2=>false])
    pinf=syndrome_inference(cl, syn, p)
    @show pinf
    @test length(pinf) == 2
end

# @testset "syndrome_inference" begin
# 	qc = QuantumCircuit(3, [Gate(mat(ComplexF64,I4), [1, 2])])
# 	p=Float64[0,0.3,0.6,0]
# 	syn=fill(1,3)
# 	syn_inf=syndrome_inference(qc,syn,fill(p,qc.n_qubits))
# 	@test syn_inf == [0,0,0]
# end
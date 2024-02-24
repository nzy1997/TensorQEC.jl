using Test, TensorQEC, TensorQEC.Yao

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
using Test, TensorQEC, TensorInference.OMEinsum, Yao, TensorInference

@testset "matrix2factor" begin
	g = cnot(2, 1, 2)
	f = TensorQEC.matrix2factor(g, [1, 2], [3, 4])
	@test f.vars == (1, 2, 3, 4)
end

@testset "projector" begin
	@test TensorQEC.projector(Bool[1,0,0,1]) == Bool[1 0 0 0; 0 0 0 1]
	@test TensorQEC.projector(Bool[0,1,1,0]) == Bool[0 1 0 0; 0 0 1 0]
end

@testset "convert to put" begin
    for g in [
        control(5, 3, 4=>X),
        control(5, 4, 2=>X),
        put(5, 2=>X),
        control(5, -3, 2=>X)
        ]
        pg = TensorQEC.convert_to_put(g)
        @test pg isa PutBlock
        @test mat(pg) ≈ mat(g)
    end
end

@testset "tensor network mapping - 1 gate" begin
	# create a circuit and convert it to the pauli basis
	yaoqc = chain(cnot(2, 1, 2))
	yaoqc2 = chain(cnot(2, 1, 2))
	yaopauli = pauli_mapping(mat(ComplexF64, yaoqc))

	# tensor network mapping of a quantum circuit
	for ci in CartesianIndices((fill(4, 2)...,))
		ps = [Yao.BitBasis._onehot(Float64, 4, ci.I[i]) for i in 1:2]
		tn = circuit2tensornetworks(yaoqc2, ps)
		p1 = probability(tn)
		p2 = yaopauli[ci.I..., :, :]
		@test p1 ≈ p2
	end
end


# |ψ> ---.------- p
#        |
# |0> ---X---.--- p
#            |
# |0> -------X--- p
@testset "tensor network mapping" begin
	# create a circuit and convert it to the pauli basis
	yaoqc = chain(cnot(3, 3, 2),cnot(3, 2, 1))
	yaoqc2 = chain(cnot(3, 3, 2),cnot(3, 2, 1))
	yaopauli = pauli_mapping(mat(ComplexF64, yaoqc))

	# tensor network mapping of a quantum circuit
	for ci in CartesianIndices((fill(4, 3)...,))
		ps = [Yao.BitBasis._onehot(Float64, 4, ci.I[i]) for i in 1:3]
		tn = circuit2tensornetworks(yaoqc2, ps)
		p1 = probability(tn)
		p2 = yaopauli[ci.I..., :, :, :]
		@test p1 ≈ p2
	end
end

# @testset "most_probable_config" begin
# 	qc = QuantumCircuit(3, [Gate(CNOT, [1, 2]), Gate(CNOT, [2, 3])])
# 	p=Float64[1,0,0,0]
# 	syn=[0,0,3]
# 	tn = _circuit2tensornetworks(qc, fill(p, qc.n_qubits); syn=syn)
# 	cfg = probability(tn)
# 	@test cfg==[1,0,0,0]
# end

# @testset "syndrome_inference" begin
# 	qc = QuantumCircuit(3, [Gate(mat(ComplexF64,I4), [1, 2])])
# 	p=Float64[0,0.3,0.6,0]
# 	syn=fill(1,3)
# 	syn_inf=syndrome_inference(qc,syn,fill(p,qc.n_qubits))
# 	@test syn_inf == [0,0,0]
# end
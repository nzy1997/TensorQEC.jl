using Test, TensorQEC

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
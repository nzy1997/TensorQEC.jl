using Test, TensorQEC, TensorQEC.Yao

@testset "measure_circuit_fault_tol" begin
    t = TensorQEC.ToricCode(3, 3)
	st = TensorQEC.stabilizers(t)
    qcm,st_pos, num_qubits = measure_circuit_fault_tol(st)
    # display(vizcircuit(qc))
end

@testset "measure_circuit_steane" begin
    t = TensorQEC.SteaneCode()
    st = stabilizers(t)
    qc, data_qubits, code = encode_stabilizers(st) 
    qcm ,st_pos, num_qubits = measure_circuit_steane(qc,data_qubits[1],st,3)
    # display(vizcircuit(qcm))
    @show st_pos
    @test num_qubits == 27
end
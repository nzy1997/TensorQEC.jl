using Test, TensorQEC, TensorQEC.Yao

@testset "measure_circuit_fault_tol" begin
    t = ToricCode(3, 3)
	st = stabilizers(t)
    qcm,st_pos= measure_circuit_fault_tol(st)
    # display(vizcircuit(qc))
end

@testset "measure_circuit_steane" begin
    t = SteaneCode()
    st = stabilizers(t)
    qc, data_qubits, code = encode_stabilizers(st) 
    qcm ,st_pos  = measure_circuit_steane(data_qubits[1],st;qcen = qc)
    # display(vizcircuit(qcm))
    @show st_pos
    @test nqubits(qcm) == 27
end
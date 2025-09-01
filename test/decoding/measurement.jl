using Test, TensorQEC, TensorQEC.Yao

@testset "measure_circuit_fault_tol" begin
    t = ToricCode(3, 3)
	st = stabilizers(t)
    qcm,st_pos= measure_circuit_fault_tol(st)
    # display(vizcircuit(qcm))
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

@testset "measurement_circuit" begin
    st = stabilizers(SurfaceCode(3,3)) 
    qcm ,st_pos = measurement_circuit(st)
    display(vizcircuit(qcm))
end

@testset "generate_measurement_circuit_with_errors" begin
    st = stabilizers(SurfaceCode(3,3)) 
    qce = TensorQEC.generate_measurement_circuit_with_errors(st, 2; before_round_data_depolarization = 0.01, after_clifford_depolarization=0.02, after_reset_flip_probability=0.03, before_measure_flip_probability=0.04)
    # display(vizcircuit(qce))
end

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

@testset "make_measurement_circuit" begin
    mci = TensorQEC.MeasurementCircuitInfo([1,2,3,4],[5],[6], [Dict(5=>1), Dict(5=>2), Dict(5=>3),Dict(5=>4)], [Dict(6=>4), Dict(6=>3), Dict(6=>2),Dict(6=>1)])
    qc = TensorQEC.make_measurement_circuit(mci)
    qc2 = chain(6)
    push!(qc2, put(6, 5 => H))
    push!(qc2, control(6, 5, 1 => X))
    push!(qc2, control(6, 4, 6 => X))

    push!(qc2, control(6, 5, 2 => X))
    push!(qc2, control(6, 3, 6 => X))

    push!(qc2, control(6, 5, 3 => X))
    push!(qc2, control(6, 2, 6 => X))

    push!(qc2, control(6, 5, 4 => X))
    push!(qc2, control(6, 1, 6 => X))
    push!(qc2, put(6, 5 => H))
    @test qc == qc2
end

@testset "generate_measurement_circuit_info" begin
    mci = TensorQEC.generate_measurement_circuit_info(SurfaceCode(3,3))
    @test mci.qubit_pos == [1,2,3,4,5,6,7,8,9]
    @test mci.xstabilizer_pos == [10,11,12,13]
    @test mci.zstabilizer_pos == [14,15,16,17]

    xdict1 = Dict(10=>1, 11=>5, 12=>3)
    @test mci.xmeasure_list[1] == xdict1

    xdict2 = Dict(10=>4, 11=>8, 12=>6)
    @test mci.xmeasure_list[2] == xdict2

    xdict3 = Dict(10=>2, 11=>6, 13=>4)
    @test mci.xmeasure_list[3] == xdict3

    xdict4 = Dict(10=>5, 11=>9, 13=>7)
    @test mci.xmeasure_list[4] == xdict4

    zdict1 = Dict(14=>2, 15=>4, 17=>8)
    @test mci.zmeasure_list[1] == zdict1

    zdict2 = Dict(14=>3, 15=>5, 17=>9)
    @test mci.zmeasure_list[2] == zdict2

    zdict3 = Dict(14=>5, 15=>7, 16=>1)
    @test mci.zmeasure_list[3] == zdict3

    zdict4 = Dict(14=>6, 15=>8, 16=>2)
    @test mci.zmeasure_list[4] == zdict4

    @test mci.H_before_list == [10,11,12,13]
    @test mci.H_after_list == [10,11,12,13]
end
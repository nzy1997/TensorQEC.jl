using Test, TensorQEC, TensorQEC.Yao, YaoPlots


@testset "make table and save" begin
    st =stabilizers(SurfaceCode{3}())
    bimat=TensorQEC.stabilizers2bimatrix(st)
    table = make_table(bimat.matrix, 1)
    @test length(table) == 14
    save_table(table, "test_table.txt")
    table2 = read_table("test_table.txt")
    @test table == table2
end

@testset "errorcorrect_circuit" begin
    st =stabilizers(SurfaceCode{3}())
    bimat=TensorQEC.stabilizers2bimatrix(st)
    table = make_table(bimat.matrix, 1)
    qc1,st_pos,num_qubits = measure_circuit_fault_tol(st)
    num_st=8
    @show num_qubits
    @show st_pos
    qc2 = correct_circuit(table, st_pos, num_qubits,num_st,9)
    display(vizcircuit(qc1))
    display(vizcircuit(qc2))
end
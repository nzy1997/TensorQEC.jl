using Test, TensorQEC, TensorQEC.Yao


@testset "make table and save" begin
    st =stabilizers(SurfaceCode(3,3))
    table = make_table(st, 1)
    @test length(table) == 14
    save_table(table, "test_table.txt")
    table2 = load_table("test_table.txt")
    @test table == table2
    rm("test_table.txt")
end

@testset "errorcorrect_circuit" begin
    t = ToricCode(3, 3)
	st = stabilizers(t)
    table = make_table(st, 1)
    qc1,st_pos,num_qubits = measure_circuit_fault_tol(st)
    num_st=16
    @show num_qubits
    @show st_pos
    qc2 = correct_circuit(table, st_pos, num_qubits,num_st,18)
    # display(vizcircuit(qc1))
    # display(vizcircuit(qc2))
end

@testset "show table" begin
    t = SurfaceCode(3, 3)
    st = stabilizers(t)
    table = make_table(st, 1)
    show_table(table, 9, 8)
end
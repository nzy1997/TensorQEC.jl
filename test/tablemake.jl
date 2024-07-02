using Test, TensorQEC, TensorQEC.Yao


@testset "make table and save" begin
    st =stabilizers(SurfaceCode(3,3))
    table = make_table(st, 1)
    @test length(table.table) == 23
    save_table(table, "test_table.txt")
    table2 = load_table("test_table.txt", 9, 8, 1)
    @test table.table == table2.table
    @test table.num_qubits == table2.num_qubits
    @test table.num_st == table2.num_st
    @test table.d == table2.d
    rm("test_table.txt")
end

@testset "errorcorrect_circuit" begin
    t = ToricCode(3, 3)
	st = stabilizers(t)
    table = make_table(st, 1;y_error=false)
    qc1,st_pos,num_qubits = measure_circuit_fault_tol(st)
    qc2 = correct_circuit(table, st_pos,num_qubits)
    chain(qc1, qc2)
end

@testset "show table" begin
    t = SurfaceCode(3, 3)
    st = stabilizers(t)
    table = make_table(st, 1)
    @show table
end
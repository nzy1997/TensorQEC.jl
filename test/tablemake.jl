using Test, TensorQEC, TensorQEC.Yao
using TensorQEC: stabilizers2bimatrix

@testset "make table" begin
    st =stabilizers(SurfaceCode(3,3))
    for (error_type,l) in [("XZ",14), ("Z",7), ("X",7),("all",23)]
        table = make_table(st, 1;error_type = error_type)
        @test length(table.table) == l
    end
end

@testset "save table and load" begin
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
    table = make_table(st, 1;error_type = "XZ")
    qc1,st_pos = measure_circuit_fault_tol(st)
    qc2 = correct_circuit(table, st_pos,nqubits(qc1))
    chain(qc1, qc2)
end

@testset "show table" begin
    t = SurfaceCode(3, 3)
    st = stabilizers(t)
    table = make_table(st, 1)
    @show table
end

@testset "table inference" begin
    t = SurfaceCode(3, 3)
    st = stabilizers(t)
    table = make_table(st, 1)
    measure_outcome = measure_syndrome!(rand_state(9), st)
    @test table_inference(table, [-1,1,1,1,1,1])[1] == (2=>Z)
    @test table_inference(table, [-1,-1,-1,-1,1,1]) === nothing
end
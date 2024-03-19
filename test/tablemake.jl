using Test, TensorQEC, TensorQEC.Yao


@testset "make_table" begin
    st =stabilizers(SurfaceCode{3}())
    bimat=TensorQEC.stabilizers2bimatrix(st)
    table = make_table(bimat.matrix, 1)
    @test length(table) == 23
    save_table(table, "test_table.txt")
    table2 = read_table("test_table.txt")
    @test table == table2
end
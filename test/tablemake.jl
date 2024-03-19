using Test, TensorQEC, TensorQEC.Yao


@testset "make_table" begin
    st =stabilizers(SurfaceCode{3}())
    bimat=TensorQEC.stabilizers2bimatrix(st)
    table = make_table(bimat.matrix, 2)

end
using Test,TensorQEC


@testset "toric_code" begin
    code = toric_code(2)
    @test code.xcodenum == 3
    @test code.zcodenum == 3
    @test code.matrix == [
        1  0  1  0  1  1  0  0  0  0  0  0  0  0  0  0;
        1  0  1  0  0  0  1  1  0  0  0  0  0  0  0  0;
        0  1  0  1  1  1  0  0  0  0  0  0  0  0  0  0;
        0  0  0  0  0  0  0  0  1  1  0  0  1  0  1  0;
        0  0  0  0  0  0  0  0  0  0  1  1  1  0  1  0;
        0  0  0  0  0  0  0  0  1  1  0  0  0  1  0  1;
    ]
end

@testset "guassian_elimination!" begin
    code = toric_code(2)
    guassian_elimination!(code)
end
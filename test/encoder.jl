using Test,TensorQEC, TensorQEC.Yao, TensorQEC.LinearAlgebra

@testset "toric code" begin
    t = TensorQEC.ToricCode(2, 3)
    result = TensorQEC.stabilizers(t)
    expected_result = PauliString.([(2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1), (1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1), (1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1), (1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1), (2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2), (4, 4, 1, 1, 1, 1, 4, 1, 1, 1, 4, 1), (4, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 4), (1, 1, 4, 4, 1, 1, 4, 1, 4, 1, 1, 1), (1, 1, 4, 4, 1, 1, 1, 4, 1, 4, 1, 1), (1, 1, 1, 1, 4, 4, 1, 1, 4, 1, 4, 1)])
    @test res == expected_result
end

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
    code = toric_code(3)
    guassian_elimination!(code)
    @test code.matrix[1:8, 1:8] == [
        1  0  0  0  0  0  0  0;
        0  1  0  0  0  0  0  0;
        0  0  1  0  0  0  0  0;
        0  0  0  1  0  0  0  0;
        0  0  0  0  1  0  0  0;
        0  0  0  0  0  1  0  0;
        0  0  0  0  0  0  1  0;
        0  0  0  0  0  0  0  1;
    ]
    @test code.matrix[9:16, 27:34] == [
        1  0  0  0  0  0  0  0;
        0  1  0  0  0  0  0  0;
        0  0  1  0  0  0  0  0;
        0  0  0  1  0  0  0  0;
        0  0  0  0  1  0  0  0;
        0  0  0  0  0  1  0  0;
        0  0  0  0  0  0  1  0;
        0  0  0  0  0  0  0  1;
    ]
end
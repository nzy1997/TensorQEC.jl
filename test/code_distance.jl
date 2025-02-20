using Test
using TensorQEC
using TensorQEC: row_echelon_form

@testset "classical_code_distance" begin
    H = [0 0 0 1 1 1 1;0 1 1 0 0 1 1; 1 0 1 0 1 0 1]
    @test code_distance(H) == 3

    H2 = (H .== 1)
    @test code_distance(H2) == 3

    H3 = Mod2.(H2)
    @test code_distance(H3) == 3
end

@testset "row_echelon_form" begin
    H = Bool[0 0 0 1 1 1 1;0 1 1 1 1 1 1; 1 0 1 0 1 0 1]
    row_echelon_form(H)
end
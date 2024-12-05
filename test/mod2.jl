using TensorQEC, Test

@testset "mod2" begin
    a = Mod2(false)
    b = Mod2(true)
    @test a + b == Mod2(true)
    @test a + a == Mod2(false)
    @test a * b == Mod2(false)
    @test a * a == Mod2(false)
    @test b * b == Mod2(true)
    # test - and minus
    @test -a == Mod2(false)
    @test -b == Mod2(true)
    @test a - b == Mod2(true)
    @test b - a == Mod2(true)
    # test zero and one
    @test zero(a) == Mod2(false)
    @test one(a) == Mod2(true)
    @test iszero(a) == true
end
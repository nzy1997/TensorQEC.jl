using TensorQEC, Test

@testset "mod2" begin
    a = Mod2(false)
    b = Mod2(true)
    @test string(a) == "0₂"
    @test string(b) == "1₂"
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

@testset "bitmul!" begin
    A = Mod2.(rand(Bool, 1000, 1000))
    B = Mod2.(rand(Bool, 1000, 1000))
    C = Mod2.(zeros(Bool, 1000, 1000))
    res1 = TensorQEC.bitmul!(C, A, B)
    res2 = A * B
    @test res1 == res2
end
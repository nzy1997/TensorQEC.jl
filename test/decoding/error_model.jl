using Test
using TensorQEC
using Random

@testset "IndependentFlipError" begin
    em = iid_error(0.1,10)
    @test random_error_qubits(em) isa Vector{Mod2}
end

@testset "IndependentDepolarizingError" begin
    em = iid_error(0.05,0.06,0.1,100000)
    ep = random_error_qubits(em)
    ex = ep.xerror
    ez = ep.zerror
    @test ex isa Vector{Mod2}
    @test ez isa Vector{Mod2}
    @test count(v->v.x,ex)/100000 ≈ 0.11 atol=0.05
    @test count(v->v.x,ez)/100000 ≈ 0.16 atol=0.05
end

@testset "check_logical_error" begin
    st = stabilizers(SurfaceCode(3, 3))
    tannerxz = CSSTannerGraph(st)
    lx,lz = TensorQEC.logical_operator(tannerxz)

    @test check_logical_error(Mod2[0,0,0,0,0,0,0,0,0],Mod2[1,1,1,0,0,0,0,0,0],lz)
    @test !check_logical_error(Mod2[0,0,0,0,0,0,0,0,0],Mod2[1,1,0,1,1,0,0,0,0],lz)

    cssep1 = TensorQEC.CSSErrorPattern(Mod2[0,0,0,0,0,0,0,0,0],Mod2[0,0,0,0,0,0,0,0,0])
    cssep2 = TensorQEC.CSSErrorPattern(Mod2[0,0,0,0,0,0,0,0,0],Mod2[0,0,0,0,0,0,0,0,0])

    @test !check_logical_error(cssep1, cssep2, lx, lz)
end
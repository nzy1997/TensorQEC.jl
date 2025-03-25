using Test
using TensorQEC
using TensorQEC.Graphs.Experimental: has_isomorph
using Random
using TensorQEC.LinearAlgebra
@testset "SimpleTannerGraph" begin
    sts = [[1, 2,3,4],[2,3,4,5]]
    nq = 5
    tanner = SimpleTannerGraph(nq, sts)
    @test tanner.q2s == [[1],[1,2],[1,2],[1,2],[2]]
    @test tanner.s2q == [[1,2,3,4],[2,3,4,5]]
    @test tanner.ns == 2
    @test tanner.H == Mod2[1 1 1 1 0; 0 1 1 1 1]
    @test count(x->x.x, tanner.H) == sum(length.(tanner.q2s))
    @test sum(length.(tanner.q2s)) == sum(length.(tanner.s2q))
end

@testset "dual_graph" begin 
    sts = [[1, 2],[2,3],[3,1]]
    nq = 3
    tanner = SimpleTannerGraph(nq, sts)
    dual = dual_graph(tanner)
    @test dual.q2s == sts
end

@testset "syndrome_extraction" begin
    sts = [[1, 2,3,4],[2,3,4,5]]
    nq = 5
    tanner = SimpleTannerGraph(nq, sts)
    error_qubits = Mod2[1,0,1,1,0]
    @test syndrome_extraction(error_qubits, tanner) == Mod2[1,0]

    Random.seed!(123)
    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    em = DepolarizingError(0.05, 0.06, 0.1)
    ep = random_error_qubits(9, em)
    syn = syndrome_extraction(ep,tanner)
    @test syn.sx == Mod2[1,0,0,0]
    @test syn.sz == Mod2[0,1,0,0]
end

@testset "product_graph" begin
    sts1 = [[1, 2],[2,3],[3,1]]
    nq1 = 3
    tanner1 = SimpleTannerGraph(nq1, sts1)
    sts2 = [[1, 2],[2,3],[3,1]]
    nq2 = 3 
    tanner2 = SimpleTannerGraph(nq2, sts2)
    
    pgraph = product_graph(tanner1, tanner2)

    st_toric = stabilizers(ToricCode(3,3);linearly_independent = false)
    tanner_toric = CSSTannerGraph(st_toric)

    g1 = get_graph(pgraph)
    g2 = get_graph(tanner_toric)
    @test has_isomorph(g1, g2)
end

@testset "random_ldpc" begin
    Random.seed!(123)
    r34ldpc = random_ldpc(3,4,6)
end

@testset "mod2matrix_inverse" begin
    H = Bool[1 0 0; 0 0 1; 0 1 0]
    q = mod2matrix_inverse(H)
    @test q * H == Matrix{Bool}(I,3,3)

    H = Bool[1 1 1; 0 1 1; 0 0 1]
    q = mod2matrix_inverse(H)
    @test q * H == Matrix{Bool}(I,3,3)
end


@testset "check_logical_error" begin
    st = stabilizers(SurfaceCode(3, 3))
    tannerxz = CSSTannerGraph(st)
    tannerz = tannerxz.stgz
    tannerx = tannerxz.stgx
    @test !TensorQEC.check_logical_error(Mod2[0,0,0,0,0,0,0,0,0],Mod2[1,1,1,0,0,0,0,0,0],tannerx.H)
    @test TensorQEC.check_logical_error(Mod2[0,0,0,0,0,0,0,0,0],Mod2[1,1,0,1,1,0,0,0,0],tannerx.H)
end
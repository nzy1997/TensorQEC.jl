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

@testset "sydrome_extraction" begin
    sts = [[1, 2,3,4],[2,3,4,5]]
    nq = 5
    tanner = SimpleTannerGraph(nq, sts)
    error_qubits = Mod2[1,0,1,1,0]
    @test sydrome_extraction(error_qubits, tanner) == Mod2[1,0]
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

@testset "belief_propagation1" begin
    sts = [[1, 2,3,4],[2,3,5,7],[3,4,5,6]]
    nq = 7
    tanner = SimpleTannerGraph(nq, sts)
    error_qubits = Mod2[1,0,0,0,0,0,0]
    syd = sydrome_extraction(error_qubits, tanner)
    bpres = belief_propagation(syd, tanner, 0.05;max_iter=100)
    @test bpres.error_qubits == error_qubits
end

@testset "belief_propagation2" begin
    Random.seed!(245)
    r34ldpc = random_ldpc(4,3,120)
    # plot_graph(r34ldpc)
    em = FlipError(0.1)
    error_qubits =  random_error_qubits(120, em)
    syd = sydrome_extraction(error_qubits, r34ldpc)
    bpres = belief_propagation(syd, r34ldpc, 0.05;max_iter=100)

    @test syd == sydrome_extraction(bpres.error_qubits, r34ldpc)
    # @test check_decode(error_qubits,bp_error,r34ldpc)
end 

@testset "belief_propagation3" begin
    Random.seed!(245)
    r34ldpc = random_ldpc(4,3,120)
    em = FlipError(0.3)
    error_qubits =  random_error_qubits(120, em)
    syd = sydrome_extraction(error_qubits, r34ldpc)
    bpres = belief_propagation(syd, r34ldpc, 0.3;max_iter=200)
    osd_error = osd(r34ldpc, bpres.error_perm,syd)
    
    @test check_decode(error_qubits,syd,r34ldpc.H)
    # @test check_decode(error_qubits,osd_error,r34ldpc)
    # @test syd == sydrome_extraction(bp_error, r34ldpc)
    # @show check_decode(error_qubits,bp_error,r34ldpc)
end

@testset "message_list" begin
    sts = [[1, 2,3,4],[2,3,5,7],[3,4,5,6]]
    nq = 7
    tanner = SimpleTannerGraph(nq, sts)
    mq2s =[[i for i in v] for v in tanner.q2s]
    @test TensorQEC.message_list(mq2s,2,tanner.q2s,tanner.s2q;exampt_qubit = 2) == [2,2,2]
    @test TensorQEC.message_list(mq2s,2,tanner.q2s,tanner.s2q;exampt_qubit = 1) == [2,2,2,2]
end

@testset "check_decode" begin
    sts = [[1, 2,3,4],[2,3,5,7],[3,4,5,6]]
    nq = 7
    tanner = SimpleTannerGraph(nq, sts)
    error_qubits1 = Mod2[1,0,0,0,0,0,0]
    syd1 = sydrome_extraction(error_qubits1, tanner)
    error_qubits2 = Mod2[0,1,1,1,0,0,0]
    syd2 = sydrome_extraction(error_qubits2, tanner)
    error_qubits3 = Mod2[0,0,0,1,0,1,0]
    syd3 = sydrome_extraction(error_qubits3, tanner)
    @test check_decode(error_qubits1,syd2,tanner.H) == true
    @test check_decode(error_qubits1,syd3,tanner.H) == true
end

@testset "check_linear_indepent" begin
    H = Bool[1 1 1 0; 0 1 1 1]
    @test check_linear_indepent(H) == true
    H = Bool[1 1 1 0; 0 1 1 0; 0 0 0 1; 1 0 0 0]
    @test check_linear_indepent(H) == false
    H = Mod2[1 1 1 0; 0 1 1 1]
    @test check_linear_indepent(H) == true
    H = Mod2[1 1 1 0; 0 1 1 0; 0 0 0 1; 1 0 0 0]
    @test check_linear_indepent(H) == false
end

@testset "tensor_osd" begin
    sts = [[1, 2,3,4],[2,3,5,7],[3,4,5,6]]
    nq = 7
    tanner = SimpleTannerGraph(nq, sts)
    error_qubits = Mod2[0,0,0,1,0,0,0]
    syd = sydrome_extraction(error_qubits, tanner)
    tn_res = tensor_osd(syd,tanner, 0.05)
    @test check_decode(tn_res,syd,tanner)

    Random.seed!(23456)
    r34ldpc = random_ldpc(4,3,32)
    # plot_graph(r34ldpc)
    em = FlipError(0.05)
    error_qubits =  random_error_qubits(32, em)
    syd = sydrome_extraction(error_qubits, r34ldpc)
    error_probabillity = tensor_infer(r34ldpc, 0.05,syd)
    tn_res = tensor_osd(syd,r34ldpc, 0.05)
    @test check_decode(tn_res,syd,r34ldpc)
end

@testset "osd" begin
    sts = [[1, 2,3,4],[2,3,5,7],[3,4,5,6]]
    nq = 7
    tanner = SimpleTannerGraph(nq, sts)

    error_qubits = Mod2[0,0,0,1,0,0,0]
    syd = sydrome_extraction(error_qubits, tanner)
    order = [1, 2, 3, 4, 5, 6, 7]
    osd_error = osd(tanner, order,syd)
    @test check_decode(error_qubits,syd,tanner)
end

@testset "mod2matrix_inverse" begin
    H = Bool[1 0 0; 0 0 1; 0 1 0]
    q = mod2matrix_inverse(H)
    @test q * H == Matrix{Bool}(I,3,3)

    H = Bool[1 1 1; 0 1 1; 0 0 1]
    q = mod2matrix_inverse(H)
    @test q * H == Matrix{Bool}(I,3,3)
end

@testset "bp_osd" begin
    Random.seed!(245)
    r34ldpc = random_ldpc(4,3,120)
    em = FlipError(0.05)
    error_qubits =  random_error_qubits(120, em)
    syd = sydrome_extraction(error_qubits, r34ldpc)
    res = bp_osd(syd, r34ldpc, 0.05;max_iter=100)

    @test check_decode(res,syd,r34ldpc.H)

    em = FlipError(0.3)
    error_qubits =  random_error_qubits(120, em)
    syd = sydrome_extraction(error_qubits, r34ldpc)
    res = bp_osd(syd, r34ldpc, 0.3;max_iter=100)

    @test check_decode(res,syd,r34ldpc.H)
end

@testset "check_logical_error" begin
    st = stabilizers(SurfaceCode(3, 3))
    tannerxz = CSSTannerGraph(st)
    tannerz = tannerxz.stgz
    tannerx = tannerxz.stgx
    @test !TensorQEC.check_logical_error(Mod2[0,0,0,0,0,0,0,0,0],Mod2[1,1,1,0,0,0,0,0,0],tannerx.H)
    @test TensorQEC.check_logical_error(Mod2[0,0,0,0,0,0,0,0,0],Mod2[1,1,0,1,1,0,0,0,0],tannerx.H)
end
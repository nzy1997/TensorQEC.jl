using Test
using TensorQEC
using LuxorGraphPlot.Luxor
using Graphs.Experimental: has_isomorph
using Random
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

@testset "plot_graph" begin
    sts = [[1, 2,3,4],[2,3,4,5]]
    nq = 5
    tanner = SimpleTannerGraph(nq, sts)
    @test plot_graph(tanner) isa Luxor.Drawing
end

@testset "sydrome_extraction" begin
    sts = [[1, 2,3,4],[2,3,4,5]]
    nq = 5
    tanner = SimpleTannerGraph(nq, sts)
    errored_qubits = Mod2[1,0,1,1,0]
    @test sydrome_extraction(errored_qubits, tanner) == Mod2[1,0]
end

@testset "product_graph" begin
    sts1 = [[1, 2],[2,3],[3,1]]
    nq1 = 3
    tanner1 = SimpleTannerGraph(nq1, sts1)
    plot_graph(tanner1)
    sts2 = [[1, 2],[2,3],[3,1]]
    nq2 = 3 
    tanner2 = SimpleTannerGraph(nq2, sts2)
    
    pgraph = product_graph(tanner1, tanner2)
    plot_graph(pgraph)

    st_toric = stabilizers(ToricCode(3,3);linearly_independent = false)
    tanner_toric = CSSTannerGraph(st_toric)
    plot_graph(tanner_toric)

    g1 = get_graph(pgraph)
    g2 = get_graph(tanner_toric)
    @test has_isomorph(g1, g2)
end

@testset "error correct" begin
    sts1 = [[1, 2],[2,3],[3,1]]
    nq1 = 3
    tanner1 = SimpleTannerGraph(nq1, sts1)
    errored_qubits = Mod2[1,0,0]
    synd = sydrome_extraction(errored_qubits, tanner1)
    @show belief_propagation(synd, tanner1, 0.05)
end

@testset "random_ldpc" begin
    Random.seed!(123)
    r34ldpc = random_ldpc(3,4,6)
    plot_graph(r34ldpc)
end

@testset "belief_propagation" begin
    sts = [[1, 2,3,4],[2,3,5,7],[3,4,5,6]]
    nq = 7
    tanner = SimpleTannerGraph(nq, sts)
    errored_qubits = Mod2[1,0,0,0,0,0,0]
    syd = sydrome_extraction(errored_qubits, tanner)
    @test belief_propagation(syd, tanner, 0.05;max_iter=10) == errored_qubits

    Random.seed!(38956783)
    r34ldpc = random_ldpc(3,4,120)
    # plot_graph(r34ldpc)
    errored_qubits = random_errored_qubits(120,0.05)
    syd = sydrome_extraction(errored_qubits, r34ldpc)
    bp_error = belief_propagation(syd, r34ldpc, 0.05;max_iter=100)
    @test syd == sydrome_extraction(bp_error, r34ldpc)
    @test check_decode(errored_qubits,bp_error,r34ldpc)

    errored_qubits = random_errored_qubits(120,0.3)
    syd = sydrome_extraction(errored_qubits, r34ldpc)
    bp_error = belief_propagation(syd, r34ldpc, 0.3;max_iter=200)
    @show bp_error
    # @test syd == sydrome_extraction(bp_error, r34ldpc)
    # @show check_decode(errored_qubits,bp_error,r34ldpc)
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
    errored_qubits1 = Mod2[1,0,0,0,0,0,0]
    errored_qubits2 = Mod2[0,1,1,1,0,0,0]
    errored_qubits3 = Mod2[0,0,0,1,0,1,0]
    @test check_decode(errored_qubits1,errored_qubits2,tanner) == true
    @test check_decode(errored_qubits1,errored_qubits3,tanner) == false
end


using Test
using TensorQEC
using Random

@testset "IPDecoder" begin
    tanner = CSSTannerGraph(SurfaceCode(3,3))
    decoder = IPDecoder()
    error_qubits = Mod2[0,0,0,1,0,0,0,0,0]
    syd = syndrome_extraction(error_qubits, tanner.stgz)

    res = decode(decoder,tanner.stgz,syd)
    @test syd == syndrome_extraction(res.error_qubits, tanner.stgz)

    res = decode(decoder,tanner.stgz,syd)
    @test syd == syndrome_extraction(res.error_qubits, tanner.stgz)

    res = decode(decoder,tanner.stgz,syd,0.02*collect(1:9))
    @test syd == syndrome_extraction(res.error_qubits, tanner.stgz)
    @test res.error_qubits == Mod2[0,0,0,0,0,0,1,0,0]

    res = decode(decoder,tanner.stgz,syd,0.02*collect(9:-1:1))
    @test syd == syndrome_extraction(res.error_qubits, tanner.stgz)
    @test res.error_qubits == Mod2[0,0,0,1,0,0,0,0,0]
end

@testset "IPDecoder" begin
    Random.seed!(245)
    r34ldpc = random_ldpc(4,3,120)
    # plot_graph(r34ldpc)
    em = FlipError(0.1)
    error_qubits =  random_error_qubits(120, em)
    syd = syndrome_extraction(error_qubits, r34ldpc)
    res = decode(IPDecoder(),r34ldpc,syd)

    @test syd == syndrome_extraction(res.error_qubits, r34ldpc)
end

@testset "CSSIPDecoder" begin
    Random.seed!(123)
    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    em = iid_error(0.05, 0.06, 0.1,9)
    ep = random_error_qubits(em)
    syn = syndrome_extraction(ep,tanner)

    res = decode(IPDecoder(),tanner,syn)
    @show res
    @test syn == syndrome_extraction(res.error_qubits, tanner)

    ct = compile(IPDecoder(), tanner)
    res = decode(ct, syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner)
end

@testset "_mixed_integer_programming_for_one_solution" begin
    d = 3
    n = d^2
    tanner = CSSTannerGraph(SurfaceCode(d,d))
    error_qubits = random_error_qubits(n, FlipError(0.2))
    syd = syndrome_extraction(error_qubits, tanner.stgz)
    res = TensorQEC._mixed_integer_programming_for_one_solution(tanner.stgz.H, syd.s)
    @test syd == syndrome_extraction(res, tanner.stgz)

    error_qubits = random_error_qubits(n, DepolarizingError(0.1,0.1,0.1))
    syd = syndrome_extraction(error_qubits, tanner)
    xe,ze = TensorQEC._mixed_integer_programming_for_one_solution(tanner, syd)
    res = TensorQEC.CSSErrorPattern(xe, ze)
    @test syd == syndrome_extraction(res, tanner)
end
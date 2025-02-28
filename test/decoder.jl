using Test
using TensorQEC
using Random

@testset "IPDecoder" begin
    tanner = CSSTannerGraph(SurfaceCode(3,3))
    decoder = IPDecoder()
    error_qubits = Mod2[1,0,0,0,0,0,0,0,0]
    syd = syndrome_extraction(error_qubits, tanner.stgz)

    res = decode(decoder,tanner.stgz,syd)
    @test syd == syndrome_extraction(res.error_qubits, tanner.stgz)
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
    em = DepolarizingError(0.05, 0.06, 0.1)
    ep = random_error_qubits(9, em)
    syn = syndrome_extraction(ep,tanner)

    res = decode(IPDecoder(),tanner,syn)
    @test syn.sx == syndrome_extraction(res.zerror_qubits, tanner.stgx)
    @test syn.sz == syndrome_extraction(res.xerror_qubits, tanner.stgz)
end
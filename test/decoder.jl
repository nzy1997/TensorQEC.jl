using Test
using TensorQEC
using Random

@testset "IPDecoder" begin
    tanner = CSSTannerGraph(SurfaceCode(3,3))
    decoder = IPDecoder()
    error_qubits = Mod2[1,0,0,0,0,0,0,0,0]
    syd = sydrome_extraction(error_qubits, tanner.stgz)

    res = decode(decoder,tanner.stgz,syd)
    @test syd == sydrome_extraction(res.error_qubits, tanner.stgz)
end

@testset "IPDecoder" begin
    Random.seed!(245)
    r34ldpc = random_ldpc(4,3,120)
    # plot_graph(r34ldpc)
    em = FlipError(0.1)
    error_qubits =  random_error_qubits(120, em)
    syd = sydrome_extraction(error_qubits, r34ldpc)
    res = decode(IPDecoder(),r34ldpc,syd)

    @test syd == sydrome_extraction(res.error_qubits, r34ldpc)
end
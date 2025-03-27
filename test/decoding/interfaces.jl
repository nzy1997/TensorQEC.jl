using Test
using TensorQEC
using Random

@testset "decode SimpleDecodingProblem" begin
    tanner = CSSTannerGraph(SurfaceCode(3,3)).stgz
    error_qubits = Mod2[0,0,0,1,0,0,0,0,0]
    syn = syndrome_extraction(error_qubits, tanner)

    for decoder in [IPDecoder(),BPDecoder(100),BPOSD(100),MatchingDecoder(TensorQEC.GreedyMatchingSolver()),TNMAP()]
        res = decode(decoder,tanner,syn)
        @test res isa DecodingResult
    end
end

@testset "decodeCSSDecodingProblem" begin
    Random.seed!(123)
    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    em = DepolarizingError(0.05, 0.06, 0.1)
    ep = random_error_qubits(9, em)
    syn = syndrome_extraction(ep,tanner)

    for decoder in [IPDecoder()]
        res = decode(decoder,tanner,syn)
        @test res isa CSSDecodingResult
    end
end
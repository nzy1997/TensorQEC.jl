using Test
using TensorQEC
using Random

@testset "tanner2fwswg" begin
    Random.seed!(123)
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d, d)).stgx
    em = FlipError(0.1)
    error_qubits =  random_error_qubits(d*d, em)
    syn = syndrome_extraction(error_qubits,tanner)

    fwg = TensorQEC.tanner2fwswg(tanner,[0.1,fill(0.2,d*d-1)...])
    @test fwg.error_path[1,5] == [2]
    @test Set(fwg.error_path[3,4]) == Set([3,7])

    fwg = TensorQEC.tanner2fwswg(tanner,[0.03,0.02,0.01,0.4,0.4,0.4,0.01,0.01,0.01])
    @test fwg.error_path[1,5] == [1]
    @test Set(fwg.error_path[3,4]) == Set([4,5,6])
end

@testset "IPMatchingSolver" begin
    Random.seed!(123)
    d = 7
    tanner = CSSTannerGraph(SurfaceCode(d, d)).stgx
    em = FlipError(0.1)
    error_qubits =  random_error_qubits(d*d, em)
    syn = syndrome_extraction(error_qubits,tanner)

    fwg = TensorQEC.tanner2fwswg(tanner)
    mwb = TensorQEC.FWSWGtoMWB(fwg,syn)

    solver = TensorQEC.IPMatchingSolver()
    ev = TensorQEC.solve_matching(mwb,solver)

    ans = extract_decoding(fwg,ev,d*d)

    @test syn == syndrome_extraction(ans,tanner)
end

@testset "decode" begin
    Random.seed!(3245)
    d = 7
    tanner = CSSTannerGraph(SurfaceCode(d, d)).stgx
    em = FlipError(0.1)
    error_qubits =  random_error_qubits(d*d, em)
    syn = syndrome_extraction(error_qubits,tanner)
    decoder = MatchingDecoder(IPMatchingSolver(),d*d)
    ans = decode(decoder,tanner,syn)
    @test syn == syndrome_extraction(ans,tanner)

    decoder = MatchingDecoder(TensorQEC.GreedyMatchingSolver(),d*d)
    ans = decode(decoder,tanner,syn)
    @test syn == syndrome_extraction(ans,tanner)
end
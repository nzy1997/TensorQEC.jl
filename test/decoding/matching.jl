using Test
using TensorQEC
using Random

@testset "tanner2fwswg" begin
    Random.seed!(123)
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d, d)).stgx
    em = iid_error(0.05,d*d)
    error_qubits =  random_error_qubits(em)
    syn = syndrome_extraction(error_qubits,tanner)

    fwg = TensorQEC.tanner2fwswg(tanner,[0.1,fill(0.2,d*d-1)...])
    @test fwg.error_path[1,5] == [2]
    @test Set(fwg.error_path[3,4]) == Set([3,7])

    fwg = TensorQEC.tanner2fwswg(tanner,[0.03,0.02,0.01,0.4,0.4,0.4,0.01,0.01,0.01])
    @test fwg.error_path[1,5] == [1]
    @test Set(fwg.error_path[3,4]) == Set([4,5,6])
end

@testset "decode" begin
    Random.seed!(3245)
    d = 7
    tanner = CSSTannerGraph(SurfaceCode(d, d)).stgx
    em = iid_error(0.05,d*d)
    error_qubits =  random_error_qubits(em)
    syn = syndrome_extraction(error_qubits,tanner)
    decoder = MatchingDecoder(IPMatchingSolver())
    ans = decode(decoder,tanner,syn)
    @test syn == syndrome_extraction(ans.error_qubits,tanner)

    decoder = MatchingDecoder(TensorQEC.GreedyMatchingSolver())
    ans = decode(decoder,tanner,syn)
    @test syn == syndrome_extraction(ans.error_qubits,tanner)
end

@testset "compile and decode" begin
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d, d))

    ct = compile(MatchingDecoder(IPMatchingSolver()),tanner.stgz)

    Random.seed!(123)
    em = iid_error(0.05,d*d)
    ep = random_error_qubits(em)
    syn = syndrome_extraction(ep,tanner.stgz)
    
    res = decode(ct,syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner.stgz)
end
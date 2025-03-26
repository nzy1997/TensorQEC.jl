using Test
using TensorQEC
using TensorQEC.Yao
using TensorQEC.OMEinsum
using TensorQEC.TensorInference
using Random

@testset "stg2tensornetwork" begin
    sts = [[1, 2,3,4],[2,3,4,5]]
    nq = 5
    tanner = SimpleTannerGraph(nq, sts)

    code = DynamicEinCode([[i] for i in 1:5],Int[])
    tensors= fill([0.1,0.9],5)
    ptn = TensorNetwork(code,tensors)
    tn = TensorQEC.stg2tensornetwork(tanner,ptn)
    tn = TensorNetworkModel(tn, evidence=Dict(6=>0,7=>1))
    @show most_probable_config(tn)
end

@testset "decode" begin
    Random.seed!(123)
    d = 13
    tanner = CSSTannerGraph(SurfaceCode(d, d))
    em = FlipError(0.05)
    ep = random_error_qubits(d*d, em)
    syn = syndrome_extraction(ep,tanner.stgz)

    decoder = TNMAP()
    res = decode(decoder,tanner.stgz,syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner.stgz)
end
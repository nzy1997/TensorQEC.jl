using Test
using TensorQEC
using Random
using TensorQEC.OMEinsum
using TensorQEC.Yao

@testset "gdp decoding" begin
    Random.seed!(123)
    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    em = iid_error(0.05,0.06,0.1,9)
    ep = random_error_pattern(em)
    syn = syndrome_extraction(ep,tanner)
    gs = TensorQEC.SimpleSyndrome([syn.sx...,syn.sz...])

    gdp,rgdp = reduce2general(tanner,em)
    res = extract_decoding(rgdp,decode(IPDecoder(),gdp,gs).error_pattern)
    @test syn == syndrome_extraction(res.error_pattern, tanner)

    gdp,rgdp = reduce2general(tanner,iid_error(0.05,0.06,0.1,9))
    res = extract_decoding(rgdp,decode(IPDecoder(),gdp,gs).error_pattern)
    @test syn == syndrome_extraction(res.error_pattern, tanner)
end

@testset "correlated error model" begin
    Random.seed!(123)
    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    em = iid_error(0.05,0.06,0.1,9)
    ep = random_error_pattern(em)
    syn = syndrome_extraction(ep,tanner)
    gs = TensorQEC.SimpleSyndrome([syn.sx...,syn.sz...])

    num_qubits = 9
    code = DynamicEinCode([[[i,i+num_qubits] for i in [1,2,3,5,6,8,9]]...,vcat([[i,i+num_qubits] for i in [4,7]]...)],Int[])
    a = zeros(2,2,2,2)
    a[1,1,1,1] = 0.99
    a[1,1,2,2] = 0.01
    tensors= [[TensorQEC.single_qubit_tensor(0.05,0.03,0.01) for j in [1,2,3,5,6,8,9]]...,a]
    tn = TensorQEC.SimpleTensorNetwork(code,tensors)
    gdp,rgdp = reduce2general(tanner,tn)

    res = extract_decoding(rgdp,decode(IPDecoder(),gdp,gs).error_pattern)
    @test syn == syndrome_extraction(res.error_pattern, tanner)

    res = extract_decoding(rgdp,decode(TNMAP(),gdp,gs).error_pattern)
    @test syn == syndrome_extraction(res.error_pattern, tanner)
end

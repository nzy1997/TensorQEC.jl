using Test
using TensorQEC
using TensorQEC:  flattengdp
using Random
using TensorQEC.OMEinsum
using TensorQEC.Yao

@testset "IPDecoder" begin
    tanner = CSSTannerGraph(SurfaceCode(3,3))
    decoder = IPDecoder()
    error_qubits = Mod2[0,0,0,1,0,0,0,0,0]
    syd = syndrome_extraction(error_qubits, tanner.stgz)

    res = decode(decoder,tanner.stgz,syd)
    @test syd == syndrome_extraction(res.error_qubits, tanner.stgz)

    res = decode(decoder,tanner.stgz,syd,0.05)
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
    em = DepolarizingError(0.05, 0.06, 0.1)
    ep = random_error_qubits(9, em)
    syn = syndrome_extraction(ep,tanner)

    res = decode(IPDecoder(),tanner,syn)
    @test syn.sx == syndrome_extraction(res.zerror_qubits, tanner.stgx)
    @test syn.sz == syndrome_extraction(res.xerror_qubits, tanner.stgz)

    res = decode(IPDecoder(),tanner,syn,fill(em,9))
    @test syn.sx == syndrome_extraction(res.zerror_qubits, tanner.stgx)
    @test syn.sz == syndrome_extraction(res.xerror_qubits, tanner.stgz)

    res = decode(IPDecoder(),tanner,syn,fill(DepolarizingError(0.05, 0.06, 0.0),9))
    @test syn.sx == syndrome_extraction(res.zerror_qubits, tanner.stgx)
    @test syn.sz == syndrome_extraction(res.xerror_qubits, tanner.stgz)
end

@testset "gdp,fdp" begin
    Random.seed!(123)
    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    em = DepolarizingError(0.05, 0.06, 0.1)
    ep = random_error_qubits(9, em)
    syn = syndrome_extraction(ep,tanner)
    gs = general_syndrome(syn)

    rgdp = reduce2general(tanner,fill([0.10,0.0,0.1],9))
    res = extract_decoding(rgdp,decode(IPDecoder(),rgdp.gdp,gs))
    @test syn.sx == syndrome_extraction(res.zerror_qubits, tanner.stgx)
    @test syn.sz == syndrome_extraction(res.xerror_qubits, tanner.stgz)


    rgdp = reduce2general(tanner,fill([0.10,0.0,0.1],9))
    res = extract_decoding(rgdp,decode(IPDecoder(),rgdp.gdp,gs))
    @test syn.sx == syndrome_extraction(res.zerror_qubits, tanner.stgx)
    @test syn.sz == syndrome_extraction(res.xerror_qubits, tanner.stgz)


    rgdp = reduce2general(tanner,fill([0.10,0.0,0.1],9))
    res = extract_decoding(rgdp,decode(IPDecoder(),rgdp.gdp,gs))
    @test syn.sx == syndrome_extraction(res.zerror_qubits, tanner.stgx)
    @test syn.sz == syndrome_extraction(res.xerror_qubits, tanner.stgz)

    num_qubits = 9
    code = DynamicEinCode([[[i,i+num_qubits,i+2*num_qubits] for i in [1,2,3,5,6,8,9]]...,vcat([[i,i+num_qubits,i+2*num_qubits] for i in [4,7]]...)],Int[])
    a = zeros(2,2,2,2,2,2)
    a[1,1,1,1,1,1] = 0.99
    a[1,1,2,1,1,2] = 0.01
    tensors= [[TensorQEC.single_qubit_tensor(0.05,0.03,0.01) for j in [1,2,3,5,6,8,9]]...,a]
    tn = TensorNetwork(code,tensors)
    rgdp = reduce2general(tanner,tn)
    res = extract_decoding(rgdp,decode(IPDecoder(),rgdp.gdp,gs))
    @test syn.sx == syndrome_extraction(res.zerror_qubits, tanner.stgx)
    @test syn.sz == syndrome_extraction(res.xerror_qubits, tanner.stgz)
end
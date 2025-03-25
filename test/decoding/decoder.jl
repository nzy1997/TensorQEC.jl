using Test
using TensorQEC
using TensorQEC:  flattengdp
using Random
using TensorQEC.OMEinsum
using TensorQEC.Yao

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

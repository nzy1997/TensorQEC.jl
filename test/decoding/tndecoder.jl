using Test
using TensorQEC
using TensorQEC.Yao
using TensorQEC.OMEinsum
using TensorQEC.TensorInference
using Random

@testset "parity_check_matrix" begin
    mat = TensorQEC.parity_check_matrix(4)
    @test mat[1,1,1,1,1] == 1
    @test mat[1,1,1,1,2] == 0
    @test mat[1,2,1,2,1] == 1
    @test mat[2,1,2,2,1] == 0
end

@testset "decode" begin
    Random.seed!(123)
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d, d))
    em = iid_error(0.05,d*d)
    ep = random_error_qubits(em)
    syn = syndrome_extraction(ep,tanner.stgz)

    decoder = TNMAP()
    res = decode(decoder,tanner.stgz,syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner.stgz)
end

@testset "compile and decode" begin
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d, d))

    ct = compile(TNMAP(),tanner.stgz)

    Random.seed!(123)
    em = iid_error(0.05,d*d)
    ep = random_error_qubits(em)
    syn = syndrome_extraction(ep,tanner.stgz)
    
    res = decode(ct,syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner.stgz)
end

@testset "CSS compile and decode" begin
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d, d))

    ct = compile(TNMAP(),tanner)

    Random.seed!(123)
    em = iid_error(0.05,0.05,0.05,d*d)
    ep = random_error_qubits(em)
    syn = syndrome_extraction(ep,tanner)
    
    res = decode(ct,syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner)
end
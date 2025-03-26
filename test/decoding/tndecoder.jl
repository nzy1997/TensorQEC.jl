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
    em = FlipError(0.05)
    ep = random_error_qubits(d*d, em)
    syn = syndrome_extraction(ep,tanner.stgz)

    decoder = TNMAP()
    res = decode(decoder,tanner.stgz,syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner.stgz)
end
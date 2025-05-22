using Test
using TensorQEC
using Random

@testset "threshold_qec" begin
    Random.seed!(123)
    tanner = CSSTannerGraph(SurfaceCode(3,3))
    decoder = IPDecoder()
    @test multi_round_qec(tanner.stgz,decoder,iid_error(0.05,9),tanner.stgx) isa Float64

    @test multi_round_qec(tanner,decoder,iid_error(0.05,0.05,0.05,9)) isa Tuple{Float64, Float64, Float64}
end
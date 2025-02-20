using Test
using TensorQEC
using TensorQEC: QuantumSimulationResult,ClassicalSimulationResult


@testset "threshold_qec" begin
    tanner = CSSTannerGraph(SurfaceCode(3,3))
    decoder = BPOSD(100)
    error_model_vec = [DepolarizingError(0.01),DepolarizingError(0.02)]
    res = threshold_qec(tanner,decoder,error_model_vec)
    @test length(res) == 2
    @test res[1] isa QuantumSimulationResult
    @test res[2] isa QuantumSimulationResult

    error_model_vec= [FlipError(0.01),FlipError(0.02)]
    res = threshold_qec(tanner.stgx,decoder,error_model_vec,tanner.stgz)
    @test length(res) == 2
    @test res[1] isa ClassicalSimulationResult
    @test res[2] isa ClassicalSimulationResult
end
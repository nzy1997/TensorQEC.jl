using Test
using TensorQEC

@testset "detector error model" begin
    circuit_str = """
    X_ERROR(0.2) 0 1
    # DEPOLARIZE1(0.0001) 0 1

    CX 0 1

    M 0 1
    DETECTOR rec[-1]
    DETECTOR rec[-2]
    DETECTOR rec[-1] rec[-2]
    """
    qc = TensorQEC.parse_stim_string(circuit_str,2)

    dem = TensorQEC.detector_error_model(qc)
    @test dem.error_rates == [0.2,0.2]
    @test dem.flipped_detectors == [[1,2],[1,3]]
end


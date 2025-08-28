using Test
using TensorQEC

@testset "detector error model" begin
    circuit_str = """
    X_ERROR(0.2) 0 1
    DEPOLARIZE1(0.03) 0 1

    CX 0 1

    M 0 1
    DETECTOR rec[-1]
    DETECTOR rec[-2]
    DETECTOR rec[-1] rec[-2]
    OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
    """
    qc = TensorQEC.parse_stim_string(circuit_str,2)

    dem = TensorQEC.detector_error_model(qc)
    println(dem)
    @test dem.error_rates ≈ [0.212,0.212] atol=1e-10
    @test dem.flipped_detectors == [[1,3,4],[1,2]]
end

@testset "detector error model with depolarizing error" begin
    circuit_str = """
    DEPOLARIZE1(0.3) 0
    M 0
    DETECTOR rec[-1]
    """
    qc = TensorQEC.parse_stim_string(circuit_str,1)

    dem = TensorQEC.detector_error_model(qc)
    println(dem)
    @test dem.error_rates ≈ [0.2] atol=1e-10
    @test dem.flipped_detectors == [[1]]

    circuit_str = """
    H 0
    CX 0 1
    DEPOLARIZE1(0.0297) 0
    H 3
    CX 0 2 1 2 3 0 3 1
    H 3
    M 2 3    
    DETECTOR rec[-2]
    DETECTOR rec[-1]
    """
    qc = TensorQEC.parse_stim_string(circuit_str,4)

    dem = TensorQEC.detector_error_model(qc)
    println(dem)
    @test dem.error_rates ≈ [0.01,0.01,0.01] atol=1e-10
    @test dem.flipped_detectors == [[1],[1,2],[2]]
end

@testset "detector error model with reset" begin
    circuit_str ="""
    X_ERROR(0.2) 0 1

    CX 0 1

    M 0 1
    DETECTOR rec[-1] rec[-2]

    R 1

    X_ERROR(0.2) 0 1
    CX 1 0

    M 0 1
    DETECTOR rec[-1] rec[-2]   
    """
    qc = TensorQEC.parse_stim_string(circuit_str,2)

    dem = TensorQEC.detector_error_model(qc)
    println(dem)
    @test dem.error_rates ≈ [0.2,0.32] atol=1e-10
    @test dem.flipped_detectors == [[1],[2]]
end

@testset "detector error model with noisy circuit" begin
    # The following noisy circuit is generated with stim by the following command:
    # noisy_circuit = stim.Circuit.generated(
    # "color_code:memory_xyz",
    # rounds=2,
    # distance=3,
    # before_round_data_depolarization=0.04,
    # before_measure_flip_probability=0.01)
    function check_dem(dem1, dem2)
        tag = true
        tag = tag && dem1.detector_list == dem2.detector_list
        tag = tag && dem1.logical_list == dem2.logical_list

        for (ep,detectors) in zip(dem1.error_rates, dem1.flipped_detectors)
            pos = findall(x->x==detectors, dem2.flipped_detectors)
            tag = tag && length(pos) == 1
            tag = tag && isapprox(ep, dem2.error_rates[pos[1]], atol=1e-10)
        end
        return tag
    end
    qc = parse_stim_file(joinpath(@__DIR__, "test_circuits", "noisy_circuit.stim"), 10);
    dem1 = TensorQEC.detector_error_model(qc)
    dem2 = TensorQEC.parse_dem_file(joinpath(@__DIR__, "test_circuits", "dem.dem"))
    # println(dem1)
    # println(dem2)
    @test check_dem(dem1, dem2)
end

@testset "insert errors" begin
    circuit_str ="""
    H 0 1

    CX 0 1

    M 0
    MR 1
    DETECTOR rec[-1] rec[-2]

    R 1

    CX 1 0

    M 0 1
    DETECTOR rec[-1] rec[-2]
    
    C_XYZ 1
    """
    qc = TensorQEC.parse_stim_string(circuit_str,2)
    # vizcircuit(qc)
    qce = TensorQEC.insert_errors(qc;after_clifford_depolarization=0.01,after_reset_flip_probability=0.02,before_measure_flip_probability=0.03)
    vizcircuit(qce)
end

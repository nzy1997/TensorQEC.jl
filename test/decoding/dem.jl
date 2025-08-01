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

@testset "detector error model" begin
    circuit_str ="""
    X_ERROR(0.2) 0 1

    CX 0 1

    M 0 1
    DETECTOR rec[-1] rec[-2]

    R 1

    X_ERROR(0.2) 0 1
    CX 0 1

    M 0 1
    DETECTOR rec[-1] rec[-2]   
    """
    qc = TensorQEC.parse_stim_string(circuit_str,2)

    dem = TensorQEC.detector_error_model(qc)
    @test dem.error_rates == [0.2, 0.2, 0.2, 0.2]
    @test dem.flipped_detectors == [Int64[], [1], Int64[], [2]]
end

@testset "detector error model" begin
    circuit_str = """
    QUBIT_COORDS(0, 0) 0
    QUBIT_COORDS(1, 0) 1
    QUBIT_COORDS(2, 0) 2
    QUBIT_COORDS(3, 0) 3
    QUBIT_COORDS(0.5, 1) 4
    QUBIT_COORDS(1.5, 1) 5
    QUBIT_COORDS(2.5, 1) 6
    QUBIT_COORDS(1, 2) 7
    QUBIT_COORDS(2, 2) 8
    QUBIT_COORDS(1.5, 3) 9
    R 0 1 2 3 4 5 6 7 8 9
    REPEAT 2 {
        TICK
        DEPOLARIZE1(0.04) 0 1 3 5 6 7 9
        C_XYZ 0 1 3 5 6 7 9
        TICK
        CX 5 4 3 2
        TICK
        CX 7 4 6 2
        TICK
        CX 1 4 6 8
        TICK
        CX 1 2 7 8
        TICK
        CX 5 2 9 8
        TICK
        CX 0 4 5 8
        TICK
        X_ERROR(0.01) 2 4 8
        MR 2 4 8
    }
    DETECTOR(2, 0, 0) rec[-3] rec[-6]
    DETECTOR(0.5, 1, 0) rec[-2] rec[-5]
    DETECTOR(2, 2, 0) rec[-1] rec[-4]
    X_ERROR(0.01) 0 1 3 5 6 7 9
    MY 0 1 3 5 6 7 9
    DETECTOR(2, 0, 1) rec[-3] rec[-4] rec[-5] rec[-6] rec[-10] rec[-13]
    DETECTOR(0.5, 1, 1) rec[-2] rec[-4] rec[-6] rec[-7] rec[-9] rec[-12]
    DETECTOR(2, 2, 1) rec[-1] rec[-2] rec[-3] rec[-4] rec[-8] rec[-11]
    OBSERVABLE_INCLUDE(0) rec[-5] rec[-6] rec[-7]
    """
    qc = TensorQEC.parse_stim_string(circuit_str,10)

    dem = TensorQEC.detector_error_model(qc)
    @test dem.error_rates == [0.2,0.2]
    @test dem.flipped_detectors == [[1,2],[1,3]]
end


using TensorQEC
using Test
using Yao

@testset "parse_stim_string" begin
    circuit_str = """
        CNOT 0 1 2 3 4 5 # A CNOT gate
        CNOT 2 1 4 3 6 5
        MR 1 3 5

        DETECTOR rec[-3]
        DETECTOR rec[-2]
        DETECTOR rec[-1]

        REPEAT 2 {
            CNOT 0 1 2 3 4 5 # A CNOT gate
            CNOT 2 1 4 3 6 5
            MR 1 3 5

            DETECTOR rec[-3] rec[-6]
            DETECTOR rec[-2] rec[-5]
            DETECTOR rec[-1] rec[-4]
            REPEAT 2 
            {
            X 0 1 2 3 4 5 6
            }
        }

        M(0.1) 0 2 4 6

        DETECTOR rec[-3] rec[-4] rec[-7]
        DETECTOR rec[-2] rec[-3] rec[-6]
        DETECTOR rec[-1] rec[-2] rec[-5]

        OBSERVABLE_INCLUDE(0) rec[-1]
    """

    qc = TensorQEC.parse_stim_string(circuit_str, 7)
    # vizcircuit(qc)
    @test qc isa ChainBlock
end

@testset "error handling" begin
    circuit_str = """
        WRONG_INSTRUCTION 1
    """
    @test_throws ErrorException TensorQEC.parse_stim_string(circuit_str, 5)
end

# The following testing circuits are copied from https://github.com/quantumlib/Stim/blob/main/doc/file_format_stim_circuit.md
@testset "teleportation.stim" begin
    qc = parse_stim_file(joinpath(@__DIR__, "test_circuits", "teleportation.stim"), 100);
    vizcircuit(qc)
    @test qc isa ChainBlock
end

@testset "repetition.stim" begin
    qc = parse_stim_file(joinpath(@__DIR__, "test_circuits", "repetition.stim"), 7);
    @test qc isa ChainBlock
end

@testset "noisy_repetition.stim" begin
    qc = parse_stim_file(joinpath(@__DIR__, "test_circuits", "noisy_repetition.stim"), 7);
    @test qc isa ChainBlock
end

@testset "noisy_surface.stim" begin
    qc = parse_stim_file(joinpath(@__DIR__, "test_circuits", "noisy_surface.stim"), 26)
    @test qc isa ChainBlock
end

# This circuit is from https://github.com/quantumlib/tesseract-decoder
@testset "stim file from tesseract-decoder" begin
    qc = parse_stim_file(joinpath(@__DIR__, "test_circuits", "r=12,d=12,p=0.001,noise=si1000,c=bivariate_bicycle_X,nkd=[[144,12,12]],q=288,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim"), 288);
    @test qc isa ChainBlock
end

@testset "dem.dem" begin
    dem = TensorQEC.parse_dem_file(joinpath(@__DIR__, "test_circuits", "dem.dem"))
    @test dem isa TensorQEC.DetectorErrorModel
end

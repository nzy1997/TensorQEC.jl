using TensorQEC, Test

@testset "deprecate" begin
    # v0.3 typo fixes
    @test TensorQEC.TrainningData === TensorQEC.TrainingData
end

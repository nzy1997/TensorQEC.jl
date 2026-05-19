using Test
using TensorQEC

bundle_dir() = normpath(joinpath(@__DIR__, "../../../../example/testing_case_cache/color_code_decode_bundle"))

@testset "bundle matching api" begin
    @test isdefined(TensorQEC, :BundleSideDecodingProblem)
    @test isdefined(TensorQEC, :BundleCSSDecodingProblem)
    @test isdefined(TensorQEC, :BundleMatchingDecoder)

    problem = BundleSideDecodingProblem(bundle_dir(), :decode_z)
    @test problem.side === :decode_z
    @test_throws ArgumentError BundleSideDecodingProblem(bundle_dir(), :x)

    css_problem = BundleCSSDecodingProblem(bundle_dir())
    @test css_problem.bundle_dir == bundle_dir()

    decoder = BundleMatchingDecoder(check_valid_syndrome=false)
    @test decoder.check_valid_syndrome == false
end

@testset "bundle matching compile path" begin
    problem = BundleSideDecodingProblem(bundle_dir(), :decode_x)
    decoder = BundleMatchingDecoder()
    compiled = compile(decoder, problem)

    @test compiled.c0_dim == 75
    @test compiled.c1_dim == 150
end

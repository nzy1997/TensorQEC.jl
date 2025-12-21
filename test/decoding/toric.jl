using Test
using TensorQEC
using DelimitedFiles
using Random

@testset "TToricDecoder" begin
    d= 3
    Random.seed!(123)
    c = FileCode(joinpath(@__DIR__,"../codes","test_codes","color_code_$(d).txt"), "color_code_$(d)")
    sts = stabilizers(c; remove_linear_dependency=false)
    tanner = CSSTannerGraph(sts)
    row_transformation = Mod2.(readdlm(joinpath(@__DIR__,"../codes","test_codes","color_code_3_row_transformation.txt"), Bool))
    column_transformation = Mod2.(readdlm(joinpath(@__DIR__,"../codes","test_codes","color_code_3_column_transformation.txt"), Bool))

    ct = compile(TToricDecoder(row_transformation,column_transformation,1,2,9), tanner)

    for _ in 1:100
    em = iid_error(0.05, 0.05, 0.05,tanner.stgx.nq)
    ep = random_error_pattern(em)
    syn = syndrome_extraction(ep,tanner)
    deres = decode(ct, syn)
    @test syn == syndrome_extraction(deres.error_pattern, tanner)
    end
end
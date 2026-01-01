using Test
using TensorQEC
using DelimitedFiles
using Random
using TensorQEC: save_ttoric_decoder, load_ttoric_decoder

@testset "TToricDecoder" begin
    d= 3
    Random.seed!(123)
    c = FileCode(joinpath(@__DIR__,"../codes","test_codes","color_code_$(d).txt"), "color_code_$(d)")
    sts = stabilizers(c; remove_linear_dependency=false)
    tanner = CSSTannerGraph(sts)
    row_transformation = Mod2.(readdlm(joinpath(@__DIR__,"../codes","test_codes","color_code_3_row_transformation.txt"), Bool))
    column_transformation = Mod2.(readdlm(joinpath(@__DIR__,"../codes","test_codes","color_code_3_column_transformation.txt"), Bool))

    # decoder_old = TToricDecoder(row_transformation,column_transformation,1,2,9)
    # decoder = TToricDecoder(row_transformation,column_transformation,[(1,1)],[(2,2)],[[3,4],[5,6]],[[3,4],[5,6]],9)
    decoder = TToricDecoder(row_transformation,column_transformation,[(1,1)],[(4,2)],[[3,4],[5,6]],[[2],[3]],[[5],[6]],9)

    save_ttoric_decoder(joinpath(@__DIR__,"../codes","test_codes","color_code_3_toric_decoder.json"), decoder)
    decoder_loaded = load_ttoric_decoder(joinpath(@__DIR__,"../codes","test_codes","color_code_3_toric_decoder.json"))

    ct = compile(decoder_loaded, tanner)

    for _ in 1:100
    em = iid_error(0.05, 0.05, 0.05,tanner.stgx.nq)
    ep = random_error_pattern(em)
    syn = syndrome_extraction(ep,tanner)
    deres = decode(ct, syn)
    @test syn == syndrome_extraction(deres.error_pattern, tanner)
    end
end

@testset "TToricDecoderbbx^-1y" begin
    Random.seed!(123)
    c = FileCode(joinpath(@__DIR__,"../codes","test_codes","bbx^-1y_3.txt"), "bbx^-1y_3")
    sts = stabilizers(c; remove_linear_dependency=false)
    tanner = CSSTannerGraph(sts)
    decoder_loaded = load_ttoric_decoder(joinpath(@__DIR__,"../codes","test_codes","bbx^-1y_3_decoder.json"); switch_xz_part = true)

    ct = compile(decoder_loaded, tanner)

    for _ in 1:100
    em = iid_error(0.05, 0.05, 0.05,tanner.stgx.nq)
    ep = random_error_pattern(em)
    syn = syndrome_extraction(ep,tanner)
    deres = decode(ct, syn)
    @test syn == syndrome_extraction(deres.error_pattern, tanner)
    end
end

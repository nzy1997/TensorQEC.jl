using DelimitedFiles
using TensorQEC

fixture_root = pkgdir(TensorQEC, "test", "codes", "test_codes")
code = FileCode(joinpath(fixture_root, "color_code_3.txt"), "color_code_3")
tanner = CSSTannerGraph(stabilizers(code; remove_linear_dependency=false))
row = Mod2.(readdlm(joinpath(fixture_root, "color_code_3_row_transformation.txt"), Bool))
column = Mod2.(readdlm(joinpath(fixture_root, "color_code_3_column_transformation.txt"), Bool))
compiled = compile(TToricDecoder(row, column, 1, 2, 9), tanner)
lx, lz = logical_operator(tanner)

n = tanner.stgx.nq
xerror = fill(Mod2(0), n)
zerror = fill(Mod2(0), n)
xerror[1] = Mod2(1)
error_pattern = CSSErrorPattern(xerror, zerror)
syndrome = syndrome_extraction(error_pattern, tanner)
result = decode(compiled, syndrome)

syndrome_extraction(result.error_pattern, tanner) == syndrome ||
    error("paper decoder returned a correction with the wrong syndrome")
!check_logical_error(result.error_pattern, error_pattern, lx, lz) ||
    error("paper decoder correction introduced a logical error")

println("decoder=unitary-decouple")
println("matching=unweighted")
println("fixture=color_code_3")
println("paper decoder smoke: ok")

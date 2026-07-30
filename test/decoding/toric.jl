using Test
using TensorQEC
using DelimitedFiles

function paper_decoder_fixture()
    d = 3
    fixture_root = joinpath(@__DIR__, "..", "codes", "test_codes")
    code = FileCode(joinpath(fixture_root, "color_code_3.txt"), "color_code_3")
    tanner = CSSTannerGraph(stabilizers(code; remove_linear_dependency=false))
    row = Mod2.(readdlm(joinpath(fixture_root, "color_code_3_row_transformation.txt"), Bool))
    column = Mod2.(readdlm(joinpath(fixture_root, "color_code_3_column_transformation.txt"), Bool))
    compiled = compile(TToricDecoder(row, column, 1, 2, d^2), tanner)
    lx, lz = logical_operator(tanner)
    return (; tanner, row, column, compiled, lx, lz)
end

@testset "TToricDecoder reports invalid column dimensions" begin
    fixture = paper_decoder_fixture()
    invalid_decoder = TToricDecoder(fixture.row, zeros(Mod2, 1, 1), 1, 2, 9)
    err = try
        compile(invalid_decoder, fixture.tanner)
        nothing
    catch caught
        caught
    end
    @test err isa DimensionMismatch
    @test occursin("column_transformation", sprint(showerror, err))
end

@testset "TToricDecoder corrects deterministic single-qubit errors" begin
    fixture = paper_decoder_fixture()
    n = fixture.tanner.stgx.nq
    for qubit in 1:min(n, 8), pauli in (:x, :z)
        xerror = fill(Mod2(0), n)
        zerror = fill(Mod2(0), n)
        pauli === :x ? (xerror[qubit] = Mod2(1)) : (zerror[qubit] = Mod2(1))
        error_pattern = CSSErrorPattern(xerror, zerror)
        syndrome = syndrome_extraction(error_pattern, fixture.tanner)
        result = decode(fixture.compiled, syndrome)
        @test syndrome_extraction(result.error_pattern, fixture.tanner) == syndrome
        @test !check_logical_error(
            result.error_pattern,
            error_pattern,
            fixture.lx,
            fixture.lz,
        )
    end
end

@testset "TToricDecoder reports invalid transformation dimensions" begin
    fixture = paper_decoder_fixture()
    invalid_decoder = TToricDecoder(zeros(Mod2, 1, 1), zeros(Mod2, 1, 1), 1, 2, 9)
    err = try
        compile(invalid_decoder, fixture.tanner)
        nothing
    catch caught
        caught
    end
    @test err isa DimensionMismatch
    @test occursin("row_transformation", sprint(showerror, err))
end

@testset "paper decoder smoke example" begin
    root = normpath(joinpath(@__DIR__, "..", ".."))
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(root) $(joinpath(root, "examples", "paper_decoder_smoke.jl"))`
    mktemp() do _, io
        process = run(pipeline(ignorestatus(command), stdout=io, stderr=io))
        seekstart(io)
        output = read(io, String)
        @test success(process)
        @test endswith(strip(output), "paper decoder smoke: ok")
    end
end

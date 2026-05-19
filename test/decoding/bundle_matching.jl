using Test
using TensorQEC
using JSON

function bundle_dir()
    for candidate in (
        normpath(joinpath(@__DIR__, "../../../../example/testing_case_cache/color_code_decode_bundle")),
        normpath(joinpath(@__DIR__, "../../../../../../example/testing_case_cache/color_code_decode_bundle")),
    )
        isdir(candidate) && return candidate
    end
    error("fixture bundle not found from $(abspath(@__DIR__))")
end

const RUNTIME_REQUIRED_FILES = ("check_matrix", "epsilon_c", "d1_toric", "phi_0", "psi_1", "K_0")

load_manifest(dir::AbstractString) = JSON.parsefile(joinpath(dir, "manifest.json"))

function write_manifest(dir::AbstractString, manifest)
    open(joinpath(dir, "manifest.json"), "w") do io
        write(io, JSON.json(manifest, 2))
    end
end

function copy_runtime_bundle(dest::AbstractString; sides=(:decode_x, :decode_z))
    manifest = load_manifest(bundle_dir())
    cp(joinpath(bundle_dir(), "manifest.json"), joinpath(dest, "manifest.json"))
    for side in sides
        mkpath(joinpath(dest, String(side)))
        for key in RUNTIME_REQUIRED_FILES
            relative_path = manifest[String(side)]["files"][key]["path"]
            cp(joinpath(bundle_dir(), relative_path), joinpath(dest, relative_path))
        end
    end
    return dest
end

function rewrite_lines(path::AbstractString, lines::Vector{String})
    open(path, "w") do io
        for line in lines
            println(io, line)
        end
    end
end

function zero_syndrome(compiled)
    return SimpleSyndrome(fill(Mod2(0), compiled.c0_dim))
end

function first_valid_syndrome(compiled)
    for qubit in 1:compiled.c1_dim
        syndrome = compiled.check_matrix[:, qubit]
        any(v -> v.x, syndrome) || continue
        any(v -> v.x, compiled.epsilon_c * syndrome) && continue
        return SimpleSyndrome(copy(syndrome))
    end
    error("failed to locate a valid nonzero syndrome for $(compiled.side)")
end

function first_invalid_syndrome(compiled)
    for check in 1:compiled.c0_dim
        syndrome = fill(Mod2(0), compiled.c0_dim)
        syndrome[check] = Mod2(1)
        any(v -> v.x, compiled.epsilon_c * syndrome) || continue
        return SimpleSyndrome(syndrome)
    end
    error("failed to locate an invalid syndrome for $(compiled.side)")
end

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
    decoder = BundleMatchingDecoder()

    compiled_x = compile(decoder, BundleSideDecodingProblem(bundle_dir(), :decode_x))
    @test compiled_x.side === :decode_x
    @test compiled_x.check_valid_syndrome === true
    @test compiled_x.c0_dim == 75
    @test compiled_x.c1_dim == 150
    @test compiled_x.t0_dim == 150
    @test compiled_x.t1_dim == 300
    @test size(compiled_x.check_matrix) == (75, 150)
    @test size(compiled_x.epsilon_c) == (2, 75)
    @test size(compiled_x.d1_toric) == (150, 300)
    @test size(compiled_x.phi_0) == (150, 75)
    @test size(compiled_x.psi_1) == (150, 300)
    @test size(compiled_x.K_0) == (150, 75)
    @test eltype(compiled_x.check_matrix) === Mod2

    compiled_z = compile(decoder, BundleSideDecodingProblem(bundle_dir(), :decode_z))
    @test compiled_z.side === :decode_z
    @test compiled_z.c0_dim == 75
    @test compiled_z.c1_dim == 150

    compiled_css = compile(decoder, BundleCSSDecodingProblem(bundle_dir()))
    @test compiled_css.side_x.side === :decode_x
    @test compiled_css.side_z.side === :decode_z
end

@testset "bundle matching compile validation" begin
    decoder = BundleMatchingDecoder()

    mktempdir() do tempdir
        copy_runtime_bundle(tempdir)
        manifest = load_manifest(tempdir)
        manifest["format"] = "wrong_format"
        write_manifest(tempdir, manifest)
        @test_throws ArgumentError compile(decoder, BundleSideDecodingProblem(tempdir, :decode_x))
    end

    for missing_side in ("decode_x", "decode_z")
        mktempdir() do tempdir
            copy_runtime_bundle(tempdir)
            manifest = load_manifest(tempdir)
            delete!(manifest, missing_side)
            write_manifest(tempdir, manifest)
            @test_throws ArgumentError compile(decoder, BundleCSSDecodingProblem(tempdir))
        end
    end

    mktempdir() do tempdir
        copy_runtime_bundle(tempdir; sides=(:decode_x,))
        rm(joinpath(tempdir, "decode_x", "phi_0.txt"))
        @test_throws ArgumentError compile(decoder, BundleSideDecodingProblem(tempdir, :decode_x))
    end

    mktempdir() do tempdir
        copy_runtime_bundle(tempdir; sides=(:decode_x,))
        check_matrix_path = joinpath(tempdir, "decode_x", "check_matrix.txt")
        rewrite_lines(check_matrix_path, readlines(check_matrix_path)[1:end-1])
        @test_throws ArgumentError compile(decoder, BundleSideDecodingProblem(tempdir, :decode_x))
    end

    mktempdir() do tempdir
        copy_runtime_bundle(tempdir; sides=(:decode_x,))
        d1_toric_path = joinpath(tempdir, "decode_x", "d1_toric.txt")
        rows = split.(readlines(d1_toric_path))
        rows[3][1] = "1"
        rewrite_lines(d1_toric_path, [join(row, " ") for row in rows])
        @test_throws ArgumentError compile(decoder, BundleSideDecodingProblem(tempdir, :decode_x))
    end
end

@testset "bundle matching css wrapper decode" begin
    problem = BundleCSSDecodingProblem(bundle_dir())
    decoder = BundleMatchingDecoder()
    compiled = compile(decoder, problem)

    sx = fill(Mod2(0), compiled.side_z.c0_dim)
    sz = fill(Mod2(0), compiled.side_x.c0_dim)
    sx .= first_valid_syndrome(compiled.side_z).s
    sz .= first_valid_syndrome(compiled.side_x).s

    res = decode(compiled, CSSSyndrome(sx, sz))

    @test length(res.error_pattern.xerror) == compiled.side_x.c1_dim
    @test length(res.error_pattern.zerror) == compiled.side_z.c1_dim
    @test sx == compiled.side_z.check_matrix * res.error_pattern.zerror
    @test sz == compiled.side_x.check_matrix * res.error_pattern.xerror
end

@testset "bundle matching single-side decode" begin
    for side in (:decode_x, :decode_z)
        compiled = compile(BundleMatchingDecoder(), BundleSideDecodingProblem(bundle_dir(), side))
        compiled_no_check = compile(
            BundleMatchingDecoder(check_valid_syndrome=false),
            BundleSideDecodingProblem(bundle_dir(), side),
        )

        result_zero = decode(compiled, zero_syndrome(compiled))
        @test result_zero.success_tag === true
        @test result_zero.error_pattern == fill(Mod2(0), compiled.c1_dim)

        valid_syndrome = first_valid_syndrome(compiled)
        result_valid = decode(compiled, valid_syndrome)
        @test result_valid.success_tag === true
        @test compiled.check_matrix * result_valid.error_pattern == valid_syndrome.s

        invalid_syndrome = first_invalid_syndrome(compiled)
        @test_throws ArgumentError decode(compiled, invalid_syndrome)

        result_invalid = decode(compiled_no_check, invalid_syndrome)
        @test result_invalid isa DecodingResult
        @test length(result_invalid.error_pattern) == compiled.c1_dim
    end
end

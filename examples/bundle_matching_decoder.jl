using TensorQEC

function bundle_dir()
    for candidate in (
        normpath(joinpath(@__DIR__, "../../../example/testing_case_cache/color_code_decode_bundle")),
        normpath(joinpath(@__DIR__, "../../../../example/testing_case_cache/color_code_decode_bundle")),
        normpath(joinpath(@__DIR__, "../../../../../example/testing_case_cache/color_code_decode_bundle")),
    )
        isdir(candidate) && return candidate
    end
    error("fixture bundle not found from $(abspath(@__DIR__))")
end

function first_valid_syndrome(compiled)
    for qubit in 1:compiled.c1_dim
        syndrome = compiled.check_matrix[:, qubit]
        any(v -> v.x, syndrome) || continue
        any(v -> v.x, compiled.epsilon_c * syndrome) && continue
        return copy(syndrome)
    end
    error("failed to locate a valid nonzero syndrome for $(compiled.side)")
end

compiled = compile(BundleMatchingDecoder(), BundleCSSDecodingProblem(bundle_dir()))
syndrome = CSSSyndrome(
    first_valid_syndrome(compiled.side_z),
    first_valid_syndrome(compiled.side_x),
)
result = decode(compiled, syndrome)

println("xerror size: ", length(result.error_pattern.xerror))
println("zerror size: ", length(result.error_pattern.zerror))

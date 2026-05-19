struct BundleSideDecodingProblem <: AbstractDecodingProblem
    bundle_dir::String
    side::Symbol

    function BundleSideDecodingProblem(bundle_dir::String, side::Symbol)
        side in (:decode_x, :decode_z) || throw(ArgumentError("side must be :decode_x or :decode_z"))
        return new(bundle_dir, side)
    end
end

struct BundleCSSDecodingProblem <: AbstractDecodingProblem
    bundle_dir::String
end

Base.@kwdef struct BundleMatchingDecoder <: AbstractClassicalDecoder
    check_valid_syndrome::Bool = true
end

struct CompiledBundleSideMatching <: CompiledDecoder
    bundle_dir::String
    side::Symbol
    check_valid_syndrome::Bool
end

function compile(decoder::BundleMatchingDecoder, problem::BundleSideDecodingProblem)
    return CompiledBundleSideMatching(
        problem.bundle_dir,
        problem.side,
        decoder.check_valid_syndrome,
    )
end

function compile(decoder::BundleMatchingDecoder, problem::BundleCSSDecodingProblem)
    return CompiledClassicalDecoder(
        compile(decoder, BundleSideDecodingProblem(problem.bundle_dir, :decode_x)),
        compile(decoder, BundleSideDecodingProblem(problem.bundle_dir, :decode_z)),
    )
end

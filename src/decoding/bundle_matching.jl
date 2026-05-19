struct BundleSideDecodingProblem <: AbstractBundleDecodingProblem
    bundle_dir::String
    decode_task::Symbol
end

struct BundleCSSDecodingProblem <: AbstractBundleDecodingProblem
    bundle_dir::String
end

struct BundleMatchingDecoder <: AbstractDecoder end

function compile(::BundleMatchingDecoder, ::BundleSideDecodingProblem)
    throw(ArgumentError("not implemented"))
end

function compile(::BundleMatchingDecoder, ::BundleCSSDecodingProblem)
    throw(ArgumentError("not implemented"))
end

# # Error Model and Syndrome Extraction
# ## Error Model
# We have 2 types of error models. [`IndependentFlipError`](@ref) is for the case that the error model is independent flip error. [`IndependentDepolarizingError`](@ref) is the error model for the case that the error model is independent depolarizing error. We can use [`iid_error`](@ref) to easily generate an independent and identically distributed error model.
using TensorQEC
iide = iid_error(0.05,0.05,0.05,7)

# We can generate a random error pattern with [`random_error_pattern`](@ref).
for _ in 1:10
    print(random_error_pattern(iide))
end

# This is the same for [`IndependentFlipError`](@ref).
random_error_pattern(iid_error(0.3,7))

# ## Syndrome Extraction
# First, we define a Tanner graph and a error pattern.
tanner = CSSTannerGraph(SteaneCode())
using Random;Random.seed!(123)
error_pattern = random_error_pattern(iide)

# Then, we can extract the syndrome from the error pattern with [`syndrome_extraction`](@ref).
syndrome = syndrome_extraction(error_pattern, tanner)
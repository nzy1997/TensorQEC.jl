# # Decoder Interface
# ## Decoding Problems
# We have 3 types of decoding problems. [`ClassicalDecodingProblem`](@ref) is for the case that the error model is independent flip error. [`IndependentDepolarizingDecodingProblem`](@ref) is the decoding problem for the case that the error model is independent depolarizing error. [`GeneralDecodingProblem`](@ref) is the decoding problem for the case that error model on different qubits are correlated. We use a tensor network to represent the error probability distribution. Here is an example of how to construct a decoding problem.
using TensorQEC
tanner = CSSTannerGraph(SurfaceCode(3,3))
problem = IndependentDepolarizingDecodingProblem(tanner,iid_error(0.05,0.05,0.05,9))

# ## Decoding Process
# ### Decoders
# There are 2 types of decoders. `AbstractClassicalDecoder` and `AbstractGeneralDecoder` solve [`ClassicalDecodingProblem`](@ref) and [`GeneralDecodingProblem`](@ref), respectively. For [`IndependentDepolarizingDecodingProblem`](@ref), we can transfer it to [`GeneralDecodingProblem`](@ref) and use `AbstractGeneralDecoder` to solve it. Also, for an [`IndependentDepolarizingDecodingProblem`](@ref) of a CSS code, we can transfer it to two [`ClassicalDecodingProblem`](@ref) and use `AbstractClassicalDecoder` to solve them.
# Here is a list of supported decoders.

# | Decoder | API  | Classification|
# |--------------|-------------|---------------------|
# | Minimum Weight Perfect Matching [^Dennis][^Higgott] | [`MatchingDecoder`](@ref) | `AbstractClassicalDecoder` |
# | Belief Propagation [^Yao] | [`BPDecoder`](@ref) |`AbstractClassicalDecoder` |
# | Lookup Table | [`TableDecoder`](@ref) | `AbstractGeneralDecoder` |
# | Integer Programming [^Landahl] | [`IPDecoder`](@ref) | `AbstractGeneralDecoder` |
# | Tensor Network Decoders [^Chubb] | [`TNMAP`](@ref) | `AbstractGeneralDecoder` |

# Here we use the integer programming decoder to solve the decoding problem.
decoder = IPDecoder()

# ### Compilation
# Usually, we use decoders to run monte carlo simulations to estimate the code capacity. We use the same decoder to solve the decoding problem of the same QEC code and error model for different error and syndrome configurations. Ones the error model and the QEC code are fixed, we can compile the decoder to avoid the overhead of decoding. For example, in Lookup Table decoder, we can make the truth table once and use it for all the decoding problems. And in tensor network decoders, we can make the tensor network once and connect it to different syndrome configurations when decoding. This is also the reason why there is no syndrome information in the decoding problem or in the decoder. 
compiled_decoder = compile(decoder, problem);

# ### Decoding
# Now we randomly generate an error configuration.
using Random
Random.seed!(123)
error_qubits = random_error_qubits(problem.pvec)

# The corresponding syndrome is
syndrome = syndrome_extraction(error_qubits, tanner)

# We can use the compiled decoder to decode the syndrome with [`decode`](@ref). The decoding result is a [`DecodingResult`](@ref) object.
res = decode(compiled_decoder, syndrome)

# We can check the result by comparing the syndrome of the decoding result.
syndrome == syndrome_extraction(res.error_qubits, tanner)

# Also, to check wether there is a logical error, we can first generate the logical operators with [`logical_operator`](@ref).
logicalx_operators,logicalz_operators = logical_operator(tanner)

# Then we can check the syndrome of the logical operators with [`check_logical_error`](@ref).
check_logical_error(error_qubits, res.error_qubits, logicalx_operators, logicalz_operators)

# Simply [`decode`](@ref) function can also be used to decode the syndrome. The compilation step is contained in the function.
decode(decoder,problem,syndrome)

# [^Dennis]: Dennis, E.; Kitaev, A.; Landahl, A.; Preskill, J. Topological Quantum Memory. Journal of Mathematical Physics 2002, 43 (9), 4452–4505. https://doi.org/10.1063/1.1499754.

# [^Higgott]: Higgott, O.; Gidney, C. Sparse Blossom: Correcting a Million Errors per Core Second with Minimum-Weight Matching. Quantum 2025, 9, 1600. https://doi.org/10.22331/q-2025-01-20-1600.

# [^Yao]: Yao, H.; Laban, W. A.; Häger, C.; Amat, A. G. i; Pfister, H. D. Belief Propagation Decoding of Quantum LDPC Codes with Guided Decimation. arXiv June 21, 2024. http://arxiv.org/abs/2312.10950 (accessed 2024-10-31).

# [^Landahl]: Landahl, A. J.; Anderson, J. T.; Rice, P. R. Fault-Tolerant Quantum Computing with Color Codes. arXiv August 29, 2011. https://doi.org/10.48550/arXiv.1108.5738.

# [^Chubb]: (1) Chubb, C. T. General Tensor Network Decoding of 2D Pauli Codes. arXiv October 13, 2021. https://doi.org/10.48550/arXiv.2101.04125.

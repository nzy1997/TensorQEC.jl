abstract type Bimatrix end
"""
	CSSBimatrix

Since the encding process may alter the generators of stabilizer group, we introduce the `CSSBimatrix` structure to store the information of encoding process.
The CSSBimatrix structure contains the following fields

* `matrix`: The bimatrix representation of the stabilizers.
* `Q`: The matrix records the Gaussian elimination process, whcih is used to recover the original stabilizers.
* `ordering`: The ordering of qubits.
* `xcodenum`: The number of X stabilizers.

"""
struct CSSBimatrix <: Bimatrix
	matrix::Matrix{Bool}
	Q::Matrix{Mod2}
	ordering::Vector{Int}
	xcodenum::Int
end

struct SimpleBimatrix <: Bimatrix
    matrix::Matrix{Bool}
	Q::Matrix{Mod2}
	ordering::Vector{Int}
end

function SimpleBimatrix(matrix::Matrix{Bool}, ordering::Vector{Int})
	return SimpleBimatrix(matrix, Matrix{Mod2}(I, size(matrix, 1), size(matrix, 1)), ordering)
end

function SimpleBimatrix(matrix::Matrix{Bool})
	return SimpleBimatrix(matrix, collect(1:size(matrix, 2)))
end

Base.copy(b::CSSBimatrix) = CSSBimatrix(copy(b.matrix), copy(b.Q), copy(b.ordering), b.xcodenum)
Yao.nqubits(b::CSSBimatrix) = size(b.matrix, 2) ÷ 2

# pauli string order
# let P = {I, X, Y, Z}, a n-qubit pauli basis is the set of all possible tensor products of elements in P.
# !!! note
#     The order of qubits is following the little-endian convention, i.e. the first qubit is the least significant qubit. For example, `pauli_basis(2)` returns
#     II, XI (in Yao, it is kron(I2, X)), YI, ZI, IX, XX, YX, ZX, IY, XY, YY, ZY, IZ, XZ, YZ, ZZ

# The following pauli strings may in different format, but are the same thing:
# 1. P"IX"
# 2. Yao.kron(I2, X)
# 3. pauli_basis(2)[1,2]
# 4. LinearAlgebra.kron(X,I2)
# 5. IX shown in the print

"""
    pauli_basis(nqubits::Int)
    pauli_basis(::Val{N}) where N

Generate the n-qubit Pauli basis, which is a vector of `PauliString`s. The size of the vector is `4^N`.

## Examples
```jldoctest; setup=:(using TensorQEC)
julia> pauli_basis(1)
4-element Vector{PauliString{1}}:
 I
 X
 Y
 Z
```
The single qubit Pauli basis has 4 elements.
"""
function pauli_basis(::Val{N}) where N
	return [PauliString((ci = pauli_l2c(Val(N), idx); ntuple(i->Pauli(ci[i]-1), Val{N}()))) for idx in 1:4^N]
end
pauli_basis(nqubits::Int) = pauli_basis(Val(nqubits))


"""
    pauli_decomposition(m::AbstractArray; atol=0)
    pauli_decomposition(dm::DensityMatrix; atol=0)
    pauli_decomposition(reg::ArrayReg; atol=0)

Decompose a matrix of size `2^N x 2^N` (or density matrix or array register) into the Pauli basis:
```math
m = \\sum_{i=1}^{4^N} c_i \\cdot \\sigma_i
```
where `c_i` is the coefficient of the `i`-th Pauli string, and `\\sigma_i` is the `i`-th Pauli string.

### Keyword Arguments
- `atol::Float64`: the absolute tolerance for the coefficients. If the coefficient is less than `atol`, it will be ignored.

## Examples
```jldoctest; setup=:(using TensorQEC)
julia> pauli_decomposition(mat(P"X" + 0.5 * P"Z"))
1.0 - 0.0im * X + 0.5 - 0.0im * Z
```
The return value is a [`SumOfPaulis`](@ref) object, which is recovered from the matrix representation of the Pauli expression.
"""
function pauli_decomposition(m::AbstractMatrix; atol=0)
    m = _csc(m)
    N = log2i(size(m, 1))
	coeffs = _pauli_decomposition(m)
    basis = pauli_basis(Val(N))
    return SumOfPaulis([coeffs[i]=>basis[i] for i in 1:length(coeffs) if !isapprox(coeffs[i], 0; atol=atol)])
end

# The return value is a vector of coefficients `c_i`, which is the same length as the Pauli basis (check [`pauli_basis`](@ref)).
function _pauli_decomposition(m::AbstractMatrix{T}) where T
	nqubits = Int(log2(size(m, 1)))
	return [tr(mat(complex(T), pauli) * m) for pauli in pauli_basis(Val(nqubits))] / (2^nqubits)
end

"""
    pauli_repr(m::AbstractMatrix) -> Matrix
    pauli_repr(m::AbstractBlock) -> Matrix

Returns the representation of a matrix in the Pauli basis.
It should not be confused with [`pauli_decomposition`](@ref), this function returns a matrix that represents how a Pauli string is mapped to others by this matrix.
If a matrix has size `2^N x 2^N`, its Pauli basis representation will be a `4^N x 4^N` real matrix.

## Examples
The Pauli operator flips the sign of `Y` and `Z` in the Pauli basis.
```jldoctest; setup=:(using TensorQEC)
julia> pauli_repr(mat(P"X"))
4×4 Matrix{Float64}:
 1.0  0.0   0.0   0.0
 0.0  1.0   0.0   0.0
 0.0  0.0  -1.0   0.0
 0.0  0.0   0.0  -1.0
```
Because we have `XZX = -Z` and `XZY = -Y`.
"""
function pauli_repr(m::AbstractMatrix{T}) where T
    m = _csc(m)
	nqubits = Int(log2(size(m, 1)))
	paulis = pauli_basis(nqubits)
	return [real(tr(mat(complex(T), pi) * m * mat(complex(T), pj) * m'))/size(m, 1) for pi in vec(paulis), pj in vec(paulis)]
end

# all the elements in the Pauli group of size N
function pauli_group(::Val{N}) where N
    return [PauliGroupElement(coeff, (ci = pauli_l2c(Val(N), j); PauliString(ntuple(i->Pauli(ci[i]-1), Val{N}())))) for coeff in 0:3, j in 1:4^N]
end
pauli_group(nqubits::Int) = pauli_group(Val(nqubits))

"""
    pauli_string_map_iter(ps::PauliString{N}, qc::ChainBlock) where N

Map the Pauli string `ps` by a quantum circuit `qc`. Return the mapped Pauli string.
"""
function pauli_string_map_iter(ps::PauliString{N}, qc::ChainBlock) where N
    if length(qc)==0
        return ps
    end
    block = convert_to_put(qc[1])
    return pauli_string_map_iter(map_pauli_string(ps, pauli_repr(block.content),[block.locs...]),qc[2:end])
end
# map a Pauli string to another one, for Clifford simulation
function map_pauli_string(ps::PauliString{N}, paulimapping::Matrix, qubits::Vector{Int}) where N
    # Q: why only get the first non-zero element?!!!
    mapped = _map(Val(length(qubits)), paulimapping, map(k->ps.operators[k], qubits))
    return PauliString(ntuple(k -> k ∈ qubits ? mapped[findfirst(==(k),qubits)] : ps.operators[k], Val{N}()))
end
# check which pauli string to map to
function _map(::Val{N}, paulimapping::Matrix, ps::Vector) where N
    @assert N == length(ps) "`ps` must have the same length as `N`, got $(length(ps)) and $N"
    idx = pauli_c2l(Val(N), map(k->k.id + 1, ps))
    for i = 1:4^N
        if paulimapping[i, idx] != 0
            @assert paulimapping[i, idx] ≈ 1 "`paulimapping` is not a permutation matrix, got $(paulimapping)"
            ci = pauli_l2c(Val(N), i)
            return PauliString(ntuple(j->Pauli(ci[j]-1), Val{N}()))
        end
    end
end

function pauli_l2c(::Val{N}, idx::Integer) where N
    return ntuple(k-> (((idx-1) >> (2*(k-1))) & 3) + 1, Val{N}())
end
function pauli_c2l(::Val{N}, indices) where N
    return sum(k->(indices[k]-1) * 4^(k-1), 1:N; init=1)
end

# YaoAPI
_csc(x) = x
_csc(x::PermMatrix) = PermMatrixCSC(x)
pauli_repr(m::AbstractBlock) = pauli_repr(mat(m))

function pauli_decomposition(dm::DensityMatrix)
    res = pauli_decomposition(dm.state)
    # convert the coefficients to real numbers.
    return SumOfPaulis([real(c)=>p for (c, p) in res.items])
end
pauli_decomposition(reg::ArrayReg) = pauli_decomposition(density_matrix(reg))
pauli_decomposition(block::AbstractBlock) = pauli_decomposition(mat(block))
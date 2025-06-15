# pauli basis
# let P = {I, X, Y, Z}, a n-qubit pauli basis is the set of all possible tensor products of elements in P.
# !!! note
#     The order of qubits is following the little-endian convention, i.e. the first qubit is the least significant qubit. For example, `pauli_basis(2)` returns
#     II, XI (in Yao, it is kron(I2, X)), YI, ZI, IX, XX, YX, ZX, IY, XY, YY, ZY, IZ, XZ, YZ, ZZ

# The following pauli strings may in different format, but are the same thing:
# 1. PauliString(1, 2)
# 2. Yao.kron(I2, X)
# 3. pauli_basis(2)[1,2]
# 4. LinearAlgebra.kron(X,I2)
# 5. X ⊗ I shown in the print

"""
    pauli_basis(nqubits::Int)
    pauli_basis(::Val{N}) where N

Generate the n-qubit Pauli basis.
"""
function pauli_basis(::Val{N}) where N
	return [PauliString(ntuple(i->Pauli(ci[i]-1), Val{N}())) for ci in CartesianIndices(ntuple(_ -> 4, Val{N}()))]
end
pauli_basis(nqubits::Int) = pauli_basis(Val(nqubits))

"""
    pauli_decomposition(m::AbstractMatrix)

Decompose a matrix into the Pauli basis.
Its coefficients are determined by the trace of the product of the matrix and the Pauli basis.
"""
function pauli_decomposition(m::AbstractMatrix{T}) where T
	nqubits = Int(log2(size(m, 1)))
	return [tr(mat(complex(T), pauli) * m) for pauli in pauli_basis(Val(nqubits))] / (2^nqubits)
end
pauli_decomposition(::Type{T}, m::AbstractBlock) where T = pauli_decomposition(mat(T, m))
pauli_decomposition(m::AbstractBlock) = pauli_decomposition(ComplexF64, m)

"""
    pauli_repr_t(m::AbstractMatrix)

Convert a linear operator to a tensor in the Pauli basis.
If a matrix has size `2^N`, its Pauli basis representation will be a real tensor of size `(4, 4, ..., 4)`.
Please also check `pauli_repr` for the matrix representation.
"""
function pauli_repr_t(m::AbstractMatrix{T}) where T
	nqubits = Int(log2(size(m, 1)))
	paulis = pauli_basis(nqubits)
	return [real(tr(mat(complex(T), pi) * m * mat(complex(T), pj) * m'))/size(m, 1) for pi in paulis, pj in paulis]
end

"""
    pauli_repr(m::AbstractMatrix)

Convert a linear operator to a matrix in the Pauli basis.
If a matrix has size `2^N x 2^N`, its Pauli basis representation will be a `4^N x 4^N` real matrix.
Please also check `pauli_repr_t` for the tensor representation.
"""
function pauli_repr(m::AbstractMatrix)
    pmat = pauli_repr_t(m)
    return reshape(pmat, (size(m) .^ 2)...)
end

# all the elements in the Pauli group of size N
function pauli_group(::Val{N}) where N
    return [PauliGroupElement(coeff, PauliString(ntuple(i->Pauli(ci[i]-1), Val{N}()))) for coeff in 0:3, ci in CartesianIndices(ntuple(_ -> 4, Val{N}()))]
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
    return pauli_string_map_iter(map_pauli_string(ps, pauli_repr_t(mat(ComplexF64,block.content)),[block.locs...]),qc[2:end])
end
# map a Pauli string to another one, for Clifford simulation
function map_pauli_string(ps::PauliString{N}, paulimapping::Array, qubits::Vector{Int}) where N
    # Q: why only get the first non-zero element?!!!
    c = findall(!iszero, paulimapping[fill(:,length(size(paulimapping)) ÷ 2)..., map(k->ps.operators[k].id + 1, qubits)...])[1]
    return PauliString(([Pauli(k ∈ qubits ? c[findfirst(==(k),qubits)]-1 : ps.operators[k].id) for k in 1:N]...,))
end

# YaoAPI
pauli_repr_t(::Type{T}, m::AbstractBlock) where T = pauli_repr_t(mat(T, m))
pauli_repr_t(m::AbstractBlock) = pauli_repr_t(ComplexF64, m)
pauli_repr(m::AbstractBlock) = pauli_repr(mat(m))
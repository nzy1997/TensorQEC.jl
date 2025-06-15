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

Generate the n-qubit Pauli basis.
"""
function pauli_basis(nqubits::Int)
	return [PauliString(ntuple(i->Pauli(ci[i]-1), nqubits)) for ci in CartesianIndices(ntuple(_ -> 4, nqubits))]
end

"""
    pauli_decomposition(m::AbstractMatrix)

Decompose a matrix into the Pauli basis.
"""
function pauli_decomposition(m::AbstractMatrix{T}) where T
	nqubits = Int(log2(size(m, 1)))
	return [tr(mat(complex(T), pauli) * m) for pauli in pauli_basis(nqubits)] / (2^nqubits)
end
pauli_decomposition(::Type{T}, m::AbstractBlock) where T = pauli_decomposition(mat(T, m))
pauli_decomposition(m::AbstractBlock) = pauli_decomposition(ComplexF64, m)

"""
    pauli_mapping(m::AbstractMatrix)

Convert a linear operator to a matrix in the Pauli basis.
"""
function pauli_mapping(m::AbstractMatrix)
	nqubits = Int(log2(size(m, 1)))
	paulis = pauli_basis(nqubits)
	return [real(tr(mat(pi) * m * mat(pj) * m')/size(m, 1)) for pi in paulis, pj in paulis]
end
pauli_mapping(::Type{T}, m::AbstractBlock) where T = pauli_mapping(mat(T, m))
pauli_mapping(m::AbstractBlock) = pauli_mapping(ComplexF64, m)

function pauli_group(n::Int)
    return [coeff => PauliString(ntuple(i->Pauli(ci[i]-1), n)) for coeff in 0:3, ci in CartesianIndices(ntuple(_ -> 4, n))]
end

pauli_repr(m::AbstractBlock) = pauli_repr(mat(m))
function pauli_repr(m::AbstractMatrix)
    pmat = pauli_mapping(m)
    return reshape(pmat, (size(m) .^2)...)
end

"""
    pauli_string_map_iter(ps::PauliString{N}, qc::ChainBlock) where N

Map the Pauli string `ps` by a quantum circuit `qc`. Return the mapped Pauli string.
"""
function pauli_string_map_iter(ps::PauliString{N}, qc::ChainBlock) where N
    if length(qc)==0
        return ps
    end
    block=convert_to_put(qc[1])
    return pauli_string_map_iter(pauli_string_map(ps, pauli_mapping(mat(ComplexF64,block.content)),[block.locs...]),qc[2:end])
end
function pauli_string_map(ps::PauliString{N}, paulimapping::Array, qubits::Vector{Int}) where N
    c = findall(!iszero, paulimapping[fill(:,length(size(paulimapping)) ÷ 2)..., map(k->ps.operators[k].id + 1, qubits)...])[1]
    return PauliString(([Pauli(k ∈ qubits ? c[findfirst(==(k),qubits)]-1 : ps.operators[k].id) for k in 1:N]...,))
end

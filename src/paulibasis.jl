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
function pauli_basis(nqubits::Int)
	return [PauliString{nqubits}(ci.I) for ci in CartesianIndices(ntuple(_ -> 4, nqubits))]
end

# pauli decomposition of a matrix
# returns the coefficients in pauli basis
function pauli_decomposition(m::AbstractMatrix)
	nqubits = Int(log2(size(m, 1)))
	return [tr(mat(pauli) * m) for pauli in pauli_basis(nqubits)] / (2^nqubits)
end

# defined the linear mapping in the pauli basis
# from the coding space to the physical qubits, i.e. (physical..., coding...) or (output..., input...)
function pauli_mapping(m::AbstractMatrix)
	nqubits = Int(log2(size(m, 1)))
	paulis = pauli_basis(nqubits)
	return [real(tr(mat(pi) * m * mat(pj) * m')/size(m, 1)) for pi in paulis, pj in paulis]
end

function pauli_group(n::Int)
    return [coeff => PauliString(ci.I) for coeff in 0:3, ci in CartesianIndices(ntuple(_ -> 4, n))]
end

pauli_repr(m::AbstractBlock) = pauli_repr(mat(m))
function pauli_repr(m::AbstractMatrix)
    pmat = pauli_mapping(m)
    return reshape(pmat, (size(m) .^2)...)
end

function pauli_string_map_iter(ps::PauliString{N}, qc::ChainBlock) where N
       if length(qc)==0
               return ps
       end
       block=convert_to_put(qc[1])
       return pauli_string_map_iter(pauli_string_map(ps,pauli_mapping(mat(ComplexF64,block.content)),[block.locs...]),qc[2:end])
end
function pauli_string_map(ps::PauliString{N}, paulimapping::Array, qubits::Vector{Int}) where N
    c=findall(!iszero, paulimapping[fill(:,length(size(paulimapping)) ÷ 2)...,ps.ids[qubits]...])[1]
    return PauliString(([k ∈ qubits ? c[findfirst(==(k),qubits)] : ps.ids[k] for k in 1:N]...,))
end
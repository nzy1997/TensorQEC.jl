# pauli basis
# let P = {I, X, Y, Z}, a n-qubit pauli basis is the set of all possible tensor products of elements in P.
function pauli_basis(nqubits::Int)
	paulis = Matrix.([I2, X, Y, Z])
	return [(length(pauli) == 1 ? pauli[1] : kron(pauli...)) for pauli in product(fill(paulis, nqubits)...)]
end

# pauli decomposition of a matrix
# returns the coefficients in pauli basis
function pauli_decomposition(m::AbstractMatrix)
	nqubits = Int(log2(size(m, 1)))
	return [tr(pauli * m) for pauli in pauli_basis(nqubits)] / (2^nqubits)
end

# defined the linear mapping in the pauli basis
# from the coding space to the physical qubits, i.e. (physical..., coding...)
function pauli_mapping(m::AbstractMatrix)
	nqubits = Int(log2(size(m, 1)))
	paulis = pauli_basis(nqubits)
	return [real(tr(pi * m * pj * m')/size(m, 1)) for pi in paulis, pj in paulis]
end
# pauli basis
# let P = {I, X, Y, Z}, a n-qubit pauli basis is the set of all possible tensor products of elements in P.
# !!! note
#     The order of qubits is following the little-endian convention, i.e. the first qubit is the least significant qubit. For example, `pauli_basis(2)` returns
#     II, XI (in Yao, it is kron(I2, X)), YI, ZI, IX, XX, YX, ZX, IY, XY, YY, ZY, IZ, XZ, YZ, ZZ
function pauli_basis(nqubits::Int)
	paulis = [I2, X, Y, Z]
	return [Matrix{ComplexF64}(kron(pauli...)) for pauli in product(fill(paulis, nqubits)...)]
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

struct PauliString{N} <: CompositeBlock{2}
    ids::NTuple{N, Int}
end
function Yao.YaoBlocks.unsafe_apply!(reg::AbstractRegister, p::PauliString{N}) where N
	Yao.YaoBlocks.unsafe_apply!(reg, kron([I2, X, Y, Z][q] for q in p.ids))
end
Yao.nqudits(::PauliString{N}) where N = N
Yao.mat(::Type{T}, p::PauliString{N}) where {T, N} = mat(T, kron([I2, X, Y, Z][q] for q in p.ids))
Yao.subblocks(p::PauliString) = (([I2, X, Y, Z][q] for q in p.ids)...,)
Base.show(io::IO, ::MIME"text/plain", sp::PauliString) = show(io, sp)
function Base.show(io::IO, ps::PauliString)
	for (i, q) in enumerate(ps.ids[end:-1:1])
		print(io, "IXYZ"[q])
		if i < length(ps.ids)
			print(io, " ⊗ ")
		end
	end
end
struct SumOfPaulis{T<:Number, N} <: Yao.CompositeBlock{2}
	items::Vector{Pair{T, PauliString{N}}}
end
Base.show(io::IO, ::MIME"text/plain", sp::SumOfPaulis) = show(io, sp)
function Base.show(io::IO, sp::SumOfPaulis)
	for (k, (c, p)) in enumerate(sp.items)
		print(io, c, " * ", p)
		k != length(sp.items) && println(io)
	end
end
function tensor2sumofpaulis(t::AbstractArray)
	SumOfPaulis([t[ci]=>PauliString(ci.I) for ci in CartesianIndices(t)] |> vec)
end
Yao.mat(::Type{T}, sp::SumOfPaulis) where T = mat(T, sum([c * kron([I2, X, Y, Z][q] for q in p.ids) for (c, p) in sp.items]))
Yao.subblocks(x::SumOfPaulis) = ((x.second for x in x.items)...,)

densitymatrix2sumofpaulis(dm::DensityMatrix) = tensor2sumofpaulis(real.(pauli_decomposition(dm.state)))
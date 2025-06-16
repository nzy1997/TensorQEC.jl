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

# input: a vector of PauliString objects
# output: a CSSBimatrix object
function stabilizers2bimatrix(stabilizers::AbstractVector{PauliString{N}}) where N
	xs = findall(s -> all(x -> x == Pauli(0) || x == Pauli(1), s), stabilizers)
	zs = findall(s -> all(x -> x == Pauli(0) || x == Pauli(3), s), stabilizers)
	@assert length(xs) + length(zs) == length(stabilizers) "Invalid PauliString"
	A = [stabilizers[xs[i]][j] == Pauli(1) for i ∈ 1:length(xs), j ∈ 1:N]
	B = [stabilizers[zs[i]][j] == Pauli(3) for i ∈ 1:length(zs), j ∈ 1:N]
	return CSSBimatrix(cat(A, B; dims = (1, 2)), Matrix{Mod2}(I, length(stabilizers), length(stabilizers)), collect(1:N), length(xs))
end

function bimatrix2stabilizers(bimat::CSSBimatrix)
	n = Yao.nqubits(bimat)
	xs = [PauliString(n, bimat.ordering[findall(isone, bimat.matrix[i, 1:n])] => Pauli(1)) for i in 1:bimat.xcodenum]
	zs = [PauliString(n, bimat.ordering[findall(isone, bimat.matrix[i, n+1:end])] => Pauli(3)) for i in bimat.xcodenum+1:size(bimat.matrix, 1)]
	return vcat(xs, zs)
end

function encode_circuit(bimat::CSSBimatrix)
	qubit_num = size(bimat.matrix, 2) ÷ 2
	qc = chain(qubit_num)
	#stage 1: H gates and Z stabilizers
	for i in 1:bimat.xcodenum
		push!(qc, put(qubit_num, bimat.ordering[i] => H))
	end
	for i in bimat.xcodenum+1:size(bimat.matrix, 1), j in qubit_num+size(bimat.matrix, 1)+1:2*qubit_num
		if bimat.matrix[i, j]
			push!(qc, cnot(qubit_num, bimat.ordering[j-qubit_num], bimat.ordering[i]))
		end
	end

	#stage 2: X stabilizers
	for i in 1:bimat.xcodenum, j in bimat.xcodenum+1:qubit_num
		if bimat.matrix[i, j]
			push!(qc, cnot(qubit_num, bimat.ordering[i], bimat.ordering[j]))
		end
	end
	return qc
end

"""
	encode_stabilizers(stabilizers::AbstractVector{PauliString{N}}) where N

Generate the encoding circuit for the given stabilizers.

### Arguments
- `stabilizers`: The vector of pauli strings, composing the generator of stabilizer group.

### Returns
- `qc`: The encoding circuit.
- `data_qubits`: The indices of data qubits.
- `bimat`: The structure storing the encoding information.
"""
function encode_stabilizers(stabilizers::AbstractVector{PauliString{N}}) where N
	bimat = stabilizers2bimatrix(stabilizers)
	gaussian_elimination!(bimat)
	qc = encode_circuit(bimat)
	data_qubits = bimat.ordering[size(bimat.matrix, 1)+1:end]
	return qc, data_qubits, bimat
end

"""
	place_qubits(reg0::AbstractRegister, data_qubits::Vector{Int}, num_qubits::Int)

Place the data qubits to the specified position. The other qubits are filled with zero state.

### Arguments
- `reg0`: The data register.
- `data_qubits`: The indices of data qubits.
- `num_qubits`: The total number of qubits.

### Returns
- `reg`: The register with data qubits placed at the specified position.
"""
function place_qubits(reg0::AbstractRegister, data_qubits::Vector{Int}, num_qubits::Int)
	@assert nqubits(reg0) == length(data_qubits)
	reg = join(reg0,zero_state(num_qubits-length(data_qubits)))
	order = collect(1:num_qubits)
	for i in 1:length(data_qubits)
		order[data_qubits[i]] = num_qubits+1-i
		order[num_qubits+1-i] = data_qubits[i]
	end
	reorder!(reg, (order...,))
	return reg
end


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

Base.copy(b::CSSBimatrix) = CSSBimatrix(copy(b.matrix), copy(b.Q), copy(b.ordering), b.xcodenum)
Yao.nqubits(b::CSSBimatrix) = size(b.matrix, 2) ÷ 2

# input: a vector of PauliString objects
# output: a CSSBimatrix object
function stabilizers2bimatrix(stabilizers::AbstractVector{PauliString{N}}) where N
	xs = findall(s -> all(x -> x == 1 || x == 2, s.ids), stabilizers)
	zs = findall(s -> all(x -> x == 1 || x == 4, s.ids), stabilizers)
	@assert length(xs) + length(zs) == length(stabilizers) "Invalid PauliString"
	A = [stabilizers[xs[i]].ids[j] == 2 for i ∈ 1:length(xs), j ∈ 1:N]
	B = [stabilizers[zs[i]].ids[j] == 4 for i ∈ 1:length(zs), j ∈ 1:N]
	return CSSBimatrix(cat(A, B; dims = (1, 2)), Matrix{Mod2}(I, length(stabilizers), length(stabilizers)), collect(1:N), length(xs))
end

function bimatrix2stabilizers(bimat::CSSBimatrix)
	n = Yao.nqubits(bimat)
	xs = [paulistring(n, 2, bimat.ordering[findall(isone, bimat.matrix[i, 1:n])]) for i in 1:bimat.xcodenum]
	zs = [paulistring(n, 4, bimat.ordering[findall(isone, bimat.matrix[i, n+1:end])]) for i in bimat.xcodenum+1:size(bimat.matrix, 1)]
	return vcat(xs, zs)
end

function switch_qubits!(bimat::CSSBimatrix, i::Int, j::Int)
	qubit_num = size(bimat.matrix, 2) ÷ 2
	bimat.matrix[:, [i, j]] = bimat.matrix[:, [j, i]]
	(bimat.matrix[:, [qubit_num + i, qubit_num + j]] = bimat.matrix[:, [qubit_num + j, qubit_num + i]])
	bimat.ordering[i], bimat.ordering[j] = bimat.ordering[j], bimat.ordering[i]
	return bimat
end

function switch_qubits!(bimat::SimpleBimatrix, i::Int, j::Int)
	bimat.matrix[:, [i, j]] = bimat.matrix[:, [j, i]]
	bimat.ordering[i], bimat.ordering[j] = bimat.ordering[j], bimat.ordering[i]
	return bimat
end

function make_corner_non_zero!(bimat,offset,start_col)
	j = findfirst(!iszero, bimat.matrix[i, start_col:end])
	j === nothing && (zero_row += 1; continue)
	switch_qubits!(bimat, qubit_offset + offset + 1, j + qubit_offset)
end

function gaussian_elimination!(bimat::Bimatrix, rows::UnitRange, col_offset::Int, qubit_offset::Int)
	start_col = col_offset + qubit_offset + 1
	zero_row = 0
	for i in rows
		Q = Matrix{Mod2}(I, length(rows), length(rows))
		offset = i - rows.start -zero_row
		j = findfirst(!iszero, bimat.matrix[i, start_col:end])
		j === nothing && (zero_row += 1; continue)
		switch_qubits!(bimat, qubit_offset + offset + 1, j + qubit_offset)
		for k in rows
			if k != i && bimat.matrix[k, offset+start_col]  # got 1, eliminate by ⊻ current row (i) to the k-th row
				bimat.matrix[k, :] .= xor.(bimat.matrix[k, :], bimat.matrix[i, :])
				Q[k-rows.start+1, i-rows.start+1] = true
			end
		end
		bimat.Q[rows, :] .= Q * bimat.Q[rows, :]
	end
	bimat
end
function gaussian_elimination!(bimat::CSSBimatrix)
	qubit_num = size(bimat.matrix, 2) ÷ 2
	gaussian_elimination!(bimat, 1:bimat.xcodenum, 0, 0)
	gaussian_elimination!(bimat, bimat.xcodenum+1:size(bimat.matrix, 1), qubit_num, bimat.xcodenum)
	return bimat
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


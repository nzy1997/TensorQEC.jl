struct Bimatrix
	matrix::Matrix{Bool}
	Q::Matrix{Mod2}
	ordering::Vector{Int}
	xcodenum::Int
end
Base.copy(b::Bimatrix) = Bimatrix(copy(b.matrix), copy(b.Q), copy(b.ordering), b.xcodenum)
Yao.nqubits(b::Bimatrix) = size(b.matrix, 2) ÷ 2

# input: a vector of PauliString objects
# output: a Bimatrix object
function stabilizers2bimatrix(stabilizers::AbstractVector{PauliString{N}}) where N
	xs = findall(s -> all(x -> x == 1 || x == 2, s.ids), stabilizers)
	zs = findall(s -> all(x -> x == 1 || x == 4, s.ids), stabilizers)
	@assert length(xs) + length(zs) == length(stabilizers) "Invalid PauliString"
	A = [stabilizers[xs[i]].ids[j] == 2 for i ∈ 1:length(xs), j ∈ 1:N]
	B = [stabilizers[zs[i]].ids[j] == 4 for i ∈ 1:length(zs), j ∈ 1:N]
	return Bimatrix(cat(A, B; dims = (1, 2)), Matrix{Mod2}(I, length(stabilizers), length(stabilizers)), collect(1:N), length(xs))
end

function bimatrix2stabilizers(bimat::Bimatrix)
	n = Yao.nqubits(bimat)
	xs = [paulistring(n, 2, bimat.ordering[findall(isone, bimat.matrix[i, 1:n])]) for i in 1:bimat.xcodenum]
	zs = [paulistring(n, 4, bimat.ordering[findall(isone, bimat.matrix[i, n+1:end])]) for i in bimat.xcodenum+1:size(bimat.matrix, 1)]
	return vcat(xs, zs)
end
function switch_qubits!(bimat::Bimatrix, i::Int, j::Int)
	qubit_num = size(bimat.matrix, 2) ÷ 2
	bimat.matrix[:, [i, j]] = bimat.matrix[:, [j, i]]
	bimat.matrix[:, [qubit_num + i, qubit_num + j]] = bimat.matrix[:, [qubit_num + j, qubit_num + i]]
	bimat.ordering[i], bimat.ordering[j] = bimat.ordering[j], bimat.ordering[i]
	return bimat
end
function gaussian_elimination!(bimat::Bimatrix, rows::UnitRange, col_offset::Int, qubit_offset::Int)
	start_col = col_offset + qubit_offset + 1
	for i in rows
		Q = Matrix{Mod2}(I, length(rows), length(rows))
		offset = i - rows.start
		j = findfirst(!iszero, bimat.matrix[i, start_col:end])
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
function gaussian_elimination!(bimat::Bimatrix)
	qubit_num = size(bimat.matrix, 2) ÷ 2
	gaussian_elimination!(bimat, 1:bimat.xcodenum, 0, 0)
	gaussian_elimination!(bimat, bimat.xcodenum+1:size(bimat.matrix, 1), qubit_num, bimat.xcodenum)
	return bimat
end

function encode_circuit(bimat::Bimatrix)
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
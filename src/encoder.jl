struct Bimatrix
	matrix::Matrix{Bool}
	Q::Matrix{Mod2}
	ordering::Vector{Int}
	xcodenum::Int
end
Base.copy(b::Bimatrix) = Bimatrix(copy(b.matrix), copy(b.Q), copy(b.ordering), b.xcodenum)
Yao.nqubits(b::Bimatrix) = size(b.matrix, 2) ÷ 2

# simple toric code
struct ToricCode
	m::Int
	n::Int
end

nsite(t::ToricCode) = t.m * t.n
Yao.nqubits(t::ToricCode) = 2 * nsite(t)
vertical_edges(t::ToricCode) = reshape(1:nsite(t), t.m, t.n)
horizontal_edges(t::ToricCode) = reshape(nsite(t)+1:2*nsite(t), t.m, t.n)

# Toric code (2*2)
# ∘---1---∘---2---∘
# |       |       |
# 5   ∘   6   ∘   5 
# |       |       |
# ∘---3---∘---4---∘
# |       |       | 
# 7   ∘   8   ∘   7
# |       |       |
# ∘---1---∘---2---∘

function stabilizers(toric::ToricCode; linearly_independent::Bool = true)
	nq, m, n = nqubits(toric), toric.m, toric.n
	output = PauliString{nq}[]
	# numbering the qubits
	ve = vertical_edges(toric)
	he = horizontal_edges(toric)
	# X stabilizers
	for j in 1:n, i in 1:m
		i == m && j == n && linearly_independent && continue # not linearly independent
		push!(output, paulistring(nq, 2, (
			he[i, j], he[mod1(i + 1, m), j],
			ve[i, j], ve[i, mod1(j + 1, n)],
		)))
	end
	# Z stabilizers
	for j in 1:n, i in 1:m
		i == m && j == n && linearly_independent && continue # not linearly independent
		push!(output, paulistring(nq, 4, (
			ve[i, j], ve[mod1(i - 1, m), j],
			he[i, j], he[i, mod1(j - 1, n)],
		)))
	end
	return output
end

# inputs:
# - n is the size of the toric code
# - k is the Pauli operator, 1 for I, 2 for X, 3 for Y, 4 for Z
# - ids is a vector of qubit ids
# output: a PauliString object
paulistring(n, k, ids) = PauliString((i ∈ ids ? k : 1 for i in 1:n)...)



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

# Surface code, m is the number of rows, n is the number of columns
struct SurfaceCode{m,n} end

# Surface code (3*3)
#       -
#     / z \
#     1---2---3 -\
#     | x | z | x |
#  /- 4---5- -6 -/
# | x | z | x |
#  \- 7---8---9
#		  \ z /
#			-
# 8 stableizers:
# Z type: 12, 2356, 4578, 89
# X type: 36, 1245, 5689, 47

function stabilizers(::SurfaceCode{m,n}) where {m,n}
	qubit_config = reshape(1:m*n, n, m)' 
	pauli_string = PauliString{m*n}[]
	for i in 1:m-1, j in 1:n-1
		if mod(i+j, 2) == 0
			push!(pauli_string, paulistring(m*n, 2, (qubit_config[i, j], qubit_config[i+1, j], qubit_config[i, j+1], qubit_config[i+1, j+1])))
		end
	end
	for i in 1:m÷2
		push!(pauli_string, paulistring(m*n, 2, (qubit_config[2*i-1+mod(n+1,2), n], qubit_config[2*i+mod(n+1,2), n])))
		if 2*i+1 <= m
			push!(pauli_string, paulistring(m*n, 2, (qubit_config[2*i, 1], qubit_config[2*i+1, 1])))
		end
		@show pauli_string[end].ids
	end
	for i in 1:m-1, j in 1:n-1
		if mod(i+j, 2) == 1
			push!(pauli_string, paulistring(m*n, 4, (qubit_config[i, j], qubit_config[i+1, j], qubit_config[i, j+1], qubit_config[i+1, j+1])))
		end
	end
	for j in 1:n÷2
		push!(pauli_string, paulistring(m*n, 4, (qubit_config[1, 2*j-1], qubit_config[1, 2*j])))
		if 2*j+1 <= n
			push!(pauli_string, paulistring(m*n, 4, (qubit_config[m, 2*j-1+mod(m,2)], qubit_config[m, 2*j+mod(m,2)])))
		end
	end
	@show qubit_config
	return pauli_string
end

function encode_stabilizers(stabilizers::AbstractVector{PauliString{N}}) where N
	bimat = stabilizers2bimatrix(stabilizers)
	gaussian_elimination!(bimat)
	qc = encode_circuit(bimat)
	data_qubits = bimat.ordering[size(bimat.matrix, 1)+1:end]
	return qc, data_qubits, bimat
end


struct ShorCode end

function stabilizers(::ShorCode; linearly_independent::Bool = true)
	nq = 9
	pauli_string = PauliString{nq}[]
	push!(pauli_string, paulistring(nq, 2, (1,2,3,4,5,6)))
	push!(pauli_string, paulistring(nq, 2, (1, 2, 3, 7,8,9)))
	linearly_independent || push!(pauli_string,paulistring(nq, 2, (4, 5, 6, 7, 8, 9)))
	push!(pauli_string, paulistring(nq, 4, (1, 2)))
	push!(pauli_string, paulistring(nq, 4, (1, 3)))
	linearly_independent || push!(pauli_string, paulistring(nq, 4, (2, 3)))
	push!(pauli_string, paulistring(nq, 4, (4, 5)))
	push!(pauli_string, paulistring(nq, 4, (4,6)))
	linearly_independent || push!(pauli_string, paulistring(nq, 4, (5, 6)))
	push!(pauli_string, paulistring(nq, 4, (7,8)))
	push!(pauli_string, paulistring(nq, 4, (7, 9)))
	linearly_independent || push!(pauli_string, paulistring(nq, 4, (8, 9)))
	return pauli_string
end


struct SteaneCode end

function stabilizers(::SteaneCode)
	nq = 7
	pauli_string = PauliString{nq}[]
	push!(pauli_string, paulistring(nq, 2, (1,3,5,7)))
	push!(pauli_string, paulistring(nq, 2, (2,3,6,7)))
	push!(pauli_string, paulistring(nq, 2, (4,5,6,7)))
	push!(pauli_string, paulistring(nq, 4, (1,3,5,7)))
	push!(pauli_string, paulistring(nq, 4, (2,3,6,7)))
	push!(pauli_string, paulistring(nq, 4, (4,5,6,7)))
	return pauli_string
end
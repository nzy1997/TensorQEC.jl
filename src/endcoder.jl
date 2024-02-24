struct Bimatrix
	matrix::Matrix{Bool}
	xcodenum::Int
	zcodenum::Int
end

# input: n is the size of the toric code
# output: a vector of PauliString objects
function toric_code(m::Int, n::Int)
    output = PauliString{2*m*n}[]
	vertical_edges = reshape(1:m*n, m, n)
	horizontal_edges = reshape(m*n+1:2*m*n, n, n)
    # X stabilizers
	for j in 1:n
	    for i in 1:m
			i == m && j == n && continue # not linearly independent
			push!(output, paulistring(2*m*n, 2, (horizontal_edges[i, j], 
                horizontal_edges[mod1(i + 1, n), j],
                vertical_edges[i, j],
                vertical_edges[i, mod1(j + 1, n)]
                )))
		end
	end
end

# inputs:
# - n is the size of the toric code
# - k is the Pauli operator, 1 for I, 2 for X, 3 for Y, 4 for Z
# - ids is a vector of qubit ids
# output: a PauliString object
paulistring(n, k, ids) = PauliString((i ∈ ids ? k : 1 for i in 1:n)...)



# input: n is the size of the toric code
# output: a Bimatrix object
function toric_code2bimatrix(n::Int)
	xcodenum = 0
	zcodenum = 0
	matrix = falses(2 * n^2 - 2, 4 * n^2)
	vertical_edges = reshape(1:n^2, n, n)
	horizontal_edges = reshape(n^2+1:2*n^2, n, n)
	for j in 1:n
	    for i in 1:n
			if i == n && j == n
				break
			end
			xcodenum += 1
			matrix[xcodenum, horizontal_edges[i, j]] = true
			matrix[xcodenum, horizontal_edges[mod1(i + 1, n), j]] = true
			matrix[xcodenum, vertical_edges[i, j]] = true
			matrix[xcodenum, vertical_edges[i, mod1(j + 1, n)]] = true
		end
	end

	for i in 1:n
		for j in 1:n
			if i == n && j == n
				break
			end
			zcodenum += 1
			matrix[zcodenum+xcodenum, vertical_edges[i, j]+2*n^2] = true
			matrix[zcodenum+xcodenum, vertical_edges[mod1(i - 1, n), j]+2*n^2] = true
			matrix[zcodenum+xcodenum, horizontal_edges[i, j]+2*n^2] = true
			matrix[zcodenum+xcodenum, horizontal_edges[i, mod1(j - 1, n)]+2*n^2] = true
		end
	end
	return Bimatrix(matrix, xcodenum, zcodenum)
end

function switch_qubits!(bimat::Bimatrix, i::Int, j::Int)
    qubit_num = Int(size(bimat.matrix, 2) / 2)
    bimat.matrix[:, [i, j]] = bimat.matrix[:, [j, i]]
    if i <= qubit_num
        bimat.matrix[:, [qubit_num + i, qubit_num + j]] = bimat.matrix[:, [qubit_num + j, qubit_num + i]]
    else
        bimat.matrix[:, [i - qubit_num, j - qubit_num]] = bimat.matrix[:, [j - qubit_num, i - qubit_num]]
    end
    return nothing
end
function guassian_elimination!(bimat::Bimatrix, start_row::Int, start_col::Int, end_row::Int)
    for i in start_row:end_row
        j = findfirst(!iszero, bimat.matrix[i, start_col:end])+start_col-1
        switch_qubits!(bimat, i-start_row+start_col, j)
		for k in start_row:end_row
			if k == i
				continue
			end
			if bimat.matrix[k, i-start_row+start_col]
				bimat.matrix[k, :] .= xor.(bimat.matrix[k, :], bimat.matrix[i, :])
			end
		end
	end
    return nothing
end
function guassian_elimination!(bimat::Bimatrix)
	qubit_num = Int(size(bimat.matrix, 2) / 2)
    guassian_elimination!(bimat,1,1, bimat.xcodenum)
    guassian_elimination!(bimat,bimat.xcodenum+1,qubit_num+bimat.xcodenum+1,bimat.xcodenum+bimat.zcodenum)
	return nothing
end

function encode_circuit(bimat::Bimatrix)


	return qc
end

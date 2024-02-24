struct Bimatrix
	matrix::Matrix{Bool}
	xcodenum::Int
	zcodenum::Int
end

# simple toric code
struct ToricCode
    m::Int
    n::Int
end
nsite(t::ToricCode) = t.m*t.n
Yao.nqubits(t::ToricCode) = 2*nsite(t)
vertical_edges(t::ToricCode) = reshape(1:nsite(t), t.m, t.n)
horizontal_edges(t::ToricCode) = reshape(nsite(t)+1:2*nsite(t), t.m, t.n)

# input: n is the size of the toric code
# output: a vector of PauliString objects
function stabilizers(toric::ToricCode)
    nq, m, n = nqubits(toric), toric.m, toric.n
    output = PauliString{nq}[]
    # numbering the qubits
    ve = vertical_edges(toric)
    he = horizontal_edges(toric)
    # X stabilizers
	for j in 1:n, i in 1:m
        i == m && j == n && continue # not linearly independent
        push!(output, paulistring(nq, 2, (
            he[i, j], he[mod1(i + 1, m), j],
            ve[i, j], ve[i, mod1(j + 1, n)]
        )))
	end
    # Z stabilizers
	for j in 1:n, i in 1:m
        i == m && j == n && continue # not linearly independent
        push!(output, paulistring(nq, 4, (
            ve[i, j], ve[mod1(i - 1, m), j],
            he[i, j], he[i, mod1(j - 1, n)]
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

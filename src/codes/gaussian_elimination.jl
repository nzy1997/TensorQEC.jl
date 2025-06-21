function switch_qubits!(bimat::CSSBimatrix, i::Int, j::Int)
	qubit_num = size(bimat.matrix, 2) ÷ 2
	for row in axes(bimat.matrix, 1)
		bimat.matrix[row, [i, j]] = bimat.matrix[row, [j, i]]
	end
	for row in axes(bimat.matrix, 1)
		bimat.matrix[row, [qubit_num + i, qubit_num + j]] = bimat.matrix[row, [qubit_num + j, qubit_num + i]]
	end
	bimat.ordering[i], bimat.ordering[j] = bimat.ordering[j], bimat.ordering[i]
	return bimat
end

function switch_qubits!(bimat::SimpleBimatrix, i::Int, j::Int)
    for row in axes(bimat.matrix, 1)
	    bimat.matrix[row, i], bimat.matrix[row, j] = bimat.matrix[row, j], bimat.matrix[row, i]
    end
	bimat.ordering[i], bimat.ordering[j] = bimat.ordering[j], bimat.ordering[i]
	return bimat
end

"""
    gaussian_elimination!(bimat::Bimatrix, rows::UnitRange, col_offset::Int, qubit_offset::Int;allow_col_operation = true)

Perform Gaussian elimination on a binary matrix.

### Arguments
- `bimat`: The binary matrix to perform Gaussian elimination on.
- `rows`: The range of rows to consider in the elimination process.
- `col_offset`: The offset for columns in the matrix.
- `qubit_offset`: The offset for qubits in the ordering.
- `allow_col_operation`: Whether column operations (qubit swapping) are allowed. Default is true.

### Returns
- `bimat`: The modified binary matrix after Gaussian elimination.
"""
function gaussian_elimination!(bimat::Bimatrix, rows::UnitRange, col_offset::Int, qubit_offset::Int;allow_col_operation = true)
    Q, matrix = Matrix(bimat.Q'), Matrix(bimat.matrix')  # convert to column major
    # Calculate the starting column for elimination
    start_col = col_offset + qubit_offset + 1
    zero_row = 0  # Counter for rows that are all zeros
    zero_col = 0  # Counter for columns that are all zeros
    
    for i in rows
        if allow_col_operation
            # Column operations allowed - we can swap qubits
            offset = i - rows.start - zero_row
            # Find the first non-zero element in the current row
            j = findfirst(!iszero, view(bimat.matrix, i, start_col:size(bimat.matrix, 2)))
            
            # If the row is all zeros, skip it and increment zero_row counter
            j === nothing && (zero_row += 1; continue)
            
            # Swap qubits to bring the non-zero element to the diagonal position
            switch_qubits!(bimat, qubit_offset + offset + 1, j + qubit_offset)
        else
            # Column operations not allowed - we can only swap rows
            # Check if we've reached the end of the matrix
            i + zero_col > size(bimat.matrix, 2) && return bimat
            
            # Find the first non-zero element in the current column (starting from current row)
            j = findfirst(!iszero, view(bimat.matrix, i:size(bimat.matrix, 1), i + zero_col))
            
            # If the column is all zeros, move to the next column
            while j === nothing
                zero_col += 1
                i + zero_col > size(bimat.matrix, 2) && return bimat
                j = findfirst(!iszero, view(bimat.matrix, i:size(bimat.matrix, 1), i + zero_col))
            end
            
            j = i+j-1  # Calculate the actual row index
            
            # If the non-zero element is not in the current row, swap rows
            if j != i
                for col in axes(bimat.Q, 2)
                    # swap rows (i, j) in Q
                    bimat.Q[i, col], bimat.Q[j, col] = bimat.Q[j, col], bimat.Q[i, col]
                end
                for col in axes(bimat.matrix, 2)
                    # swap rows (i, j) in matrix
                    bimat.matrix[i, col], bimat.matrix[j, col] = bimat.matrix[j, col], bimat.matrix[i, col]
                end
            end
            
            offset = i - rows.start + zero_col
        end
        
        # Eliminate all other rows using the current pivot row
        for k in rows
            # If the element in the same column of another row is 1, XOR the rows
            if k != i && bimat.matrix[k, offset+start_col]  # got 1, eliminate by ⊻ current row (i) to the k-th row
                for col in axes(bimat.matrix, 2)
                    bimat.matrix[k, col] = bimat.matrix[k, col] ⊻ bimat.matrix[i, col]
                end
                for col in axes(bimat.Q, 2)
                    bimat.Q[k, col] = bimat.Q[k, col] + bimat.Q[i, col]
                end
            end
        end
    end
    bimat
end
function gaussian_elimination!(bimat::CSSBimatrix)
	qubit_num = size(bimat.matrix, 2) ÷ 2
	gaussian_elimination!(bimat, 1:bimat.xcodenum, 0, 0)
	gaussian_elimination!(bimat, bimat.xcodenum+1:size(bimat.matrix, 1), qubit_num, bimat.xcodenum)
	return bimat
end

function mod2matrix_inverse(H::Matrix{Bool})
    bm = SimpleBimatrix(copy(H),Matrix{Mod2}(I, size(H,1), size(H,1)),collect(1:size(H,2)))
    gaussian_elimination!(bm, 1:size(bm.matrix,1), 0, 0;allow_col_operation = false)
    return bm.Q
end

function mod2matrix_inverse(H::Matrix{Mod2})
    return mod2matrix_inverse([a.x for a in H])
end

function _check_linear_indepent(H::Matrix{Bool})
    bm = SimpleBimatrix(H,Matrix{Mod2}(I, size(H,1), size(H,1)),collect(1:size(H,2)))
    gaussian_elimination!(bm, 1:size(bm.matrix,1), 0, 0)
    return bm,!(bm.matrix[end,:] == fill(Mod2(0),size(H,2)))
end

function check_linear_indepent(H::Matrix{Bool})
    _ , res= _check_linear_indepent(H)
    return res
end

function check_linear_indepent(H::Matrix{Mod2})
    return check_linear_indepent([a.x for a in H])
end
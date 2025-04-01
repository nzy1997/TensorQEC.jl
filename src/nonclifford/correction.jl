"""
    correction_dict(st::Vector{PauliString{N}}, d::Int64; et="XZ")

Generate the correction dictionary for the given stabilizers.
### Arguments
- `st`: The vector of pauli strings, composing the generator of stabilizer group.
- `d`: The maximum number of errors.
- `et`: The type of error to be corrected. It can be "X", "Z", or "XZ". Default is "XZ".
### Returns
- `table`: The correction dictionary, mapping the syndrome to the corresponding error pattern.
"""
function correction_dict(st::Vector{PauliString{N}}, d::Int64;et="XZ") where N
	mat = stabilizers2bimatrix(st).matrix
	num_st = size(mat, 1)
	table = Dict{Int, Int}()
    if et == "X"
        error_vec = N+1:2*N
    elseif et == "Z"
        error_vec = 1:N
    elseif et == "XZ"
        error_vec = 1:2*N
    else 
        error("`et` should be one of 'X', 'Z', 'XZ'.")
    end
	for i in 1:d
		all_combinations = combinations(error_vec, i)
		for combo in all_combinations
			bivec = combo2bivec(combo, N)

			# syndrome = 0(false) means the measure outcome is 1
			# syndrome = 1(true) means the measure outcome is -1
			syndrome = 0

			for j in 1:num_st
				reduce(xor, bivec[findall(mat[j, :])]) && (syndrome |= 1 << (j - 1))
			end
			table[syndrome] = bit_literal(Int.(bivec)...).buf
		end
	end
	return table
end

function combo2bivec(combo::AbstractVector{Int64}, qubit_num::Int64)
	bivec = zeros(Bool, 2 * qubit_num)
	for i in combo
		bivec[mod1(i, 2 * qubit_num)] = true
		if i > 2 * qubit_num
			bivec[i-qubit_num] = true
		end
	end
	return bivec
end

function error_type(i::Int, num_qubits::Int)
	if i <= num_qubits
		return Z, i
    else i <= 2 * num_qubits
		return X, i - num_qubits
    end
end


"""
	correction_circuit(table::Dict{Int,Int}, st_pos::Vector{Int},num_qubits::Int,num_st::Int,num_data_qubits::Int)

Generate the error correction circuit by embedding the truth table into the quantum circuit.

### Arguments
- `table`: The truth table for error correction.
- `st_pos`: The indices of ancilla qubits that measure stabilizers.
- `num_qubits`: The total number of qubits in the circuit.
- `num_st`: The number of stabilizers.
- `num_data_qubits`: The number of data qubits.

### Returns
- `qc`: The error correction circuit.
"""
function correction_circuit(table::Dict{Int,Int},num_qubits::Int,num_st::Int, st_pos::AbstractVector{Int}, total_qubits::Int)
	qc = chain(total_qubits)
	for (k, v) in table
		for i in find1pos(v, 2 * num_qubits)
			type, pos = error_type(i, num_qubits)
            # push!(qc, control(total_qubits, st_pos[error_qubits(k, tb.num_st)], pos => type))
			push!(qc, control(total_qubits, [i ∈ find1pos(k, num_st) ? st_pos[i] : -st_pos[i] for i in 1:length(st_pos)], pos => type))
		end
	end
	return qc
end
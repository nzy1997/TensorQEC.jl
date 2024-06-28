function combo2bivec(combo::AbstractVector{Int64}, qubit_num::Int64)
	bivec = zeros(Bool, 2 * qubit_num)
	for i in combo
		bivec[mod1(i,2*qubit_num)] = true
		if i > 2 * qubit_num
			bivec[i-qubit_num] = true
		end
	end
	return bivec
end

"""
    make_table(st::Vector{PauliString{N}}, d::Int64) where N

Generate the truth table for error correction. Use function [`show_table`](@ref) to print the table. 

### Arguments
- `st`: The vector of Pauli strings, composing the generator of stabilizer group.
- `d`: The maximum number of errors.

### Returns
- `table`: The truth table for error correction.
"""
function make_table(st::Vector{PauliString{N}}, d::Int64) where N
    mat = stabilizers2bimatrix(st).matrix
    num_st=size(mat,1)
    table = Dict{Int,Int}()
	for i in 1:d
		all_combinations = combinations(1:2*N, i)
		for combo in all_combinations
            bivec = combo2bivec(combo, N)

            # sydrome = 0(false) means the measure outcome is 1
            # sydrome = 1(true) means the measure outcome is -1
            sydrome = 0

            for j in 1:num_st
                reduce(xor, bivec[findall(mat[j,:])]) && (sydrome |= 1<< (j-1))
            end
            table[sydrome] = Yao.BitBasis.bit_literal(Int.(bivec)...).buf
		end
	end
    return table
end
"""
    save_table(table::Dict{Int,Int}, filename::String)

Save the truth table for error correction to a file.
"""
function save_table(table::Dict{Int,Int}, filename::String)
    writedlm(filename, hcat(collect(keys(table)), collect(values(table))))
end

"""
    load_table(filename::String)

Load the truth table for error correction from a file.
"""
function load_table(filename::String)
    data = readdlm(filename)
    return Dict{Int,Int}(zip(data[:,1], data[:,2]))
end

"""
    show_table(table::Dict{Int,Int}, num_qubits::Int, num_st::Int)

Print the error information for each error pattern in the table.

### Arguments
- `table`: The truth table for error correction.
- `num_qubits`: The number of qubits.
- `num_st`: The number of stabilizers.
"""
function show_table(table::Dict{Int,Int},num_qubits::Int,num_st::Int)
    for (k, v) in table
        println("Errored stabilizers $(error_qubits(k,num_st))")
        print("Error qubits: ")
        for i in error_qubits(v,2*num_qubits)
            type,pos = error_type(i,num_qubits)
			print("$(pos)=>$(type) ")
		end
        println()
    end
end

function error_type(i::Int, num_qubits::Int)
    if i <= num_qubits
        return Z, i
    else
        return X, i - num_qubits
    end
end

function error_qubits(v::Int, num_qubits::Int)
    return findall([Yao.BitBasis.BitStr{num_qubits}(v)...].==1)
end

"""
    correct_circuit(table::Dict{Int,Int}, st_pos::Vector{Int},num_qubits::Int,num_st::Int,num_data_qubits::Int)

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
function correct_circuit(table::Dict{Int,Int}, st_pos::Vector{Int},num_qubits::Int,num_st::Int,num_data_qubits::Int)
	qc = chain(num_qubits)
	for (k, v) in table 
		for i in error_qubits(v,2*num_data_qubits)
            type,pos = error_type(i,num_data_qubits)
			push!(qc, control(num_qubits, st_pos[error_qubits(k,num_st)], pos => type))
		end
	end
	return qc
end

# function code_distance(bimat::Bimatrix; max_distance::Int = 5)
# 	for k in 1:max_distance
# 		qubit_num = size(bimat.matrix, 2) ÷ 2
# 		all_combinations = combinations(1:qubit_num, k)
# 		for combo in all_combinations
# 			println(combo)
# 			if iszero(rank(bimat.matrix[combo, :]))
# 				return k
# 			end
# 		end
# 	end
# end
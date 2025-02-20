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

"""
	TruthTable

The truth table for error correction.

### Fields
- `table::Dict{Int,Int}`: The truth table for error correction.
- `num_qubits::Int`: The number of qubits.
- `num_st::Int`: The number of stabilizers.
- `d::Int64`: The maximum number of errors.
"""
struct TruthTable
	table::Dict{Int, Int}
	num_qubits::Int
	num_st::Int
	d::Int64
end

# visualization
Base.show(io::IO, ::MIME"text/plain", tb::TruthTable) = show(io, tb)
function Base.show(io::IO, tb::TruthTable)
	show_table(io, tb.table, tb.num_qubits, tb.num_st, tb.d)
	return nothing
end

"""
	show_table(table::Dict{Int,Int}, num_qubits::Int, num_st::Int)

Print the error information for each error pattern in the table.

### Arguments
- `io`: The output stream.
- `table`: The truth table for error correction.
- `num_qubits`: The number of qubits.
- `num_st`: The number of stabilizers.
- `d`: The maximum number of errors.
"""
function show_table(io::IO, table::Dict{Int, Int}, num_qubits::Int, num_st::Int, d::Int64)
    es = Vector{Vector{Int64}}()
	es = Vector{Vector{Int64}}()
	eq = []
	[push!(eq, []) for _ in 1:d]
    eq = Vector{String}()
	for (k, v) in table
        sti=""
		for i in error_qubits(v, 2 * num_qubits)
            type, pos = error_type(i, num_qubits)
            sti *= "$pos => $type"
            if i != error_qubits(v, 2 * num_qubits)[end]
                sti *= ", "
            end
		end
		push!(es, error_qubits(k, num_st))
        push!(eq, sti)
	end
	header = ["Sydrome", "Error"]
	pt = pretty_table(io, hcat(es, eq); header)
	return pt
end

"""
	make_table(st::Vector{PauliString{N}}, d::Int64) where N

Generate the truth table for error correction.

### Arguments
- `st`: The vector of Pauli strings, composing the generator of stabilizer group.
- `d`: The maximum number of errors.

### Returns
- `table`: The truth table for error correction.
"""
function make_table(st::Vector{PauliString{N}}, d::Int64;error_type="all") where N
	mat = stabilizers2bimatrix(st).matrix
	num_st = size(mat, 1)
	table = Dict{Int, Int}()
    if error_type == "all"
        error_vec = 1:3 * N
    elseif error_type == "X"
        error_vec = N+1:2*N
    elseif error_type == "Z"
        error_vec = 1:N
    elseif error_type == "XZ"
        error_vec = 1:2*N
    else 
        error("error_type should be one of 'all', 'X', 'Z', 'XZ'.")
    end
	for i in 1:d
		all_combinations = combinations(error_vec, i)
		for combo in all_combinations
			bivec = combo2bivec(combo, N)

			# sydrome = 0(false) means the measure outcome is 1
			# sydrome = 1(true) means the measure outcome is -1
			sydrome = 0

			for j in 1:num_st
				reduce(xor, bivec[findall(mat[j, :])]) && (sydrome |= 1 << (j - 1))
			end
			table[sydrome] = Yao.BitBasis.bit_literal(Int.(bivec)...).buf
		end
	end
	return TruthTable(table, N, num_st, d)
end

"""
    same_table(tb::TruthTable, filename::String)

Save the truth table for error correction to a file.
"""
function save_table(tb::TruthTable, filename::String)
    save_table(tb.table, filename)
end

function save_table(table::Dict{Int, Int}, filename::String)
	writedlm(filename, hcat(collect(keys(table)), collect(values(table))))
end

"""
	load_table(filename::String)

Load the truth table for error correction from a file.
"""
function load_table(filename::String, num_qubits::Int, num_st::Int, d::Int64)
	data = readdlm(filename)
	return TruthTable(Dict{Int, Int}(zip(data[:, 1], data[:, 2])), num_qubits, num_st, d)
end

function error_type(i::Int, num_qubits::Int)
	if i <= num_qubits
		return Z, i
    else i <= 2 * num_qubits
		return X, i - num_qubits
    end
end

function error_qubits(v::Int, num_qubits::Int)
	return findall([Yao.BitBasis.BitStr{num_qubits}(v)...] .== 1)
end

"""
    table_inference(table::TruthTable, measure_outcome::Vector{Int})

Infer the error type and position from the measure outcome.

### Arguments
- `table`: The truth table for error correction.
- `measure_outcome`: The measure outcome of the stabilizers.

### Returns
- `error`: The error type and position. If the syndrome is not in the truth table, it will print "No such syndrome in the truth table." and return `nothing`
"""
function table_inference(table::TruthTable, measure_outcome::Vector{Int})
    syndrome = 0
    for i in findall(==(-1),measure_outcome)
        syndrome |= 1 << (i - 1)
    end
    return table_inference(table, syndrome)
end
function table_inference(table::TruthTable, syndrome::Int) 
    if haskey(table.table, syndrome)
        error = []
        for i in error_qubits(table.table[syndrome], 2 * table.num_qubits)
            type, pos = error_type(i, table.num_qubits)
            push!(error, (pos => type))
        end
        return error
    else
        print("No such syndrome in the truth table.")
        return nothing
    end
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
function correct_circuit(tb::TruthTable, st_pos::AbstractVector{Int}, total_qubits::Int)
	qc = chain(total_qubits)
	for (k, v) in tb.table
		for i in error_qubits(v, 2 * tb.num_qubits)
			type, pos = error_type(i, tb.num_qubits)
            # push!(qc, control(total_qubits, st_pos[error_qubits(k, tb.num_st)], pos => type))
			push!(qc, control(total_qubits, [i ∈ error_qubits(k, tb.num_st) ? st_pos[i] : -st_pos[i] for i in 1:length(st_pos)], pos => type))
		end
	end
	return qc
end

function code_distance(h::Matrix{Mod2}, xgroup::Vector,max_distance) 
	N = size(h, 2)
	st_num = size(h, 1)
	for k in 1:max_distance
		for combo in combinations(1:N, k)
			eq = zeros(Bool, N)
			eq[combo] .= true
			if sydrome_extraction(Mod2.(eq), h) == zeros(Mod2, st_num)
				if eq ∉ xgroup
					return k
				end
			end
		end
	end
	return 0
end

function code_distance(sts::Vector{PauliString{N}}; max_distance = 5) where N
	bimat = stabilizers2bimatrix(sts)
	qubit_num = size(bimat.matrix, 2) ÷ 2
	xgroup = generate_xgroup(sts[1:bimat.xcodenum])
	zgroup = generate_xgroup(sts[bimat.xcodenum+1:end])

	kx = code_distance(Mod2.(bimat.matrix[1:bimat.xcodenum, 1:qubit_num]),zgroup, max_distance)
	kz = code_distance(Mod2.(bimat.matrix[bimat.xcodenum+1:end, qubit_num+1:end]),xgroup, max_distance)
	return min(kx, kz)
end

function generate_xgroup_ps(st::Vector{PauliString{N}}) where N
	return [pg.ps for pg in generate_group([PauliGroup(0,p) for p in st])]
end

function generate_xgroup(st::Vector{PauliString{N}}) where N
	sts = generate_xgroup_ps(st)
	return [collect(ps.ids .!= 1) for ps in sts]
end
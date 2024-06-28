function measure_circuit_fault_tol!(qc::ChainBlock, st::PauliString{N}, pos::Vector{Int}) where N
	non_one_positions = findall(x -> x != 1, st.ids)
	num_qubits = qc.n
	push!(qc, put(num_qubits, pos[1] => H))
	for i in 2:length(non_one_positions)
		push!(qc, cnot(num_qubits, pos[1], pos[i]))
	end
	for i in 1:length(non_one_positions)
		push!(qc, control(num_qubits,pos[i], non_one_positions[i] => st.ids[non_one_positions[i]] == 2 ? X : (st.ids[non_one_positions[i]] == 3 ? Y : Z)))
	end
	for i in 2:length(non_one_positions)
		push!(qc, cnot(num_qubits,pos[1], pos[i]))
	end
	push!(qc, put(num_qubits,pos[1] => H))
	return qc
end
"""
	measure_circuit_fault_tol(sts::Vector{PauliString{N}}) where N

Generate the Shor type measurement circuit for fault tolerant measurement.

### Arguments
- `sts`: The vector of Pauli strings, composing the generator of stabilizer group.

### Returns
- `qc`: The measurement circuit.
- `st_pos`: The ancilla qubit indices that measrue corresponding stabilizers.
- `num_qubits`: The total number of qubits in the circuit.
"""
function measure_circuit_fault_tol(sts::Vector{PauliString{N}}) where N
	st_length = [count(x -> x != 1, st.ids) for st in sts]
	st_pos = [1 + sum(st_length[1:i-1]) for i in 1:length(st_length)]
	num_qubits = sum(st_length) + N
	qc = chain(num_qubits)
	for i in 1:length(sts)
		measure_circuit_fault_tol!(qc, sts[i], collect(N+st_pos[i]:N+st_pos[i]+st_length[i]-1))
	end
	return qc, st_pos .+ N, num_qubits
end


function measure_circuit!(qc::ChainBlock, st::PauliString{N}, pos::Int) where N
	non_one_positions = findall(x -> x != 1, st.ids)
	num_qubits = qc.n
	push!(qc, put(num_qubits, pos => H))
	for idx in non_one_positions
		push!(qc, control(num_qubits,pos, idx => st.ids[idx] == 2 ? X : (st.ids[idx] == 3 ? Y : Z)))
	end
	push!(qc, put(num_qubits,pos => H))
	return qc
end

function measure_circuit(sts::Vector{PauliString{N}}) where N
	num_qubits = length(sts) + N
	qc = chain(num_qubits)
	for i in 1:length(sts)
		measure_circuit!(qc, sts[i], N+i)
	end
	return qc, collect(N+1:num_qubits), num_qubits
end

function correct_circuit(table::Dict{Int,Int}, st_pos::Vector{Int},num_qubits::Int,num_st::Int,num_data_qubits::Int)
	qc = chain(num_qubits)
	for (k, v) in table
		for  i in findall([Yao.BitBasis.BitStr{num_qubits}(v)...].==1)
			@show i
			if i <= num_data_qubits
				push!(qc, control(num_qubits, st_pos[findall([Yao.BitBasis.BitStr{num_st}(k)...].==1)], i=> Z))
			else
				push!(qc, control(num_qubits, st_pos[findall([Yao.BitBasis.BitStr{num_st}(k)...].==1)],i-num_data_qubits=> X ))
			end
		end
	end
	return qc
end
"""
	measure_circuit_steane(qcen::ChainBlock, data_qubit::Int, sts::Vector{PauliString{N}},xst_num::Int) where N

Generate the Steane type measurement circuit.

### Arguments
- `qcen`: The encoding circuit.
- `data_qubit`: The index of the data qubit.
- `sts`: The vector of Pauli strings, composing the generator of stabilizer group.
- `xst_num`: The number of X type stabilizers.

### Returns
- `qc`: The measurement circuit.
- `st_pos`: The ancilla qubit indices that measrue corresponding stabilizers.
- `num_qubits`: The total number of qubits in the circuit.
"""
function measure_circuit_steane(qcen::ChainBlock, data_qubit::Int, sts::Vector{PauliString{N}},xst_num::Int) where N
	num_sts = length(sts)
	num_qubits = 3 * N + num_sts
	qc = chain(num_qubits)
	copy_circuit_steane!(qc,qcen,data_qubit,sts)
	return _measure_circuit_steane!(qc,sts,xst_num)
end

function copy_circuit_steane!(qc::ChainBlock, qcen::ChainBlock, data_qubit::Int, sts::Vector{PauliString{N}}) where N
	num_sts = length(sts)
	num_qubits = 3 * N + num_sts
	push!(qc, put(num_qubits, 2*N+data_qubit => H))
	push!(qc,subroutine(num_qubits, qcen, (N+1):2*N))
	push!(qc,subroutine(num_qubits, qcen, (2*N+1):(3*N)))
	[push!(qc, control(num_qubits, N+i, i => X) ) for i in 1:N]
	[push!(qc, control(num_qubits, i , 2*N+i => X) ) for i in 1:N]
	return qc
end

function _measure_circuit_steane!(qc::ChainBlock, sts::Vector{PauliString{N}},xst_num::Int) where N
	num_sts = length(sts)
	num_qubits = 3 * N + num_sts
	qcx, _, _ = measure_circuit(sts[1:xst_num])
	push!(qc, subroutine(num_qubits, qcx, (N+1:2*N)∪(3*N+1:3*N+xst_num)))
	qcz, _, _ = measure_circuit(sts[xst_num+1:end])
	push!(qc, subroutine(num_qubits, qcz, (2*N+1:3*N)∪(3*N+xst_num+1:3*N+num_sts)))
	return qc, 3*N+1:3*N+num_sts, num_qubits
end
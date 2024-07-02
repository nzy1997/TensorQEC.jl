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
function measure_circuit_steane(data_qubit::Int, sts::Vector{PauliString{N}},xst_num::Int;qcen = nothing) where N
	num_sts = length(sts)
	num_qubits = 3 * N + num_sts
	qc = chain(num_qubits)

	_single_type!(qc,qcen,(N+1):2*N, 3*N+1:3*N+xst_num, sts[1:xst_num],  true)

	_single_type!(qc,qcen, 2*N+1:3*N,3*N+xst_num+1:3*N+num_sts, sts[xst_num+1:end],  false; data_pos = data_qubit+2*N)

	return qc, 3*N+1:3*N+num_sts, num_qubits
end

function measure_circuit_steane_single_type(data_qubit::Int, sts::Vector{PauliString{N}},xtype::Bool;qcen = nothing) where N
	num_sts = length(sts)
	num_qubits = 2 * N + num_sts
	qc = chain(num_qubits)
	data_pos = nothing
	xtype || (data_pos = data_qubit+N)
	_single_type!(qc,qcen,(N+1):2*N, 2*N+1:2*N+num_sts, sts,  xtype;data_pos)
	return qc, 2*N+1:2*N+num_sts, num_qubits	
end
function _single_type!(qc::ChainBlock, qcen, copy_pos::AbstractVector{Int},st_pos::AbstractVector{Int}, sts::Vector{PauliString{N}}, mtype::Bool; data_pos=nothing) where N
	num_qubits = nqubits(qc)
	qcen === nothing || state_prepare!(qc,qcen,copy_pos;data_pos)
	if mtype
		copy_circuit_steane!(qc,copy_pos,1:N)
	else
		copy_circuit_steane!(qc,1:N,copy_pos)
	end
	qcx, _, _ = measure_circuit(sts)
	push!(qc, subroutine(num_qubits, qcx, copy_pos ∪ st_pos))
end

function state_prepare!(qc::ChainBlock,qcen::ChainBlock, pos::AbstractVector{Int64};data_pos = nothing)
	num_qubits = nqubits(qc)
	data_pos === nothing || push!(qc, put(num_qubits, data_pos => H))
	push!(qc,subroutine(num_qubits, qcen, pos))
end

function copy_circuit_steane!(qc::ChainBlock,  control_pos::AbstractVector{Int64}, target_pos::AbstractVector{Int64})
	num_qubits = nqubits(qc)
	[push!(qc, control(num_qubits, control_pos[i], target_pos[i] => X) ) for i in 1:length(control_pos)]
	return qc
end
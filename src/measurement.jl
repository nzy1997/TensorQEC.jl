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
	return qc, st_pos .+ N
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
	return qc, collect(N+1:num_qubits)
end

"""
	measure_circuit_steane(data_qubit::Int, sts::Vector{PauliString{N}};qcen = nothing) where N

Generate the Steane type measurement circuit.

### Arguments
- `data_qubit`: The index of the data qubit.
- `sts`: The vector of Pauli strings, composing the generator of stabilizer group.
- `qcen`: The encoding circuit. If `nothing`, the measurement circuit will not contain the encoder for ancilla qubits.

### Returns
- `qc`: The measurement circuit.
- `st_pos`: The ancilla qubit indices that measrue corresponding stabilizers.
"""
function measure_circuit_steane(data_qubit::Int, sts::Vector{PauliString{N}};qcen = nothing) where N
	xst_num = count([st.ids[findfirst(!=(1),st.ids)] == 2 for st in sts])
	num_sts = length(sts)
	num_qubits = N
	qc = chain(num_qubits)
	st_pos = Int[]
	if xst_num > 0
		qc = _single_type(qcen, sts[1:xst_num], true;data_pos = data_qubit)
		st_pos = num_qubits+N+1:num_qubits+N+xst_num
		num_qubits += N + xst_num
	end
	if xst_num < num_sts
		zst_num = num_sts - xst_num
		qcz = _single_type(qcen, sts[xst_num+1:end], false;data_pos=data_qubit)
		qc = chain(num_qubits+N+zst_num,subroutine(num_qubits+N+zst_num,qc,1:num_qubits), subroutine(num_qubits+N+zst_num,qcz,(1:N)∪(num_qubits+1:num_qubits+N+zst_num)))
		st_pos = st_pos ∪ (num_qubits+N+1:num_qubits+N+zst_num)
	end
	return qc, st_pos
end

function _single_type(qcen,sts::Vector{PauliString{N}}, mtype::Bool; data_pos = nothing) where N
	qc = chain(2*N+length(sts))
	return _single_type!(qc,qcen,(N+1:2*N),(2*N+1):(2*N+length(sts)),sts,mtype;data_pos)
end
function _single_type!(qc::ChainBlock, qcen, copy_pos::AbstractVector{Int},st_pos::AbstractVector{Int}, sts::Vector{PauliString{N}}, mtype::Bool; data_pos=nothing) where N
	num_qubits = nqubits(qc)
	qcen === nothing || state_prepare!(qc,qcen,copy_pos;data_pos)
	if mtype
		copy_circuit_steane!(qc,copy_pos,1:N)
	else
		copy_circuit_steane!(qc,1:N,copy_pos)
	end
	qcx, _= measure_circuit(sts)
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
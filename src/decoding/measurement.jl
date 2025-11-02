function measure_circuit_fault_tol!(qc::ChainBlock, st::PauliString{N}, pos::Vector{Int}) where N
	non_one_positions = findall(x -> x != Pauli(0), st)
	num_qubits = qc.n
	push!(qc, put(num_qubits, pos[1] => H))
	for i in 2:length(non_one_positions)
		push!(qc, cnot(num_qubits, pos[1], pos[i]))
	end
	for i in 1:length(non_one_positions)
		push!(qc, control(num_qubits,pos[i], non_one_positions[i] => st[non_one_positions[i]] == Pauli(1) ? X : (st[non_one_positions[i]] == Pauli(2) ? Y : Z)))
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
	st_length = [count(x -> x != Pauli(0), st) for st in sts]
	st_pos = [1 + sum(st_length[1:i-1]) for i in 1:length(st_length)]
	num_qubits = sum(st_length) + N
	qc = chain(num_qubits)
	for i in 1:length(sts)
		measure_circuit_fault_tol!(qc, sts[i], collect(N+st_pos[i]:N+st_pos[i]+st_length[i]-1))
	end
	return qc, st_pos .+ N
end


function measurement_circuit!(qc::ChainBlock, st::PauliString{N}, pos::Int) where N
	non_one_positions = findall(x -> x != Pauli(0), st.operators)
	num_qubits = qc.n
	push!(qc, put(num_qubits, pos => H))
	for idx in non_one_positions
		push!(qc, control(num_qubits,pos, idx => st.operators[idx].id == 1 ? X : (st.operators[idx].id == 2 ? Y : Z)))
	end
	push!(qc, put(num_qubits,pos => H))
	return qc
end

function measurement_circuit(sts::Vector{PauliString{N}}) where N
	num_qubits = length(sts) + N
	qc = chain(num_qubits)
	for i in 1:length(sts)
		measurement_circuit!(qc, sts[i], N+i)
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
	xst_num = count([st.operators[findfirst(!=(Pauli(0)), st.operators)].id == 1 for st in sts])
	num_sts = length(sts)
	num_qubits = N
	qc = chain(num_qubits)
	st_pos = Int[]
	if xst_num > 0
		qc = _single_type(qcen, sts[1:xst_num], true)
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
	qcx, _= measurement_circuit(sts)
	push!(qc, subroutine(num_qubits, qcx, copy_pos ∪ st_pos))
end

function state_prepare!(qc::ChainBlock,qcen::ChainBlock, pos::AbstractVector{Int64};data_pos = nothing)
	num_qubits = nqubits(qc)
	data_pos === nothing || push!(qc, put(num_qubits, pos[data_pos] => H))
	push!(qc,subroutine(num_qubits, qcen, pos))
end

function copy_circuit_steane!(qc::ChainBlock,  control_pos::AbstractVector{Int64}, target_pos::AbstractVector{Int64})
	num_qubits = nqubits(qc)
	[push!(qc, control(num_qubits, control_pos[i], target_pos[i] => X) ) for i in 1:length(control_pos)]
	return qc
end

function generate_measurement_circuit_with_errors(sts::Vector{PauliString{N}}, round::Int;  before_round_data_depolarization = 0.0, after_clifford_depolarization = 0.0, after_reset_flip_probability = 0.0, before_measure_flip_probability = 0.0) where N
	qc = generate_measurement_circuit(sts, round; before_round_data_depolarization)
	qc = insert_errors(qc; after_clifford_depolarization, after_reset_flip_probability, before_measure_flip_probability)
	return qc
end

function generate_measurement_circuit(sts::Vector{PauliString{N}}, round::Int;  before_round_data_depolarization = 0.0) where N
	# Only for CSS code
	tanner = CSSTannerGraph(sts)

	num_qubits = length(sts) + N
	qc = chain(num_qubits)
	measure_list = Vector{NumberedMeasure}(undef, length(sts))
	measure_count = 0
	detector_count = 0
	for n in 1:N
		push!(qc, put(num_qubits, n => Measure(1;resetto=bit"0")))
	end

	if !iszero(before_round_data_depolarization)
		for n in 1:N
			push!(qc, put(num_qubits, n => DepolarizingChannel(1,before_round_data_depolarization)))
		end
	end

	for i in 1:length(sts)
		measurement_circuit!(qc, sts[i], N+i)
		measure_count += 1
		m = NumberedMeasure(Measure(1; resetto=bit"0"), measure_count)
		push!(qc, put(num_qubits, N+i => m))
		measure_list[i] = m
		
		# check if the stabilizer is a Z stabilizer
		if  !any(x -> (x.id == 1) || (x.id == 2), sts[i].operators)
			detector_count += 1
			db = DetectorBlock{2}([measure_list[i]], detector_count, 0)
			push!(qc, put(num_qubits, N+i => db))
		end
	end

	for d in 1:round
		if !iszero(before_round_data_depolarization)
			for n in 1:N
				push!(qc, put(num_qubits, n => DepolarizingChannel(1,before_round_data_depolarization)))
			end
		end

		for n in 1:N
			push!(qc, put(num_qubits, n => Measure(1;resetto=bit"0")))
		end

		for i in 1:length(sts)
			measurement_circuit!(qc, sts[i], N+i)
			measure_count += 1
			m = NumberedMeasure(Measure(1; resetto=bit"0"), measure_count)
			push!(qc, put(num_qubits, N+i => m))

			detector_count += 1
			db = DetectorBlock{2}([m, measure_list[i]], detector_count, 0)
			push!(qc, put(num_qubits, N+i => db))

			measure_list[i] = m
		end
	end
	
	qubit_measure_list = Vector{NumberedMeasure}(undef, N)
	for n in 1:N
		m = NumberedMeasure(Measure(1), measure_count+n)
		push!(qc, put(num_qubits, n => m))
		qubit_measure_list[n] = m
	end

	for i in 1:length(sts)
		if  !any(x -> (x.id == 1) || (x.id == 2), sts[i].operators)
			zpos = findall(x -> (x.id == 3), sts[i].operators)
			detector_count += 1
			db = DetectorBlock{2}([qubit_measure_list[zpos]..., measure_list[i]], detector_count, 0)
			push!(qc, put(num_qubits, zpos[1] => db))
		end
	end

	_, lzs = logical_operator(tanner)
	for lz in eachrow(lzs)
		detector_count += 1
		db = DetectorBlock{2}(qubit_measure_list[findall(x -> x.x, lz)], detector_count, 1)
		push!(qc, put(num_qubits, 1 => db))
	end
	return qc
end

struct MeasurementCircuitInfo
	qubit_pos::Vector{Int}
	xstabilizer_pos::Vector{Int}
	zstabilizer_pos::Vector{Int}
	xmeasure_list::Vector{Dict{Int,Int}}
	zmeasure_list::Vector{Dict{Int,Int}}
	H_before_list::Vector{Int}
	H_after_list::Vector{Int}
end

MeasurementCircuitInfo(qubit_pos::Vector{Int}, xstabilizer_pos::Vector{Int}, zstabilizer_pos::Vector{Int}, xmeasure_list::Vector{Dict{Int,Int}}, zmeasure_list::Vector{Dict{Int,Int}}) = MeasurementCircuitInfo(qubit_pos, xstabilizer_pos, zstabilizer_pos, xmeasure_list, zmeasure_list, xstabilizer_pos, xstabilizer_pos)

function make_measurement_circuit(mci::MeasurementCircuitInfo; with_measurement::Bool = false)
	total_qubit_num = length(mci.qubit_pos) + length(mci.xstabilizer_pos) + length(mci.zstabilizer_pos)
	qc = chain(total_qubit_num)
	for x_pos in mci.H_before_list
		push!(qc, put(total_qubit_num, x_pos => H))
	end
	for (x_dict,z_dict) in zip(mci.xmeasure_list,mci.zmeasure_list)
		for (x_key,x_value) in x_dict
			push!(qc, control(total_qubit_num, x_key, x_value => X))
		end
		for (z_key,z_value) in z_dict
			push!(qc, control(total_qubit_num, z_value, z_key => X))
		end
	end
	for x_pos in mci.H_after_list
		push!(qc, put(total_qubit_num, x_pos => H))
	end
	if with_measurement
		push!(qc, put(total_qubit_num, mci.xstabilizer_pos ∪ mci.H_after_list => Measure(length(mci.xstabilizer_pos ∪ mci.H_after_list);resetto=bit"0")))
	end
	return qc
end

function generate_measurement_circuit_info(c::SurfaceCode; type::Int = 1,switch_data_qubit::Bool = false) # type = 1,2,3,4
	data_qubit_num = c.m*c.n
	tanner = CSSTannerGraph(stabilizers(c))
	xs2q = tanner.stgx.s2q
	xstab_num = length(xs2q)
	zs2q = tanner.stgz.s2q
	zstab_num = length(zs2q)
	qubit_pos = collect(1:data_qubit_num)
	xstabilizer_pos = collect(data_qubit_num+1:data_qubit_num+xstab_num)
	zstabilizer_pos = collect(data_qubit_num+xstab_num+1:data_qubit_num+xstab_num+zstab_num)
	H_after_list = copy(xstabilizer_pos)
	if switch_data_qubit
		xmeasure_list = [Dict{Int,Int}() for _ in 1:5]
		zmeasure_list = [Dict{Int,Int}() for _ in 1:5]
	else
		xmeasure_list = [Dict{Int,Int}() for _ in 1:4]
		zmeasure_list = [Dict{Int,Int}() for _ in 1:4]
	end
	typex_4 = [[1,3,2,4],[3,1,4,2],[2,4,1,3],[4,2,3,1]]
	typex_2_pos = [[1,2,3,4],[1,2,3,4],[3,4,1,2],[3,4,1,2]]
	typex_2 = [[1,2],[2,1],[1,2],[2,1]]
	for (xidx,pos) in enumerate(xs2q)
		if length(pos) == 4
			pos_sorted = sort(pos)
			xmeasure_list[1][xidx+data_qubit_num] = pos_sorted[typex_4[type][1]]
			xmeasure_list[2][xidx+data_qubit_num] = pos_sorted[typex_4[type][2]]
			xmeasure_list[3][xidx+data_qubit_num] = pos_sorted[typex_4[type][3]]
			if switch_data_qubit
				xmeasure_list[4][pos_sorted[typex_4[type][4]]] = xidx+data_qubit_num
				xmeasure_list[5][xidx+data_qubit_num] = pos_sorted[typex_4[type][4]]
				replace!(H_after_list, xidx+data_qubit_num => pos_sorted[typex_4[type][4]])
			else
				xmeasure_list[4][xidx+data_qubit_num] = pos_sorted[typex_4[type][4]]
			end
		elseif length(pos) == 2
			pos_sorted = sort(pos)
			if pos_sorted[1] % c.n == 0
				xmeasure_list[typex_2_pos[type][1]][xidx+data_qubit_num] = pos_sorted[typex_2[type][1]]
				xmeasure_list[typex_2_pos[type][2]][xidx+data_qubit_num] = pos_sorted[typex_2[type][2]]
			else 
				xmeasure_list[typex_2_pos[type][3]][xidx+data_qubit_num] = pos_sorted[typex_2[type][1]]
				xmeasure_list[typex_2_pos[type][4]][xidx+data_qubit_num] = pos_sorted[typex_2[type][2]]
			end
		end
	end
	
	typez_4 = [[1,2,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]]
	typez_2_pos = [[3,4,1,2],[1,2,3,4],[3,4,1,2],[1,2,3,4]]
	typez_2 = [[1,2],[1,2],[2,1],[2,1]]
	for (zidx,pos) in enumerate(zs2q)
		if length(pos) == 4
			pos_sorted = sort(pos)
			zmeasure_list[typez_4[type][1]][zidx+data_qubit_num+xstab_num] = pos_sorted[typez_4[type][1]]
			zmeasure_list[typez_4[type][2]][zidx+data_qubit_num+xstab_num] = pos_sorted[typez_4[type][2]]
			zmeasure_list[typez_4[type][3]][zidx+data_qubit_num+xstab_num] = pos_sorted[typez_4[type][3]]
			if switch_data_qubit
				zmeasure_list[4][pos_sorted[typez_4[type][4]]] = zidx+data_qubit_num+xstab_num
				zmeasure_list[5][zidx+data_qubit_num+xstab_num] = pos_sorted[typez_4[type][4]]
			else
				zmeasure_list[4][zidx+data_qubit_num+xstab_num] = pos_sorted[typez_4[type][4]]
			end
		elseif length(pos) == 2
			pos_sorted = sort(pos)
			if pos_sorted[1] <= c.n
				zmeasure_list[typez_2_pos[type][1]][zidx+data_qubit_num+xstab_num] = pos_sorted[typez_2[type][1]]
				zmeasure_list[typez_2_pos[type][2]][zidx+data_qubit_num+xstab_num] = pos_sorted[typez_2[type][2]]
			else
				zmeasure_list[typez_2_pos[type][3]][zidx+data_qubit_num+xstab_num] = pos_sorted[typez_2[type][1]]
				zmeasure_list[typez_2_pos[type][4]][zidx+data_qubit_num+xstab_num] = pos_sorted[typez_2[type][2]]
			end
		end
	end
	return MeasurementCircuitInfo(qubit_pos, xstabilizer_pos, zstabilizer_pos, xmeasure_list, zmeasure_list, xstabilizer_pos, H_after_list)
end
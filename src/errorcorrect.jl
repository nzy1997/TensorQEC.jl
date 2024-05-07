function correction_pauli_string(qubit_num::Int, syn::Dict{Int, Bool}, prob::Dict{Int, Vector{Float64}})
	ps = ones(Int, qubit_num)
	for (k, v) in prob
		if k ∈ keys(syn)
			if syn[k] #syn[k] is true, measure outcome is -1, use X or Y
				ps[k] = (findmax(v)[2]) == 1 ? 2 : 3
			elseif findmax(v)[2] == 2
				ps[k] = 4
			end
		else
            ps[k] = [1,2,3,4][findmax(v)[2]]
		end
	end
	return PauliString(ps[1:end]...)
end


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

function correct_circuit(table::Dict{Int,Int}, st_pos::Vector{Int},num_qubits::Int,num_st::Int,num_logical_qubits::Int)
	qc = chain(num_qubits)
	for (k, v) in table
		for  i in findall([Yao.BitBasis.BitStr{num_qubits}(v)...].==1)
			if i <= num_logical_qubits
				@show st_pos[findall([Yao.BitBasis.BitStr{num_st}(k)...].==1)]
				@show i
				push!(qc, control(num_qubits, st_pos[findall([Yao.BitBasis.BitStr{num_st}(k)...].==1)], i=> X))
			else
				push!(qc, control(num_qubits, st_pos[findall([Yao.BitBasis.BitStr{num_st}(k)...].==1)],i-num_logical_qubits=> Z ))
			end
		end
	end
	return qc
end

function measure_circuit_steane(qcen::ChainBlock, sts::Vector{PauliString{N}}) where N
	num_sts = length(sts)
	num_qubits = 3 * N + num_sts
	@show num_qubits
	qc = chain(num_qubits)
	push!(qc,subroutine(num_qubits, qcen, (N+1):(N+18)))
	push!(qc,subroutine(num_qubits, qcen, (N+1):(N+18)))
	return qc
end
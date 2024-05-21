using Yao, YaoPlots, TensorQEC
function GHZ()
	qubits = 4
	qc = chain(qubits)
	push!(qc, put(qubits, 1 => H))
	push!(qc, cnot(qubits, 1, 2))
	push!(qc, cnot(qubits, 1, 3))
	push!(qc, cnot(qubits, 1, 4))
	# push!(qc, put(qubits, 1 => H))
	# qc=chain(qc, Measure(4;locs=1))
	vizcircuit(qc; starting_texts = vcat([raw"0"], [" " for i ∈ 1:nqubits(qc)-1]))
end

function encodez()
	qubits = 4
	qc = chain(qubits)
	push!(qc, cnot(qubits, 2, 1))
	push!(qc, cnot(qubits, 3, 1))
	push!(qc, cnot(qubits, 4, 1))
	vizcircuit(qc; starting_texts = vcat([raw"0"], [" " for i ∈ 1:nqubits(qc)-1]))
end

function draw()
	t = TensorQEC.ToricCode(2, 2)
	st = TensorQEC.stabilizers(t)
	qn = nqubits(st[1])
	qcen, data_qubits, code = TensorQEC.encode_stabilizers(st)
	stli = TensorQEC.stabilizers(t, linearly_independent = false)
	qcm, st_pos, num_qubits = measure_circuit_fault_tol(stli)

	bimat = TensorQEC.stabilizers2bimatrix(stli)
	table = make_table(bimat.matrix, 1)
	qccr = correct_circuit(table, st_pos, num_qubits, qn, qn)

	gates = chain(subroutine(num_qubits, qcen, 1:qn), qcm)
	Optimise.simplify(gates, rules=[Optimise.to_basictypes, Optimise.eliminate_nested])

	# qcf=chain(gates,qccr)
	YaoPlots.CircuitStyles.r[] = 0.3
	vizcircuit(gates; starting_texts = 1:num_qubits, filename = "measure_free.svg")
end
draw()

function toric_x_mf()
	st=[PauliString((4,4,4,4,1,1,1)),PauliString((1,1,1,4,4,4,4))]
	qcm, st_pos, num_qubits = measure_circuit_fault_tol(st)
	
	push!(qcm, control(num_qubits, st_pos, 4 => X))
	YaoPlots.CircuitStyles.r[] = 0.3
	vizcircuit(qcm; starting_texts = 1:num_qubits, filename = "measure_free.svg")
end
toric_x_mf()
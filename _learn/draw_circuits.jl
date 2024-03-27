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
	t = TensorQEC.ToricCode(3, 3)
	st = TensorQEC.stabilizers(t)
	qcen, data_qubits, code = TensorQEC.encode_stabilizers(st)
	stli = TensorQEC.stabilizers(t, linearly_independent = false)
	qcm, st_pos, num_qubits = measure_circuit_fault_tol(stli)

	bimat = TensorQEC.stabilizers2bimatrix(stli)
	table = make_table(bimat.matrix, 1)
	qccr = correct_circuit(table, st_pos, 90, 18, 18)

	gates = chain(subroutine(90, qcen, 1:18), qcm)
	Optimise.simplify(gates, rules=[Optimise.to_basictypes, Optimise.eliminate_nested])

	qcf=chain(gates,qccr)
	YaoPlots.CircuitStyles.r[] = 0.3
	vizcircuit(qcf; starting_texts = 1:90, filename = "measure_free.svg")
end
draw()

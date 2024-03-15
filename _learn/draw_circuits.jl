using Yao, YaoPlots
function GHZ()
	qubits = 4
	qc = chain(qubits)
	push!(qc, put(qubits, 1 => H))
	push!(qc, cnot(qubits, 1, 2))
	push!(qc, cnot(qubits, 1, 3))
	push!(qc, cnot(qubits, 1, 4))
	# push!(qc, put(qubits, 1 => H))
	# qc=chain(qc, Measure(4;locs=1))
	vizcircuit(qc; starting_texts=vcat([raw"0"],[" " for i = 1:nqubits(qc)-1]))
end

function encodez()
	qubits = 4
	qc = chain(qubits)
	push!(qc, cnot(qubits, 2, 1))
	push!(qc, cnot(qubits, 3, 1))
	push!(qc, cnot(qubits, 4, 1))
	vizcircuit(qc; starting_texts=vcat([raw"0"],[" " for i = 1:nqubits(qc)-1]))
end
GHZ()
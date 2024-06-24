using TensorQEC, TensorQEC.Yao, YaoPlots

function ge_me_qc()
	qubit_num2 = 17
	qc_measure = chain(qubit_num2)
	for i in 10:17
		push!(qc_measure, put(qubit_num2, i => H))
	end

	# X stabilizers
	# 24, 1235, 5789, 68 (Z error)
	push!(qc_measure, control(qubit_num2, 10, (2, 4) => kron(X, X)))
	push!(qc_measure, control(qubit_num2, 11, (1, 2, 3, 5) => kron(X, X, X, X)))
	push!(qc_measure, control(qubit_num2, 12, (5, 7, 8, 9) => kron(X, X, X, X)))
	push!(qc_measure, control(qubit_num2, 13, (6, 8) => kron(X, X)))

	# Z type: 13, 2457, 3568, 79 (X error)
	push!(qc_measure, control(qubit_num2, 14, (1, 3) => kron(Z, Z)))
	push!(qc_measure, control(qubit_num2, 16, (3, 5, 6, 8) => kron(Z, Z, Z, Z)))
	push!(qc_measure, control(qubit_num2, 15, (2, 4, 5, 7) => kron(Z, Z, Z, Z)))
	push!(qc_measure, control(qubit_num2, 17, (7, 9) => kron(Z, Z)))

	for i in 10:17
		push!(qc_measure, put(qubit_num2, i => H))
	end
	return qc_measure
end

function ge_co_qc()
	qubit_num2 = 17
	qc_correct = chain(qubit_num2)
	# Z error
	push!(qc_correct, control(qubit_num2, (10, 11), 2 => Z))
	push!(qc_correct, control(qubit_num2, (11, 12), 5 => Z))
	push!(qc_correct, control(qubit_num2, (12, 13), 8 => Z))
	push!(qc_correct, control(qubit_num2, (10, -11), 4 => Z))
	push!(qc_correct, control(qubit_num2, (-12, 13), 6 => Z))
	push!(qc_correct, control(qubit_num2, (-13, 12, -11), 9 => Z))
	push!(qc_correct, control(qubit_num2, (-10, 11, -12), 3 => Z))

	# X error
	push!(qc_correct, control(qubit_num2, (14, 16), 3 => X))
	push!(qc_correct, control(qubit_num2, (15, 16), 5 => X))
	push!(qc_correct, control(qubit_num2, (15, 17), 7 => X))
	push!(qc_correct, control(qubit_num2, (-15, 17), 9 => X))
	push!(qc_correct, control(qubit_num2, (14, -16), 1 => X))
	push!(qc_correct, control(qubit_num2, (-14, -15, 16), 6 => X))
	push!(qc_correct, control(qubit_num2, (-16, 15, -17), 2 => X))
	return qc_correct
end
# define the stabilizers
qubit_num = 9
st = stabilizers(SurfaceCode{3}())
@info "stabilizers: $st"

# Generate the encoding circuits of the stabilizers
qc, data_qubits, code = encode_stabilizers(st)
@info "encoding circuits: $qc, data qubits: $data_qubits"

# Create a quantum register. Qubits in "data_qubits" are randomly initilized, and the rest ancilla qubits are in the |0> state.
reg = join(rand_state(1),zero_state(8))
# Apply the encoding circuits.
regcopy = copy(reg)
apply!(reg, qc)

# Apply a X error on the third qubit
apply!(reg, put(9, 3 => X))
@info "applied X error on the third qubit"

# Measure the syndrome
measure_outcome = measure_syndrome!(reg, st)
@info "measured syndrome: $measure_outcome"

# Generate the syndrome dictionary
syn_dict = TensorQEC.generate_syndrome_dict(code, syndrome_transform(code, measure_outcome))

# Generate the tensor network for syndrome inference
cl = clifford_network(qc)
p = fill([0.85, 0.05, 0.05, 0.05], qubit_num)
pinf = syndrome_inference(cl, syn_dict, p)
@info "inferred error probability: $pinf"

# Generate the Pauli string for error correction
ps_ec_phy = TensorQEC.pauli_string_map_iter(correction_pauli_string(qubit_num, syn_dict, pinf), qc)
@info "Pauli string for error correction: $ps_ec_phy"

# Apply the error correction
apply!(reg, Yao.YaoBlocks.Optimise.to_basictypes(ps_ec_phy))

# Measure the syndrome after error correction
syndrome_result = measure_syndrome!(reg, st)
@info "measured syndrome: $syndrome_result"
apply!(reg, qc')
fidelity_after = fidelity(density_matrix(reg, [data_qubits...]), density_matrix(regcopy, [data_qubits...]))
@info "fidelity after error correction: $fidelity_after"

qubit_num2 = 17
reg2 = join(zero_state(8), rand_state(1), zero_state(8))
focus!(reg2, (1:9))
apply!(reg2, qc)
relax!(reg2, (1:9))

apply!(reg2, put(17, 3 => Y))

qc_measure = ge_me_qc()
apply!(reg2, qc_measure)
qc_correct = ge_co_qc()
apply!(reg2, qc_correct)

focus!(reg2, (1:9))
[(real.(measure(Yao.YaoBlocks.Optimise.to_basictypes(st[i]),reg2))) for i in 1:8]

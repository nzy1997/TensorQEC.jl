using TensorQEC, TensorQEC.Yao
# define the stabilizers
qubit_num = 3
st = [PauliString((1,4,4)),PauliString((4,1,4))]
@info "stabilizers: $st"

# Generate the encoding circuits of the stabilizers
qc, data_qubits, code = encode_stabilizers(st)
@info "encoding circuits: $qc, data qubits: $data_qubits"

# Create a quantum register. Qubits in "data_qubits" are randomly initilized, and the rest ancilla qubits are in the |0> state.
reg = join(rand_state(1), zero_state(2))  # join(qubit3, qubit2, qubit1)
# Apply the encoding circuits.
regcopy = copy(reg)
apply!(reg, qc)

# Apply a X error on the third qubit
apply!(reg, put(qubit_num, 3=>X))
@info "applied X error on the third qubit"

# Measure the syndrome
measure_outcome=measure_syndrome!(reg, st)
@info "measured syndrome: $measure_outcome"

# Generate the syndrome dictionary
syn_dict=TensorQEC.generate_syndrome_dict(code,syndrome_transform(code, measure_outcome))

# Generate the tensor network for syndrome inference
cl = clifford_network(qc)
p = fill([0.85,0.05,0.05,0.05],qubit_num)
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
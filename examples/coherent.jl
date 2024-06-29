# # Coherent Error Simulation
# Coherent error or unitary error is a type of error that can be described by a unitary matrix acting on the quantum state. For example, if we want to apply an unitary $U$ on the state, however, we apply a unitary $U'$ instead, which differ from $U$ slighty. The error can be described by an unitary $E = U'U^{\dagger}$ acting on the quantum state. Usually, this unitary is non-Clifford, thus it is hard to simulate with the stabilizer formalism. Here, we use tensor network to simulate a quantum circuit with coherent error.

# ## Quantum Circuit Construction
# First, we define the stabilizers for 3-qubit repetition code. 
using TensorQEC, TensorQEC.Yao
using TensorQEC.OMEinsum
st = [PauliString((1,4,4)),PauliString((4,1,4))]

# Then we generate the encoding circuits of the stabilizers by [`encode_stabilizers`](@ref). `qc` is the encoding circuit, `data_qubits` are the qubits that we should put initial qubtis in, and `code` is the structure records information of the encoding circuit.
qcen, data_qubits, code = encode_stabilizers(st)
vizcircuit(qcen)

# Now we construct a 3-qubit quantum circuit to perform the following operations:
# 1. Encoding the initial state with the encoding circuit.
# 2. Apply an logical X gate which consists of three X gates on the three qubits.
# 3. Decoding the state with the encoding circuit.
# 4. Apply an X gate to recover the initial state.
qc = chain(qcen, put(3, 1=>X),put(3,2=>X),put(3,3=>X), qcen', put(3, 3=>X))
vizcircuit(qc)

# This circuit should act trivially on the third qubit. We can transform the circuit to a tensor network to check the infidelity after the circuit.
tn = fidelity_tensornetwork(qc, QCInfo(data_qubits, 3))
optnet = optimize_code(tn, TreeSA(; ntrials=1, niters=3), OMEinsum.MergeVectors())
inf = 1-abs(contract(optnet)[1]/4)

# ## Coherent Error Simulation
# We add coherent error to the circuit by adding unitary error to every unitary gate. Here we suppose that for each unitary, the errored unitary is fixed to it, i.e., we apply the same errored X gate every time we want to apply an X gate. If you want to apply random errored unitary, you can call [`coherent_error_unitary`](@ref) to generate the errored unitary ervey time you want to apply it. 

# [`error_pairs`](@ref) generates the error pairs for the gates to be replaced. Here we replace all the X gates and CNOT gates with the same errored X gate and CNOT gate. 
pairs, vector = error_pairs(1e-5; gates = [X,ConstGate.CNOT])

# Then we can generate the error quantum circuit by [`error_quantum_circuit`](@ref), which replaces the gates in the original circuit with the errored gates.
eqc = error_quantum_circuit(qc, pairs)
vizcircuit(eqc)

# Finally, we can check the infidelity after the circuit with coherent error.
tn = fidelity_tensornetwork(eqc, QCInfo(data_qubits, 3))
optnet = optimize_code(tn, TreeSA(; ntrials=1, niters=3), OMEinsum.MergeVectors())
inf = 1-abs(contract(optnet)[1]/4)
# # Coherent Error Simulation
# Coherent error or unitary error is a type of error that can be described by a unitary matrix acting on the quantum state. For example, if we want to apply an unitary $U$ on the state, however, we apply a unitary $U'$ instead, which differ from $U$ slighty. The error can be described by an unitary $E = U'U^{\dagger}$ acting on the quantum state. Usually, this unitary is non-Clifford, thus it is hard to simulate with the stabilizer formalism. Here, we use tensor network to simulate a quantum circuit with coherent error.

# ## Quantum Circuit Construction
# First, we define the stabilizers for Steane code. 
using TensorQEC, TensorQEC.Yao
using TensorQEC.OMEinsum
st = stabilizers(SteaneCode()) 

# Then we generate the encoding circuits of the stabilizers by [`encode_stabilizers`](@ref). `qc` is the encoding circuit, `data_qubits` are the qubits that we should put initial qubtis in, and `code` is the structure records information of the encoding circuit.
qcen, data_qubits, code = encode_stabilizers(st)
vizcircuit(qcen)

# Now we construct a 7-qubit quantum circuit to perform the following operations:
# 1. Encoding the initial state with the encoding circuit.
# 2. Apply an logical X gate which consists of three X gates on the seven qubits.
# 3. Decoding the state with the encoding circuit.
# 4. Apply an X gate to recover the initial state.
qc = chain(qcen)
push!(qc, [put(7, i => X) for i in 1:7]...)
push!(qc, qcen')
push!(qc, put(7, 6 => X))
vizcircuit(qc)

# This circuit should act trivially on the data qubit. We will check this later.

# ## Circuit Simulation with Tensor Networks
# Simulating quantum circuits using tensor networks is a powerful technique, particularly for circuits that are not easily amenable to simulation with classical computers[^Markov]. We can replace gates and density matries by tensors to get the tensor network representation of the quantum circuit. Applying a quantum channel on a density matix is equivalent to connecting two tensors together and contracting them.
# ![](../images/applygate.svg)

# To trace out a matrix in the tensor network, we can simply connect the two indices of the matrix and contract them. To partially trace out the ancilla qubits, we can simply connect the output indices of the ancilla qubits and contract them.
# ![](../images/qchannel.svg)

# We can use the function [`simulation_tensornetwork`](@ref) to generate the tensor network of the quantum channel. 
tn,input_indices,output_indices = simulation_tensornetwork(qc, QCInfo(data_qubits, 7))

# And we contract the tensor network to get the matrix representation of the quantum channel, which is an identity channel. 
optnet = optimize_code(tn, TreeSA(; ntrials=1, niters=3), OMEinsum.MergeVectors())
matr = contract(optnet)

# Also we can compute the circuit fidelity with identity channel directly by connecting the input and output indices of the quantum channel and contracting them.
# ![](../images/fidelity.svg)

# [`fidelity_tensornetwork`](@ref) transforms the circuit to a tensor network to calculate fidelity.
tn = fidelity_tensornetwork(qc, QCInfo(data_qubits, 7))
optnet = optimize_code(tn, TreeSA(; ntrials=1, niters=3), OMEinsum.MergeVectors())
infidelity = 1 - abs(contract(optnet)[1])

# ## Coherent Error Simulation with Tensor Network
# We add coherent error to the circuit by adding unitary error to every unitary gate. Here we suppose that for each unitary, the errored unitary is fixed to it, i.e., we apply the same errored X gate every time we want to apply an X gate. If you want to apply random errored unitary, you can call [`coherent_error_unitary`](@ref) to generate the errored unitary ervey time you want to apply it. 

# [`error_pairs`](@ref) generates the error pairs for the gates to be replaced. Here we replace all the X gates and CNOT gates with the same errored X gate and CNOT gate. 
pairs, vector = error_pairs(1e-5; gates = [X,ConstGate.CNOT])

# Then we can generate the error quantum circuit by [`error_quantum_circuit`](@ref), which replaces the gates in the original circuit with the errored gates.
eqc = error_quantum_circuit(qc, pairs)
vizcircuit(eqc)

# Finally, we can check the infidelity after the circuit with coherent error.
tn = fidelity_tensornetwork(eqc, QCInfo(data_qubits, 7))
optnet = optimize_code(tn, TreeSA(; ntrials=1, niters=3), OMEinsum.MergeVectors())
infidelity = 1 - abs(contract(optnet)[1])

# [^Markov]: Markov, I. L., & Shi, Y. (2008). Simulating Quantum Computation by Contracting Tensor Networks. SIAM Journal on Computing, 38(3), 963–981. https://doi.org/10.1137/050644756
# # Measurement-Free QEC 
# This example demonstrates how to use tensor network to simulate a error correction process.
# We use the $[[7,1,3]]$ steane code and the measurement-free QEC[^Heußen] as an example. There are non-clifford gates in the quantum circuit, so we use tensor network to simulate the process.

# ## Measurement-Free Quantum Error Correction
# Traditional quantum error correction involves several key procedures. First, the encoding procedure involves mapping the logical quantum information into a larger, redundant quantum state using QEC codes such as the Shor code or the surface code. Usually, those quantum code are defined by stabilizers. Then, the syndrome extraction is the process that we extract the value of the stabilizers into ancilla qubits. And we measure the ancilla qubits to detect the value of stabilizers. After detection, the error syndromes are identified, which indicates the presence and location of errors in the quantum state. Next, the error correction procedure uses quantum gates to apply operations that reverse the effects of errors, effectively restoring the quantum state to its original form. 

# In the measurement-free QEC protocol, we embed the classical truth table of the error correction into the quantum circuit directly. For example, if we measure the stabilizer 1 and 2 to ancilla qubit 1 and 2, and if they are both in state $|1\rangle$, we know that there is an X error on the first qubit. We can encode this information into the quantum circuit directly by a multi-controlled-X gate.

# ![](../images/ccx.svg)

# Since such multi-controlled gate is non-clifford, we can't simulate it with clifford circuit simulator.

# ## Definition of Stabilizers and Encoding Circuits
using TensorQEC, TensorQEC.Yao
using TensorQEC.OMEinsum
st = stabilizers(SteaneCode())

# Generate the encoding circuits of the stabilizers by [`encode_stabilizers`](@ref). `qcen` is the encoding circuit, `data_qubits` are the qubits that we should put initial qubtis in, and `code` is the structure records information of the encoding circuit.
qcen, data_qubits, code = encode_stabilizers(st)
vizcircuit(qcen)

# ## Syndrome Extraction and Measurement-Free Error Correction
# First, we generate the steane measurement circuit by [`measure_circuit_steane`](@ref) and `st_pos` records the ancilla qubits that store the measurement results of the stabilizers.
qcm,st_pos, num_qubits = measure_circuit_steane(data_qubits[1],st,3;qcen)
vizcircuit(qcm)

# Then we generate truth table for the error correction by [`make_table`](@ref). For more detials on truth table, please check [Inference with Truth Table](@ref).
table = make_table(st, 1;error_type = "Z")

# Now we use [`correct_circuit`](@ref) to generate the measurement-free correction circuit by encoding the truth table on the quantum circuit directly.
qccr = correct_circuit(table, 22:24, num_qubits)
vizcircuit(qccr)

# ## Circuit Simulation with Tensor Networks
# We connect the encoding circuit, the measurement circuit, and the correction circuit to form a full circuit. And we apply a Y error on the third qubit after encoding.
qcf=chain(subroutine(num_qubits, qcen, 1:7),put(27,3=>Z),qcm,qccr,subroutine(num_qubits, qcen', 1:7))
vizcircuit(qcf)

# Then we transform the circuit to a tensor network and optimize its contraction order. [`QCInfo`](@ref) records the information of the quantum circuit, including the data qubits and the number of qubits. [`fidelity_tensornetwork`](@ref) constructs the tensor network to calculate the fidelity after error correction.
tn = fidelity_tensornetwork(qcf, QCInfo(data_qubits, 27))

# Finally, we optimize the contraction order and contract the tensor network to get the infidelity after error correction.
optnet = optimize_code(tn, TreeSA(; ntrials=1, niters=5), OMEinsum.MergeVectors()) 
infidelity = 1 - abs(contract(optnet)[1])

# [^Heußen]: Heußen, S., Locher, D. F., & Müller, M. (2024). Measurement-Free Fault-Tolerant Quantum Error Correction in Near-Term Devices. PRX Quantum, 5(1), 010333. https://doi.org/10.1103/PRXQuantum.5.010333

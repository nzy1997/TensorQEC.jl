# # Tensor Network Simulation
# This example demonstrates how to use tensor network to simulate the error correction process.
# We use the $[[7,1,3]]$ steane code and the measurement-free QEC as an example. There are none-clifford gates in the circuit, so 
# TODO: Add reference

# ## Definition of Stabilizers and Encoding Circuits
using TensorQEC, TensorQEC.Yao
using TensorQEC.OMEinsum
st = stabilizers(SteaneCode())

# Generate the encoding circuits of the stabilizers.
qcen, data_qubits, code = encode_stabilizers(st)

# ## Syndrome Extraction and Measurement-Free Error Correction
# First, we generate the steane measurement circuit and 'st_pos' records the ancilla qubits that store the measurement results of the stabilizers.
qcm,st_pos, num_qubits = measure_circuit_steane(qcen,data_qubits[1],st,3)

# Then we generate truth table for the error correction.
table = make_table(stabilizers2bimatrix(st).matrix, 1)

# Now we can generate the measurement-free correction circuit by encoding the truth table on the quantum circuit directly.
qccr = correct_circuit(table, collect(st_pos), num_qubits, 6, 7)

# ## Circuit Simulation with Tensor Networks
# We connect the encoding circuit, the measurement circuit, and the correction circuit to form a full circuit. And we apply a Y error on the third qubit after encoding.
qcf=chain(subroutine(num_qubits, qcen, 1:7),put(27,3=>Y),qcm,qccr,subroutine(num_qubits, qcen', 1:7))
vizcircuit(qcf)

# Then we transform the circuit to a tensor network and optimize its contraction order.
tn = fidelity_tensornetwork(qcf, ConnectMap(data_qubits,setdiff(1:27, data_qubits), 27))
optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors()) 

# Finally, we contract the tensor network to get the fidelity after error correction.
inf = 1-abs(contract(optnet)[1]/4)

# ## Coherent Error

# # Tensor Network Inference
# This example demonstrates how to define stabilizers, encode data qubits measure syndromes, use tensor network to infer error probability, and correct the error. The main reference is [^Ferris]. 
# We use the $3*3$ surface code as an example. The stabilizers are defined as follows: 

# ## Definition of Stabilizers
using TensorQEC, TensorQEC.Yao
st = stabilizers(SurfaceCode(3,3))

# Then we can generate the encoding circuits of the stabilizers. 'qc' is the encoding circuit, 'data_qubits' are the qubits that we should put initial qubtis in, and 'code' is the structure records information of the encoding circuit.
qc, data_qubits, code = encode_stabilizers(st)
vizcircuit(qc)

# ## Circuit Simulation with Yao.jl
# Create a random qubit state to be encoded.
reg1 = rand_state(1)
# We use 'place_qubits' to create a quantum register. We place the data qubits in 'data_qubits' , and the rest ancilla qubits are in the $|0\rangle$ state.
reg = place_qubits(reg1, data_qubits, nqubits(qc))

# Apply the encoding circuits.
apply!(reg, qc)

# Apply an X error on the third qubit.
apply!(reg, put(9, 3 => X))

# ## Measure the Syndrome and Inference the Error Probability
# We first measure the stabilizers to get the error syndrome.
syn_dict = generate_syndrome_dict(code, syndrome_transform(code, measure_syndrome!(reg, st)))

# Then generate the tensor network for syndrome inference.
cl = clifford_network(qc)

# Define the prior error probability of each physical qubit.
p = fill([0.85, 0.05, 0.05, 0.05], 9)

# Infer the error probability.
pinf = syndrome_inference(cl, syn_dict, p)

# Generate the Pauli string for error correction. Since there is a stabilizer $X_3X_6$, applying $X_3$ or $X_6$ on the coding space are equivalent.
ps_ec_phy = pauli_string_map_iter(correction_pauli_string(9, syn_dict, pinf), qc)

# Or we can simply use the 'inference!' function to measure syndrome and infer error probability.
ps_ec_phy = inference!(reg, code, st, qc, p)

# ## Error Correction
# Apply the error correction.
apply!(reg, Yao.YaoBlocks.Optimise.to_basictypes(ps_ec_phy))

# Finally, we can measure the stabilizers after error correction to check whether the error is corrected.
generate_syndrome_dict(code, syndrome_transform(code, measure_syndrome!(reg, st)))

# And we can calculate the fidelity after error correction to check whether the initial state is recovered.
apply!(reg, qc')
fidelity_after = fidelity(density_matrix(reg, data_qubits), density_matrix(reg1))

# [^Ferris]: Ferris, A. J.; Poulin, D. Tensor Networks and Quantum Error Correction. Phys. Rev. Lett. 2014, 113 (3), 030501. https://doi.org/10.1103/PhysRevLett.113.030501.

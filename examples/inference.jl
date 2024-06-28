# # Inference with Tensor Network
# This example demonstrates how to define stabilizers, encode data qubits measure syndromes, use tensor network to infer error probability, and correct the error. The main reference is [^Ferris]. 

# We take the $3*3$ surface code as an example. We use Yao.jl to simulate a physical quantum devise and perform error correction.

# ## Definition of Stabilizers
using TensorQEC, TensorQEC.Yao
st = stabilizers(SurfaceCode(3,3))

# Then we can generate the encoding circuits of the stabilizers by [`encode_stabilizers`](@ref). `qc` is the encoding circuit, `data_qubits` are the qubits that we should put initial qubtis in, and `code` is the structure records information of the encoding circuit.
qc, data_qubits, code = encode_stabilizers(st)
vizcircuit(qc)

# During the process of getting the encoding circuit, we may adjust the generators of the stabilizer group. The current generators are
TensorQEC.bimatrix2stabilizers(code)

# ## Circuit Simulation with Yao.jl
# Create a random qubit state to be encoded.
reg1 = rand_state(1)
# We use [`place_qubits`](@ref) to create a quantum register. `data_qubits` records the position of data qubits, and the rest ancilla qubits are in the $|0\rangle$ state.
reg = place_qubits(reg1, data_qubits, nqubits(qc))

# Apply the encoding circuits.
apply!(reg, qc)

# Apply an X error on the third qubit.
apply!(reg, put(9, 3 => X))

# ## Measure the Syndrome and Inference the Error Probability
# We first measure the stabilizers to get the error syndrome by [`measure_syndrome!`](@ref). 1 means the stabilizer is not violated, and -1 means the stabilizer is violated. Though the stabilizers are not the same as the initial stabilizers, we can't directly measure the current stabilizers to get the syndrome. The reason is that there may be some long range term in the current stabilizers, which can' be measrued physically. So we still measure the initial stabilizers to get the syndrome.
measure_outcome = measure_syndrome!(reg, st)

# Then we transform the syndrome in the current stabilizers by [`transformed_sydrome_dict`](@ref). The syndrome is transformed to 0 if the measurement outcome is 1, and 1 if the measurement outcome is -1.
syn_dict = transformed_sydrome_dict(measure_outcome, code)

# Now we generate the tensor network for syndrome inference by [`clifford_network`](@ref).
cl = clifford_network(qc)

# Define the prior error probability of each physical qubit. Here we assume the error probability of each qubit is the same. There are probability of 85% that the qubits are correct, and 5% that there is an X error, Y error, or Z error respectively.
p = fill([0.85, 0.05, 0.05, 0.05], 9)

# We can use the [`syndrome_inference`](@ref) function to infer the error probability.
pinf = syndrome_inference(cl, syn_dict, p)

# Generate the Pauli string for error correction. [`correction_pauli_string`](@ref) generates the error Pauli string in the coding space. To correct the error, we still need to transform it to the physical space by [`pauli_string_map_iter`](@ref). The corretion pauli string here is $X_6$. Since there is a stabilizer $X_3X_6$, applying $X_3$ or $X_6$ on the coding space are equivalent.
ps_ec_phy = pauli_string_map_iter(correction_pauli_string(9, syn_dict, pinf), qc)

# Or we can simply use the [`inference`](@ref) function to infer error pauli string in one function.
ps_ec_phy = inference(measure_outcome, code, qc, p)

# ## Error Correction
# Apply the error correction.
apply!(reg, Yao.YaoBlocks.Optimise.to_basictypes(ps_ec_phy))

# Finally, we can measure the stabilizers after error correction to check whether the error is corrected.
measure_syndrome!(reg, st)

# And we can calculate the fidelity after error correction to check whether the initial state is recovered.
apply!(reg, qc')
fidelity_after = fidelity(density_matrix(reg, data_qubits), density_matrix(reg1))

# [^Ferris]: Ferris, A. J.; Poulin, D. Tensor Networks and Quantum Error Correction. Phys. Rev. Lett. 2014, 113 (3), 030501. https://doi.org/10.1103/PhysRevLett.113.030501.

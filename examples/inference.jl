# # Inference with Tensor Network
# This example demonstrates how to define stabilizers, encode data qubits measure syndromes, and use tensor network to infer the most likely error[^Ferris]. 

# We take the $3\times 3$ surface code as an example, and use [Yao.jl](https://github.com/QuantumBFS/Yao.jl) to verify the error correction circuit.

# ## Definition of Stabilizers
using TensorQEC, TensorQEC.Yao
surface_stabilizers = stabilizers(SurfaceCode(3,3))

# Then we can generate the encoding circuits of the stabilizers by [`encode_stabilizers`](@ref).
encoder, data_qubits, bimatrix = encode_stabilizers(surface_stabilizers)
vizcircuit(encoder)
# where `encoder` is the encoding circuit, `data_qubits` are the qubits that we should put initial qubtis in, and `bimatrix` is a [`Bimatrix`](@ref) instance that records information of the encoding circuit. For more details on `Bimatrix`, please check [^Gottesman]. 

# The process of obtaining the encoding circuit requires adjusting the generators of the stabilizer group. The new generators are
TensorQEC.bimatrix2stabilizers(bimatrix)

# ## Circuit Simulation with Yao.jl
# Create a random qubit state to be encoded.
data = rand_state(1)
# We use [`place_qubits`](@ref) to create a quantum register. `data_qubits` records the position of data qubits, and the rest ancilla qubits are in the $|0\rangle$ state.
logic_state = place_qubits(data, data_qubits, nqubits(encoder))

# Apply the encoding circuits.
apply!(logic_state, encoder)

# Apply an X error on the third qubit.
apply!(logic_state, put(9, 3 => X))

# ## Measure the Syndrome and Inference the Error Probability
# We first measure the stabilizers to get the error syndrome by [`measure_syndrome!`](@ref). 1 means the stabilizer is not violated, and -1 means the stabilizer is violated. Though the stabilizers are not the same as the initial stabilizers, we can't directly measure the current stabilizers to get the syndrome. The reason is that there may be some long range term in the current stabilizers, which can' be measrued physically. So we still measure the initial stabilizers to get the syndrome.
measure_outcome = measure_syndrome!(logic_state, surface_stabilizers)

# Then we transform the syndrome in the current stabilizers by [`transformed_sydrome_dict`](@ref). The syndrome is transformed to 0 if the measurement outcome is 1, and 1 if the measurement outcome is -1.
syn_dict = transformed_sydrome_dict(measure_outcome, bimatrix)

# Now we generate the tensor network for syndrome inference by [`clifford_network`](@ref).
tensor_network = clifford_network(encoder)

# Define the prior error probability of each physical qubit. Here we assume the error probability of each qubit is the same. There are probability of 85% that the qubits are correct, and 5% that there is an X error, Y error, or Z error respectively.
prior = fill([0.85, 0.05, 0.05, 0.05], 9)

# We can use the [`syndrome_inference`](@ref) function to infer the error probability.
pinf = syndrome_inference(tensor_network, syn_dict, prior)

# Generate the Pauli string for error correction. [`correction_pauli_string`](@ref) generates the error Pauli string in the coding space. To correct the error, we still need to transform it to the physical space by [`clifford_simulate`](@ref). The corretion pauli string here is $X_6$. Since there is a stabilizer $X_3X_6$, applying $X_3$ or $X_6$ on the coding space are equivalent.
ps_ec_phy = clifford_simulate(correction_pauli_string(9, syn_dict, pinf), encoder).output

# Or we can simply use the [`inference`](@ref) function to infer error pauli string in one function.
ps_ec_phy = inference(measure_outcome, bimatrix, encoder, prior)

# ## Error Correction
# Apply the error correction.
apply!(logic_state, Yao.YaoBlocks.Optimise.to_basictypes(ps_ec_phy))

# Finally, we can measure the stabilizers after error correction to check whether the error is corrected.
measure_syndrome!(logic_state, surface_stabilizers)

# And we can calculate the fidelity after error correction to check whether the initial state is recovered.
apply!(logic_state, encoder')
fidelity_after = fidelity(density_matrix(logic_state, data_qubits), density_matrix(data))

# [^Ferris]: Ferris, A. J.; Poulin, D. Tensor Networks and Quantum Error Correction. Phys. Rev. Lett. 2014, 113 (3), 030501. https://doi.org/10.1103/PhysRevLett.113.030501.
# [^Gottesman]: Gottesman, D. (1997). Stabilizer Codes and Quantum Error Correction (arXiv:quant-ph/9705052). arXiv. http://arxiv.org/abs/quant-ph/9705052

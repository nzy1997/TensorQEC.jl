# # Measurement-Free QEC 
# This example demonstrates how to use tensor network to simulate a error correction process.
# We use the $[[7,1,3]]$ steane code and the measurement-free QEC[^Heußen] as an example. There are non-clifford gates in the quantum circuit, so we use tensor network to simulate the process.

# ## Background Knowledge

# Traditional quantum error correction requires measurment and feed back into
# quantum circuit. Limited by laws of physics, measurements are doomed to be
# costly timewise, e.g. in the NMR computing schme[^Boykin].
# Decoherence may very well happen during measurement.
# Measurement-free quantum error correction was a scheme to circumvent this
# problem.

# Measurement-free quantum error correction was first proposed by Aharonov et.
# al [^Aharonov] in the 90s. Without using measurements, a universal set of
# fault-tolerant gates on encoded qubits was constructed in section 4 of the
# paper [^Aharonov]. They used CSS code and assumed noise is local and
# un-correlated in time, i.e Markovian. They obtained a threshold of $\eta_c
# \approx 10^{-6}$, which is the considerably lower than that of the
# conventional method[^DiVincenzo]. This threshold was later improved to be
# "only about an order of magnitude worse than conventional schemes" [^Ercan] with
# the Bacon-Shor code[^Paz].

# Experimental realizations include

# - Realization on trapped ion platform [^Schindler]

# - Realization on bosonic code qubits. [^Gertler]

# ## Implementation

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
qcm,st_pos  = measure_circuit_steane(data_qubits[1],st;qcen)
vizcircuit(qcm)

# Then we generate correction dictionary for the error correction by [`correction_dict`](@ref).
table = correction_dict(st, 1;et = "Z")

# Now we use [`correction_circuit`](@ref) to generate the measurement-free correction circuit by encoding the truth table on the quantum circuit directly.
num_qubits = nqubits(qcm)
qccr = correction_circuit(table, num_qubits, 3, 25:27, 27)
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
# [^Aharonov]: Aharonov, D. and Ben-Or, M. (1997). Fault-tolerant quantum computation with constant error. In: Proceedings of the twenty-ninth annual ACM symposium on Theory of computing; pp. 176–188.
# [^Boykin]: Roychowdhury, V. P.; Boykin, P.; Vatan, F. and Mor, T. (jul 2004). Fault Tolerant Computation on Ensemble Quantum Computers. In: 2004 International Conference on Dependable Systems and Networks (IEEE Computer Society, Los Alamitos, CA, USA); p. 157.
# [^DiVincenzo]: DiVincenzo, D. P. and Aliferis, P. (2007). Effective fault-tolerant quantum computation with slow measurements. Physical review letters 98, 020501.
# [^Ercan]: Ercan, H. E.; Ghosh, J.; Crow, D.; Premakumar, V. N.; Joynt, R.; Friesen, M. and Coppersmith, S. (2018). Measurement-free implementations of small-scale surface codes for quantum-dot qubits. Physical Review A 97, 012318.
# [^Paz]: Paz-Silva, G. A.; Brennen, G. K. and Twamley, J. (2010). On fault-tolerance with noisy and slow measurements, arXiv preprint arXiv:1002.1536.
# [^Schindler]: Schindler, P.; Barreiro, J. T.; Monz, T.; Nebendahl, V.; Nigg, D.; Chwalla, M.; Hennrich, M. and Blatt, R. (2011). Experimental Repetitive Quantum Error Correction. Science 332, 1059–1061, arXiv:https://www.science.org/doi/pdf/10.1126/science.1203329.
# [^Gertler]: Gertler, J. M.; Baker, B.; Li, J.; Shirol, S.; Koch, J. and Wang, C. (2021). Protecting a bosonic qubit with autonomous quantum error correction. Nature 590, 243–248.

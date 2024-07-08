# # Measurement-Free QEC 
# This example demonstrates how to use tensor network to simulate a error correction process.
# We use the $[[7,1,3]]$ steane code and the measurement-free QEC[^Heußen] as an example. There are non-clifford gates in the quantum circuit, so we use tensor network to simulate the process.

# ## History of Measurement-Free Quantum Error Correction

# Traditional Quantum Error Correction requires measurment and feed back into
# quantum circuit. Limited by laws of physics, measurements are doomed to be costly
# timewise. Decoherence may very well happen during measurement. Measurement-free
# quantum error correction was a scheme to circumvent this problem.

# The concept and proof for fault-tolerant measurement-free schemes have all been
# done in the 90s. The rest of the work are either experimental proof or pushing
# the threshold up.

# ### First of Measurement Free Schemes

# The first paper that mentions measurement free quantum error correction was due
# to Aharonov et. al [^Aharonov]. Not only did they improve Shor's previous result on
# threshold of quantum error correction from being polylogarithmic to circuit
# size and depth to constant, but also did they show in Section 4. in the paper [^Aharonov] that
# no-measurement was necessary for quantum error correction. More concretely, they
# constructed a universal set of fault-tolerant gates without using measurement. In the paper,
# they used CSS code and assumed noise is local and un-correlated in time, i.e
# Markovian. The threshold they obtain is $\eta_c \approx 10^{-6}$.

# In a following paper [^Boykin], measurement free quantum error correction was used to
# circumvent the problem of single molecule measurement being impossible on NMR
# machine. They included concrete example of how to implement a measurement free error correction scheme.


# !!! note "Entanglement Understanding"
#     In a different paper by Peres[^Peres], he linked the measurement of error
#     syndrom being un-necessary with entanglement and teleportation. However, no
#     concrete construction or threshold analysis was given.

# ### Later Developments

# Much later, [^Divincenzo] mentions what measurement free scheme was. However, he did not give any
# reference to paper. In this paper he was considering the effect of slow and fast
# measurement on error correction threshold. [^Ercan] reasoned that slow measurement
# was ok because "measurements can take place concurrently within the many levels
# of concatenation required to achieve fault tolerance." Hence making
# measurement-free scheme only desirable in intermediate scale.

# The next development in measurement free scheme was in [^Paz]. It used Bacon-Shor
# code for example. They have shown that the measurement free scheme was "only about an
# order of magnitude worse than conventional schemes" [^Ercan].

# ### Experimental Realizations and Numerical Estimations

# We collect some experimental realizations and  numerical estimations
# of measurement free quantum error correction.

# [^Schindler] is an experimental paper that realizes measurement-free quantum error
# correction on trapped ions.

# [^Crow] improved the design of Measurement-Free Quantum Error Correction "by using redundant
# syndrome extractions and reported thresholds for three qubit bitflip (BF),
# Bacon-Shor, and Steane codes that are comparable to measurement-based values".[^Ercan]

# [^Omanakuttan] provides usage of Measurement-Free Quantum Error Correction on platform with spins implemnting qudits.

# [^Ercan] benchmarks measurement free quantum error correction on quantum dots
# systems.

# [^Gertler] implemented Measurement-Free Quantum Error Correction in bosonic code.

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
qcm,st_pos, num_qubits = measure_circuit_steane(qcen,data_qubits[1],st,3)
vizcircuit(qcm)

# Then we generate truth table for the error correction by [`make_table`](@ref). For more detials on truth table, please check [Inference with Truth Table](@ref).
table = make_table(st, 1)

# Now we use [`correct_circuit`](@ref) to generate the measurement-free correction circuit by encoding the truth table on the quantum circuit directly.
qccr = correct_circuit(table, collect(st_pos), num_qubits)
vizcircuit(qccr)

# ## Circuit Simulation with Tensor Networks
# We connect the encoding circuit, the measurement circuit, and the correction circuit to form a full circuit. And we apply a Y error on the third qubit after encoding.
qcf=chain(subroutine(num_qubits, qcen, 1:7),put(27,3=>Y),qcm,qccr,subroutine(num_qubits, qcen', 1:7))
vizcircuit(qcf)

# Then we transform the circuit to a tensor network and optimize its contraction order. [`QCInfo`](@ref) records the information of the quantum circuit, including the data qubits and the number of qubits. [`fidelity_tensornetwork`](@ref) constructs the tensor network to calculate the fidelity after error correction.
tn = fidelity_tensornetwork(qcf, QCInfo(data_qubits, 27))

# Finally, we optimize the contraction order and contract the tensor network to get the infidelity after error correction.
optnet = optimize_code(tn, TreeSA(; ntrials=1, niters=5), OMEinsum.MergeVectors()) 
infidelity = 1 - abs(contract(optnet)[1])

# [^Heußen]: Heußen, S., Locher, D. F., & Müller, M. (2024). Measurement-Free Fault-Tolerant Quantum Error Correction in Near-Term Devices. PRX Quantum, 5(1), 010333. https://doi.org/10.1103/PRXQuantum.5.010333
# [^Aharonov]: Aharonov, D. and Ben-Or, M. (1997). Fault-tolerant quantum computation with constant error. In: Proceedings of the twenty-ninth annual ACM symposium on Theory of computing; pp. 176–188.
# [^Boykin]: Roychowdhury, V. P.; Boykin, P.; Vatan, F. and Mor, T. (jul 2004). Fault Tolerant Computation on Ensemble Quantum Computers. In: 2004 International Conference on Dependable Systems and Networks (IEEE Computer Society, Los Alamitos, CA, USA); p. 157.
# [^Peres]: Peres, A. (1998). Quantum disentanglement and computation. Superlattices and Microstructures 23, 373–379.
# [^Divincenzo]: DiVincenzo, D. P. and Aliferis, P. (2007). Effective fault-tolerant quantum computation with slow measurements. Physical review letters 98, 020501.
# [^Ercan]: Ercan, H. E.; Ghosh, J.; Crow, D.; Premakumar, V. N.; Joynt, R.; Friesen, M. and Coppersmith, S. (2018). Measurement-free implementations of small-scale surface codes for quantum-dot qubits. Physical Review A 97, 012318.
# [^Paz]: Paz-Silva, G. A.; Brennen, G. K. and Twamley, J. (2010). On fault-tolerance with noisy and slow measurements, arXiv preprint arXiv:1002.1536.
# [^Schindler]: Schindler, P.; Barreiro, J. T.; Monz, T.; Nebendahl, V.; Nigg, D.; Chwalla, M.; Hennrich, M. and Blatt, R. (2011). Experimental Repetitive Quantum Error Correction. Science 332, 1059–1061, arXiv:https://www.science.org/doi/pdf/10.1126/science.1203329.
# [^Omanakuttan]: Omanakuttan, S.; Buchemmavari, V.; Gross, J. A.; Deutsch, I. H. and Marvian, M. (2024). Fault-Tolerant Quantum Computation Using Large Spin-Cat Codes. PRX Quantum 5, 020355.
# [^Crow]: Crow, D.; Joynt, R. and Saffman, M. (2016). Improved error thresholds for measurement-free error correction. Physical review letters 117, 130503.
# [^Gertler]: Gertler, J. M.; Baker, B.; Li, J.; Shirol, S.; Koch, J. and Wang, C. (2021). Protecting a bosonic qubit with autonomous quantum error correction. Nature 590, 243–248.

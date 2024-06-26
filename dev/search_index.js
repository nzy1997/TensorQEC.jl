var documenterSearchIndex = {"docs":
[{"location":"generated/codes/","page":"codes","title":"codes","text":"EditURL = \"../../../examples/codes.jl\"","category":"page"},{"location":"generated/codes/#QEC-Codes","page":"codes","title":"QEC Codes","text":"","category":"section"},{"location":"generated/codes/","page":"codes","title":"codes","text":"We provide a number of quantum error correction codes.","category":"page"},{"location":"generated/codes/","page":"codes","title":"codes","text":"using TensorQEC","category":"page"},{"location":"generated/codes/#Toric-Code","page":"codes","title":"Toric Code","text":"","category":"section"},{"location":"generated/codes/","page":"codes","title":"codes","text":"The Toric code is a 2D topological code. We can define a Toric code instance  by ToricCode(m, n), where m and n are the number of rows and columns of the Toric code.","category":"page"},{"location":"generated/codes/","page":"codes","title":"codes","text":"tc = ToricCode(2, 3)","category":"page"},{"location":"generated/codes/","page":"codes","title":"codes","text":"Here is a schematic diagram of 2*3 Toric code: (Image: )","category":"page"},{"location":"generated/codes/","page":"codes","title":"codes","text":"The Toric code has two types of stabilizers: X stabilizers and Z stabilizers. X stabilizers are plaquettes of the lattice, and Z stabilizers are vertices of the lattice. We can get the stabilizers of the toric code by","category":"page"},{"location":"generated/codes/","page":"codes","title":"codes","text":"st = stabilizers(tc)","category":"page"},{"location":"generated/codes/#Surface-Code","page":"codes","title":"Surface Code","text":"","category":"section"},{"location":"generated/codes/","page":"codes","title":"codes","text":"The surface code is a 2D topological code. Similarly to Toric code, we can define a surface code instance by SurfaceCode(m, n) and get the stabilizers of the surface code by stabilizers(sc).","category":"page"},{"location":"generated/codes/","page":"codes","title":"codes","text":"sc = SurfaceCode(3, 3)\nst = stabilizers(sc)","category":"page"},{"location":"generated/codes/","page":"codes","title":"codes","text":"Here is a schematic diagram of 3*3 surface code: (Image: )","category":"page"},{"location":"generated/codes/#Others","page":"codes","title":"Others","text":"","category":"section"},{"location":"generated/codes/","page":"codes","title":"codes","text":"We also includes Shor code, Steane code and 832 code. The usage is similar to the above examples. Shor Code:","category":"page"},{"location":"generated/codes/","page":"codes","title":"codes","text":"shor = ShorCode()\nst = stabilizers(shor)","category":"page"},{"location":"generated/codes/","page":"codes","title":"codes","text":"Steane Code: (Image: )","category":"page"},{"location":"generated/codes/","page":"codes","title":"codes","text":"steane = SteaneCode()\nst = stabilizers(steane)","category":"page"},{"location":"generated/codes/","page":"codes","title":"codes","text":"832","category":"page"},{"location":"generated/codes/","page":"codes","title":"codes","text":"Code: (Image: )","category":"page"},{"location":"generated/codes/","page":"codes","title":"codes","text":"code832 = Code832()\nst = stabilizers(code832)","category":"page"},{"location":"generated/codes/","page":"codes","title":"codes","text":"","category":"page"},{"location":"generated/codes/","page":"codes","title":"codes","text":"This page was generated using Literate.jl.","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"EditURL = \"../../../examples/repetition_code3.jl\"","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"using TensorQEC, TensorQEC.Yao","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"define the stabilizers","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"qubit_num = 3\nst = [PauliString((1,4,4)),PauliString((4,1,4))]\n@info \"stabilizers: $st\"","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"Generate the encoding circuits of the stabilizers","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"qc, data_qubits, code = encode_stabilizers(st)\n@info \"encoding circuits: $qc, data qubits: $data_qubits\"","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"Create a quantum register. Qubits in \"data_qubits\" are randomly initilized, and the rest ancilla qubits are in the |0> state.","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"reg = join(rand_state(1), zero_state(2))  # join(qubit3, qubit2, qubit1)","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"Apply the encoding circuits.","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"regcopy = copy(reg)\napply!(reg, qc)","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"Apply a X error on the third qubit","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"apply!(reg, put(qubit_num, 3=>X))\n@info \"applied X error on the third qubit\"","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"Measure the syndrome","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"measure_outcome=measure_syndrome!(reg, st)\n@info \"measured syndrome: $measure_outcome\"","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"Generate the syndrome dictionary","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"syn_dict=TensorQEC.generate_syndrome_dict(code,syndrome_transform(code, measure_outcome))","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"Generate the tensor network for syndrome inference","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"cl = clifford_network(qc)\np = fill([0.85,0.05,0.05,0.05],qubit_num)\npinf = syndrome_inference(cl, syn_dict, p)\n@info \"inferred error probability: $pinf\"","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"Generate the Pauli string for error correction","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"ps_ec_phy = TensorQEC.pauli_string_map_iter(correction_pauli_string(qubit_num, syn_dict, pinf), qc)\n@info \"Pauli string for error correction: $ps_ec_phy\"","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"Apply the error correction","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"apply!(reg, Yao.YaoBlocks.Optimise.to_basictypes(ps_ec_phy))","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"Measure the syndrome after error correction","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"syndrome_result = measure_syndrome!(reg, st)\n@info \"measured syndrome: $syndrome_result\"\napply!(reg, qc')\nfidelity_after = fidelity(density_matrix(reg, [data_qubits...]), density_matrix(regcopy, [data_qubits...]))\n@info \"fidelity after error correction: $fidelity_after\"","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"","category":"page"},{"location":"generated/repetition_code3/","page":"-","title":"-","text":"This page was generated using Literate.jl.","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"EditURL = \"../../../examples/simulation.jl\"","category":"page"},{"location":"generated/simulation/#Tensor-Network-Simulation","page":"Simulation","title":"Tensor Network Simulation","text":"","category":"section"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"This example demonstrates how to use tensor network to simulate the error correction process. We use the 713 steane code and the measurement-free QEC[Heußen] as an example. There are non-clifford gates in the quantum circuit, so we use tensor network to simulate the process.","category":"page"},{"location":"generated/simulation/#Definition-of-Stabilizers-and-Encoding-Circuits","page":"Simulation","title":"Definition of Stabilizers and Encoding Circuits","text":"","category":"section"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"using TensorQEC, TensorQEC.Yao\nusing TensorQEC.OMEinsum\nst = stabilizers(SteaneCode())","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"Generate the encoding circuits of the stabilizers.","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"qcen, data_qubits, code = encode_stabilizers(st)\nvizcircuit(qcen)","category":"page"},{"location":"generated/simulation/#Syndrome-Extraction-and-Measurement-Free-Error-Correction","page":"Simulation","title":"Syndrome Extraction and Measurement-Free Error Correction","text":"","category":"section"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"First, we generate the steane measurement circuit and 'st_pos' records the ancilla qubits that store the measurement results of the stabilizers.","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"qcm,st_pos, num_qubits = measure_circuit_steane(qcen,data_qubits[1],st,3)\nvizcircuit(qcm)","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"Then we generate truth table for the error correction.","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"table = make_table(TensorQEC.stabilizers2bimatrix(st).matrix, 1)","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"Now we can generate the measurement-free correction circuit by encoding the truth table on the quantum circuit directly.","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"qccr = correct_circuit(table, collect(st_pos), num_qubits, 6, 7)\nvizcircuit(qccr)","category":"page"},{"location":"generated/simulation/#Circuit-Simulation-with-Tensor-Networks","page":"Simulation","title":"Circuit Simulation with Tensor Networks","text":"","category":"section"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"We connect the encoding circuit, the measurement circuit, and the correction circuit to form a full circuit. And we apply a Y error on the third qubit after encoding.","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"qcf=chain(subroutine(num_qubits, qcen, 1:7),put(27,3=>Y),qcm,qccr,subroutine(num_qubits, qcen', 1:7))\nvizcircuit(qcf)","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"Then we transform the circuit to a tensor network and optimize its contraction order.","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"tn = fidelity_tensornetwork(qcf, ConnectMap(data_qubits,setdiff(1:27, data_qubits), 27))\noptnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"Finally, we contract the tensor network to get the fidelity after error correction.","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"inf = 1-abs(contract(optnet)[1]/4)","category":"page"},{"location":"generated/simulation/#Coherent-Error","page":"Simulation","title":"Coherent Error","text":"","category":"section"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"Since coherent error is also non-clifford, we can use the same method to simulate the error correction process. First, we can generate the coherent error unitaries. 'Pair' records the error pairs of the gates. 'vector' records the error rates of the gates. We add coherent error to X, Y, Z, H, CZ, CNOT, CCZ, Toffoli gates by default. We customize our error gates to only single qubit gates.","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"pairs, vector = error_pairs(1e-5; gates = [X,Y,Z,H])","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"Then we can generate the error quantum circuit.","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"qc = chain(subroutine(num_qubits, qcen, 1:7), qcm,qccr,subroutine(num_qubits, qcen', 1:7))\neqc = error_quantum_circuit(qc, pairs)\nvizcircuit(eqc)","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"Finally, we can simulate the error correction process with the coherent error.","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"tn = fidelity_tensornetwork(eqc, ConnectMap(data_qubits,setdiff(1:27, data_qubits), 27))\noptnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())\ninf = 1-abs(contract(optnet)[1]/4)","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"[Heußen]: Heußen, S., Locher, D. F., & Müller, M. (2024). Measurement-Free Fault-Tolerant Quantum Error Correction in Near-Term Devices. PRX Quantum, 5(1), 010333. https://doi.org/10.1103/PRXQuantum.5.010333","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"","category":"page"},{"location":"generated/simulation/","page":"Simulation","title":"Simulation","text":"This page was generated using Literate.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = TensorQEC","category":"page"},{"location":"#TensorQEC","page":"Home","title":"TensorQEC","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package utilizes the tensor network to study the properties of quantum error correction (QEC).The main features include","category":"page"},{"location":"","page":"Home","title":"Home","text":"Quantum error correction code decoder with tensor network,\nQuantum circuit simulation with tensor network to estimate the threshold of QEC.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Also, we include more general QEC tools, including","category":"page"},{"location":"","page":"Home","title":"Home","text":"Commonly used QEC code stabilizer generators,\nQEC code distance calculation,\nQEC encoding circuit construction,\nDecoding truth table construction,\nMeasurement circuit construction.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Documentation for TensorQEC.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [TensorQEC]","category":"page"},{"location":"#TensorQEC.PauliString","page":"Home","title":"TensorQEC.PauliString","text":"PauliString{N} <: CompositeBlock{2}\n\nA Pauli string is a tensor product of Pauli gates, e.g. XYZ. The matrix representation of a Pauli string is evaluated as\n\nA = bigotimes_i=1^N sigma_idsN-i+1\n\nwhere ids is the array of integers representing the Pauli gates. Note the order of ids is following the little-endian convention, i.e. the first qubit is the least significant qubit. For example, the Pauli string XYZ has matrix representation Z ⊗ Y ⊗ X.\n\nFields\n\nids::NTuple{N, Int}: the array of integers (1-4) representing the Pauli gates.\n1: I (σ_0)\n2: X (σ_1)\n3: Y (σ_2)\n4: Z (σ_3)\n\n\n\n\n\n","category":"type"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"EditURL = \"../../../examples/inference.jl\"","category":"page"},{"location":"generated/inference/#Tensor-Network-Inference","page":"Inference","title":"Tensor Network Inference","text":"","category":"section"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"This example demonstrates how to define stabilizers, encode data qubits measure syndromes, use tensor network to infer error probability, and correct the error. The main reference is [Ferris]. We use the 3*3 surface code as an example. The stabilizers are defined as follows:","category":"page"},{"location":"generated/inference/#Definition-of-Stabilizers","page":"Inference","title":"Definition of Stabilizers","text":"","category":"section"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"using TensorQEC, TensorQEC.Yao\nst = stabilizers(SurfaceCode(3,3))","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"Then we can generate the encoding circuits of the stabilizers. 'qc' is the encoding circuit, 'data_qubits' are the qubits that we should put initial qubtis in, and 'code' is the structure records information of the encoding circuit.","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"qc, data_qubits, code = encode_stabilizers(st)\nvizcircuit(qc)","category":"page"},{"location":"generated/inference/#Circuit-Simulation-with-Yao.jl","page":"Inference","title":"Circuit Simulation with Yao.jl","text":"","category":"section"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"Create a random qubit state to be encoded.","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"reg1 = rand_state(1)","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"We use 'placequbits' to create a quantum register. We place the data qubits in 'dataqubits' , and the rest ancilla qubits are in the 0rangle state.","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"reg = place_qubits(reg1, data_qubits, nqubits(qc))","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"Apply the encoding circuits.","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"apply!(reg, qc)","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"Apply an X error on the third qubit.","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"apply!(reg, put(9, 3 => X))","category":"page"},{"location":"generated/inference/#Measure-the-Syndrome-and-Inference-the-Error-Probability","page":"Inference","title":"Measure the Syndrome and Inference the Error Probability","text":"","category":"section"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"We first measure the stabilizers to get the error syndrome.","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"syn_dict = generate_syndrome_dict(code, syndrome_transform(code, measure_syndrome!(reg, st)))","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"Then generate the tensor network for syndrome inference.","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"cl = clifford_network(qc)","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"Define the prior error probability of each physical qubit.","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"p = fill([0.85, 0.05, 0.05, 0.05], 9)","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"Infer the error probability.","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"pinf = syndrome_inference(cl, syn_dict, p)","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"Generate the Pauli string for error correction. Since there is a stabilizer X_3X_6, applying X_3 or X_6 on the coding space are equivalent.","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"ps_ec_phy = pauli_string_map_iter(correction_pauli_string(9, syn_dict, pinf), qc)","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"Or we can simply use the 'inference!' function to measure syndrome and infer error probability.","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"ps_ec_phy = inference!(reg, code, st, qc, p)","category":"page"},{"location":"generated/inference/#Error-Correction","page":"Inference","title":"Error Correction","text":"","category":"section"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"Apply the error correction.","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"apply!(reg, Yao.YaoBlocks.Optimise.to_basictypes(ps_ec_phy))","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"Finally, we can measure the stabilizers after error correction to check whether the error is corrected.","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"generate_syndrome_dict(code, syndrome_transform(code, measure_syndrome!(reg, st)))","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"And we can calculate the fidelity after error correction to check whether the initial state is recovered.","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"apply!(reg, qc')\nfidelity_after = fidelity(density_matrix(reg, data_qubits), density_matrix(reg1))","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"[Ferris]: Ferris, A. J.; Poulin, D. Tensor Networks and Quantum Error Correction. Phys. Rev. Lett. 2014, 113 (3), 030501. https://doi.org/10.1103/PhysRevLett.113.030501.","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"","category":"page"},{"location":"generated/inference/","page":"Inference","title":"Inference","text":"This page was generated using Literate.jl.","category":"page"}]
}

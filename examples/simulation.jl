using TensorQEC, TensorQEC.Yao
using TensorQEC.OMEinsum
# define the stabilizers
st = stabilizers(SteaneCode())
# Generate the encoding circuits of the stabilizers
qcen, data_qubits, code = encode_stabilizers(st)
qcm,st_pos, num_qubits = measure_circuit_steane(qcen,data_qubits[1],st,3)

bimat = TensorQEC.stabilizers2bimatrix(st)
table = make_table(bimat.matrix, 1)
qccr = TensorQEC.correct_circuit(table, collect(st_pos), num_qubits, 6, 7)

qcf=chain(subroutine(num_qubits, qcen, 1:7),put(27,3=>Y),qcm,qccr,subroutine(num_qubits, qcen', 1:7))
YaoPlots.CircuitStyles.r[] = 0.3
vizcircuit(qcf)

tn = fidelity_tensornetwork(qcf, ConnectMap(data_qubits,setdiff(1:27, data_qubits), 27))
optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors()) 
inf = 1-abs(contract(optnet)[1]/4)


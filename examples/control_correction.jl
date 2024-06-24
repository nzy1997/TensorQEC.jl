using TensorQEC, TensorQEC.Yao
# define the stabilizers
st = stabilizers(SteaneCode())
# Generate the encoding circuits of the stabilizers
qc, data_qubits, code = encode_stabilizers(st)
qcen, data_qubits, code = encode_stabilizers(st)
qcm,st_pos, num_qubits = measure_circuit_steane(qcen,st,3)

gates = chain(subroutine(num_qubits, qcen, 1:18), qcm)
Optimise.simplify(gates, rules=[Optimise.to_basictypes, Optimise.eliminate_nested])

bimat = stabilizers2bimatrix(st_me)
table = make_table(bimat.matrix, 1)
qccr = correct_circuit(table, st_pos, num_qubits, 18, 18)

qcf=chain(qcm,qccr)
YaoPlots.CircuitStyles.r[] = 0.3
vizcircuit(qcf; starting_texts = 1:num_qubits, filename = "examples/mf.svg")

measure_circuit_steane(qcen,st_me)
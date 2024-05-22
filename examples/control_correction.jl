using TensorQEC
using TensorQEC.Yao

t = TensorQEC.ToricCode(3, 3)
st = TensorQEC.stabilizers(t)
qcen, data_qubits, code = TensorQEC.encode_stabilizers(st)
st_me = TensorQEC.stabilizers(t,linearly_independent = false)
qcm,st_pos, num_qubits = measure_circuit_fault_tol(st_me)

gates = chain(subroutine(num_qubits, qcen, 1:18), qcm)
Optimise.simplify(gates, rules=[Optimise.to_basictypes, Optimise.eliminate_nested])

bimat = TensorQEC.stabilizers2bimatrix(st_me)
table = make_table(bimat.matrix, 1)
qccr = correct_circuit(table, st_pos, num_qubits, 18, 18)

qcf=chain(qcm,qccr)
YaoPlots.CircuitStyles.r[] = 0.3
vizcircuit(qcf; starting_texts = 1:num_qubits, filename = "examples/mf.svg")

measure_circuit_steane(qcen,st_me)
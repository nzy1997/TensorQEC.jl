using TensorQEC

t = TensorQEC.ToricCode(3, 3)
st = TensorQEC.stabilizers(t)
qcen, data_qubits, code = TensorQEC.encode_stabilizers(st)
stli = TensorQEC.stabilizers(t,linearly_independent = false)
qcm,st_pos, num_qubits = measure_circuit_fault_tol(stli)

ps = TensorQEC.paulistring(90, 2, 3)
ps2,vals=TensorQEC.clifford_simulate(ps, qcm)

# gates = chain(subroutine(90, qcen, 1:18), qcm)
# Optimise.simplify(gates, rules=[Optimise.to_basictypes, Optimise.eliminate_nested])
# qc[1] |> occupied_locs

bimat = TensorQEC.stabilizers2bimatrix(stli)
table = make_table(bimat.matrix, 1)
qccr = correct_circuit(table, st_pos, 90, 18, 18)

qcf=chain(gates,qccr)
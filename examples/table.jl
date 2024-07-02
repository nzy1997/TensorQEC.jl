# # Inference with Truth Table
# This example demonstrates how to define stabilizers, encode data qubits measure syndromes, use truth table to infer error type and position, and correct the error.

# We take the [[7,1,3]] Steane code as an example. We use Yao.jl to simulate a physical quantum devise and perform error correction.

# ## Definition of Stabilizers
using TensorQEC, TensorQEC.Yao
st = stabilizers(SteaneCode())

# Then we can generate the encoding circuits of the stabilizers by [`encode_stabilizers`](@ref). `qc` is the encoding circuit, `data_qubits` are the qubits that we should put initial qubtis in, and `code` is the structure records information of the encoding circuit.
qc, data_qubits, code = encode_stabilizers(st)
vizcircuit(qc)

# ## Construction of the Truth Table
# We can generate the truth table for the error correction by [`make_table`](@ref). The [`TruthTable`](@ref) is a struct that records the truth table, the number of qubits, the number of stabilizers, and maximum error legth of the errors.
table = make_table(st, 1)

# We can save the truth table to a file by [`save_table`](@ref), and load the truth table from a file by [`load_table`](@ref).
save_table(table, "test_table.txt")
table2 = load_table("test_table.txt", 9, 8, 1)
table.table == table2.table

# ## Circuit Simulation with Yao.jl
# Create a random qubit state to be encoded.
reg1 = rand_state(1)
# We use [`place_qubits`](@ref) to create a quantum register. `data_qubits` records the position of data qubits, and the rest ancilla qubits are in the $|0\rangle$ state.
reg = place_qubits(reg1, data_qubits, nqubits(qc))

# Apply the encoding circuits.
apply!(reg, qc)

# Apply an X error on the third qubit.
apply!(reg, put(7, 5 => Y))

# ## Measure the Syndrome and Inference the Error Type and Position
# We first measure the stabilizers to get the error syndrome by [`measure_syndrome!`](@ref). 1 means the stabilizer is not violated, and -1 means the stabilizer is violated.
measure_outcome = measure_syndrome!(reg, st)

# The measrue outcome shows that the stabilizer 4 and 6 are violated. According to the truth table, the error is $X_5$, which is exactly the error we applied. We can use [`table_inference`](@ref) to look up the syndromes in the truth table and infer the error type and position.
table_inference(table,measure_outcome)

# If we look up the syndrome that is not in the truth table, it will return `nothing`.
table_inference(table, [-1,-1,-1,-1,1,1])
# ## Error Correction
# Then the following error correction is trivial. We apply the error correction.
apply!(reg, put(7, 5 => X))

# Measure the stabilizers after error correction to check whether the error is corrected.
measure_syndrome!(reg, st)

# And we can calculate the fidelity after error correction to check whether the initial state is recovered.
apply!(reg, qc')
fidelity_after = fidelity(density_matrix(reg, data_qubits), density_matrix(reg1))

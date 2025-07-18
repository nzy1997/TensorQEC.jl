using TensorQEC
using Test
using Yao

circuit_str = """
# Distribute a Bell Pair.
H 0
CNOT 0 3

# Sender creates an arbitrary qubit state to send.
H 1
S 1

# Sender performs a Bell Basis measurement.
CNOT 0 1
H 0
M 0 1  # Measure both of the sender's qubits.

# Receiver performs frame corrections based on measurement results.
CZ rec[-2] 3
CNOT rec[-1] 3
"""

qc = TensorQEC.parse_stim_string(circuit_str, 4)
vizcircuit(qc)

circuit_str = """
# Measure the parities of adjacent data qubits.
# Data qubits are 0, 2, 4, 6.
# Measurement qubits are 1, 3, 5.
CNOT 0 1 2 3 4 5
CNOT 2 1 4 3 6 5
MR 1 3 5

# Annotate that the measurements should be deterministic.
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]

# Perform 3 more rounds of measurements.
REPEAT 3 {
    # Measure the parities of adjacent data qubits.
    CNOT 0 1 2 3 4 5
    CNOT 2 1 4 3 6 5
    MR 1 3 5

    # Annotate that the measurements should agree with previous round.
    DETECTOR rec[-3] rec[-6]
    DETECTOR rec[-2] rec[-5]
    DETECTOR rec[-1] rec[-4]
    
}

# Measure data qubits.
M 0 2 4 6

# Annotate that the data measurements should agree with the parity measurements.
DETECTOR rec[-3] rec[-4] rec[-7]
DETECTOR rec[-2] rec[-3] rec[-6]
DETECTOR rec[-1] rec[-2] rec[-5]

# Declare one of the data qubit measurements to a logical measurement result.
OBSERVABLE_INCLUDE(0) rec[-1]
"""

qc = TensorQEC.parse_stim_string(circuit_str, 7)
vizcircuit(qc)

circuit_str = """
# Generated repetition_code circuit.
# task: memory
# rounds: 1000
# distance: 3
# before_round_data_depolarization: 0
# before_measure_flip_probability: 0
# after_reset_flip_probability: 0
# after_clifford_depolarization: 0.001
# layout:
# L0 Z1 d2 Z3 d4 Z5 d6
# Legend:
#     d# = data qubit
#     L# = data qubit with logical observable crossing
#     Z# = measurement qubit
R 0 1 2 3 4 5 6
TICK
CX 0 1 2 3 4 5
DEPOLARIZE2(0.001) 0 1 2 3 4 5
TICK
CX 2 1 4 3 6 5
DEPOLARIZE2(0.001) 2 1 4 3 6 5
TICK
MR 1 3 5
DETECTOR(1, 0) rec[-3]
DETECTOR(3, 0) rec[-2]
DETECTOR(5, 0) rec[-1]
REPEAT 2 {
    TICK
    CX 0 1 2 3 4 5
    DEPOLARIZE2(0.001) 0 1 2 3 4 5
    TICK
    CX 2 1 4 3 6 5
    DEPOLARIZE2(0.001) 2 1 4 3 6 5
    TICK
    MR 1 3 5
    SHIFT_COORDS(0, 1)
    DETECTOR(1, 0) rec[-3] rec[-6]
    DETECTOR(3, 0) rec[-2] rec[-5]
    DETECTOR(5, 0) rec[-1] rec[-4]
}
M 0 2 4 6
DETECTOR(1, 1) rec[-3] rec[-4] rec[-7]
DETECTOR(3, 1) rec[-2] rec[-3] rec[-6]
DETECTOR(5, 1) rec[-1] rec[-2] rec[-5]
OBSERVABLE_INCLUDE(0) rec[-1]
"""

qc = TensorQEC.parse_stim_string(circuit_str, 7);
vizcircuit(qc)
circuit_str = """
# Generated surface_code circuit.
# task: rotated_memory_x
# rounds: 1000
# distance: 3
# before_round_data_depolarization: 0
# before_measure_flip_probability: 0
# after_reset_flip_probability: 0
# after_clifford_depolarization: 0.001
# layout:
#                 X25
#     L15     d17     d19
# Z14     X16     Z18
#     L8      d10     d12
#         Z9      X11     Z13
#     L1      d3      d5 
#         X2 
# Legend:
#     d# = data qubit
#     L# = data qubit with logical observable crossing
#     X# = measurement qubit (X stabilizer)
#     Z# = measurement qubit (Z stabilizer)
QUBIT_COORDS(1, 1) 1
QUBIT_COORDS(2, 0) 2
QUBIT_COORDS(3, 1) 3
QUBIT_COORDS(5, 1) 5
QUBIT_COORDS(1, 3) 8
QUBIT_COORDS(2, 2) 9
QUBIT_COORDS(3, 3) 10
QUBIT_COORDS(4, 2) 11
QUBIT_COORDS(5, 3) 12
QUBIT_COORDS(6, 2) 13
QUBIT_COORDS(0, 4) 14
QUBIT_COORDS(1, 5) 15
QUBIT_COORDS(2, 4) 16
QUBIT_COORDS(3, 5) 17
QUBIT_COORDS(4, 4) 18
QUBIT_COORDS(5, 5) 19
QUBIT_COORDS(4, 6) 25
RX 1 3 5 8 10 12 15 17 19
R 2 9 11 13 14 16 18 25
TICK
H 2 11 16 25
DEPOLARIZE1(0.001) 2 11 16 25
TICK
CX 2 3 16 17 11 12 15 14 10 9 19 18
DEPOLARIZE2(0.001) 2 3 16 17 11 12 15 14 10 9 19 18
TICK
CX 2 1 16 15 11 10 8 14 3 9 12 18
DEPOLARIZE2(0.001) 2 1 16 15 11 10 8 14 3 9 12 18
TICK
CX 16 10 11 5 25 19 8 9 17 18 12 13
DEPOLARIZE2(0.001) 16 10 11 5 25 19 8 9 17 18 12 13
TICK
CX 16 8 11 3 25 17 1 9 10 18 5 13
DEPOLARIZE2(0.001) 16 8 11 3 25 17 1 9 10 18 5 13
TICK
H 2 11 16 25
DEPOLARIZE1(0.001) 2 11 16 25
TICK
MR 2 9 11 13 14 16 18 25
DETECTOR(2, 0, 0) rec[-8]
DETECTOR(2, 4, 0) rec[-3]
DETECTOR(4, 2, 0) rec[-6]
DETECTOR(4, 6, 0) rec[-1]
REPEAT 2 {
    TICK
    H 2 11 16 25
    DEPOLARIZE1(0.001) 2 11 16 25
    TICK
    CX 2 3 16 17 11 12 15 14 10 9 19 18
    DEPOLARIZE2(0.001) 2 3 16 17 11 12 15 14 10 9 19 18
    TICK
    CX 2 1 16 15 11 10 8 14 3 9 12 18
    DEPOLARIZE2(0.001) 2 1 16 15 11 10 8 14 3 9 12 18
    TICK
    CX 16 10 11 5 25 19 8 9 17 18 12 13
    DEPOLARIZE2(0.001) 16 10 11 5 25 19 8 9 17 18 12 13
    TICK
    CX 16 8 11 3 25 17 1 9 10 18 5 13
    DEPOLARIZE2(0.001) 16 8 11 3 25 17 1 9 10 18 5 13
    TICK
    H 2 11 16 25
    DEPOLARIZE1(0.001) 2 11 16 25
    TICK
    MR 2 9 11 13 14 16 18 25
    SHIFT_COORDS(0, 0, 1)
    DETECTOR(2, 0, 0) rec[-8] rec[-16]
    DETECTOR(2, 2, 0) rec[-7] rec[-15]
    DETECTOR(4, 2, 0) rec[-6] rec[-14]
    DETECTOR(6, 2, 0) rec[-5] rec[-13]
    DETECTOR(0, 4, 0) rec[-4] rec[-12]
    DETECTOR(2, 4, 0) rec[-3] rec[-11]
    DETECTOR(4, 4, 0) rec[-2] rec[-10]
    DETECTOR(4, 6, 0) rec[-1] rec[-9]
}
MX 1 3 5 8 10 12 15 17 19
DETECTOR(2, 0, 1) rec[-8] rec[-9] rec[-17]
DETECTOR(2, 4, 1) rec[-2] rec[-3] rec[-5] rec[-6] rec[-12]
DETECTOR(4, 2, 1) rec[-4] rec[-5] rec[-7] rec[-8] rec[-15]
DETECTOR(4, 6, 1) rec[-1] rec[-2] rec[-10]
OBSERVABLE_INCLUDE(0) rec[-3] rec[-6] rec[-9]
"""

qc = TensorQEC.parse_stim_string(circuit_str, 26)
vizcircuit(qc)

qc = TensorQEC.parse_stim_file("data/testcir.stim", 144);

m = Measure(1)
reg = ArrayReg(ComplexF64[0,1])
c = TensorQEC.condition(m, X, nothing)
@show c

qc = chain(put(3, 1=>X), put(3, 3=>X))
push!(qc,put(3,1 => m))
push!(qc,put(3,3 => c))
vizcircuit(qc)

qc2 = chain(put(3, 1=>X), put(3, 3=>X))
ccj = ComplexConj(X)
push!(qc2,put(3,3 => ccj))
vizcircuit(qc2)

m1 = Measure(1)
m2 = Measure(1)
db = TensorQEC.DetectorBlock([m1,m2])

using Yao
qc3 = chain(2)
push!(qc3,put(2,1 => Measure(1)))
push!(qc3,put(2,2 => Measure(1)))
vizcircuit(qc3)

qc3 = chain(2)
push!(qc3,put(2,(1,2) => Measure(2)))
vizcircuit(qc3)
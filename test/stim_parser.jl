using TensorQEC
using Test

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
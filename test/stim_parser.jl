using TensorQEC
using Test

circuit_str = """
H 0
CNOT 0 1
M 0 1
"""

TensorQEC.parse_stim_string(circuit_str)
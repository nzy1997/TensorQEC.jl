module TensorQEC

using TensorInference
using TensorInference: Factor
using Yao
using Yao.ConstGate: PauliGate
using Base.Iterators: product
using LinearAlgebra
using Combinatorics
using DelimitedFiles
using OMEinsum
using Yao.YaoBlocks.Optimise

# pauli basis
export pauli_basis, pauli_decomposition, pauli_mapping
import Yao.YaoArrayRegister.StaticArrays: SizedVector
import Yao.YaoArrayRegister.LuxurySparse

# Mod
export Mod2

# tensor network
export clifford_network, CliffordNetwork, generate_tensor_network, circuit2tensornetworks
export ExtraTensor, UNITY4, PXY, PIZ, PXY_UNITY2, PIZ_UNITY2
export densitymatrix2sumofpaulis, SumOfPaulis
export PauliString, PauliGroup, isanticommute

# inference
export syndrome_inference, measure_syndrome!,correction_pauli_string 

# encoder
export Bimatrix, toric_code, stabilizers, syndrome_transform, encode_stabilizers
export ToricCode, SurfaceCode, ShorCode,SteaneCode

# measurement
export  measure_circuit_fault_tol, correct_circuit, measure_circuit_steane,measure_circuit

# tablemake
export make_table, save_table, read_table

# clifford group
export pauli_group, clifford_group, clifford_simulate

# qc2ein
export ComplexConj, SymbolRecorder,IdentityRecorder, ein_circ, ConnectMap, qc2enisum

# coerror
export coherent_error_unitary, error_quantum_circuit,toput, error_pairs

include("mod2.jl")
include("paulistring.jl")
include("cliffordgroup.jl")
include("paulibasis.jl")
include("tensornetwork.jl")
include("encoder.jl")
include("inferences.jl")
include("measurement.jl")
include("tablemake.jl")
                             
include("qc2ein.jl")
include("coerror.jl")
end

module TensorQEC

using QECCore
using TensorInference
using TensorInference: Factor
using Yao, Yao.YaoAPI
using Yao.ConstGate: PauliGate
using Base.Iterators: product
using LinearAlgebra
using Combinatorics
using DelimitedFiles
using OMEinsum
using Yao.YaoBlocks.Optimise
using Yao.YaoBlocks.LuxurySparse: PermMatrixCSC
using PrettyTables
using Optimisers
using BitBasis
using YaoToEinsum
# using Random

using Graphs
using SimpleWeightedGraphs
using JuMP
using SCIP
import Yao.YaoArrayRegister.StaticArrays: SizedVector
import Yao.YaoArrayRegister.LuxurySparse

# reexport some YaoAPI
export mat

# Mod
export Mod2

# pauli basis
export pauli_basis, pauli_decomposition, pauli_repr
export Pauli, SumOfPaulis, @P_str, yaoblock
export PauliString, PauliGroupElement, isanticommute

# clifford group
export CliffordGate, clifford_simulate, compile_clifford_circuit

# tensor network
export clifford_network, CliffordNetwork, generate_tensor_network, circuit2tensornetworks

# inference
export syndrome_inference, measure_syndrome!, correction_pauli_string, generate_syndrome_dict, pauli_string_map_iter, inference, transformed_syndrome_dict

# codes 
export stabilizers, ToricCode, SurfaceCode, ShorCode, SteaneCode, Code832, Code422, Code1573, Code513, BivariateBicycleCode, Color488, Color666

# encoder
export CSSBimatrix, syndrome_transform, encode_stabilizers, place_qubits

# measurement
export measure_circuit_fault_tol, measure_circuit_steane, measurement_circuit

# tablemake
export make_table, save_table, load_table, correction_circuit, TruthTable, correction_dict

# simulation
export ComplexConj, SymbolRecorder, IdentityRecorder, ein_circ, QCInfo, qc2enisum
export coherent_error_unitary, error_quantum_circuit, toput, error_pairs, fidelity_tensornetwork, simulation_tensornetwork, error_quantum_circuit_pair_replace

# ldpc
export SimpleTannerGraph, syndrome_extraction, product_graph, CSSTannerGraph, dual_graph, get_graph, belief_propagation, random_ldpc, check_linear_indepent
export osd, check_logical_error

# tableaux
export Tableau, new_tableau, tableau_simulate

# error model
export IndependentFlipError, IndependentDepolarizingError, random_error_pattern, SimpleSyndrome, CSSSyndrome, iid_error, CSSErrorPattern
@const_gate CCZ::ComplexF64 = diagm([1, 1, 1, 1, 1, 1, 1, -1])

# decoder
export BPDecoder, IPDecoder, MatchingDecoder, IPMatchingSolver, TNMAP, TNMMAP, TableDecoder

# decoding
export decode, reduce2general, extract_decoding, DecodingResult, compile, IndependentDepolarizingDecodingProblem, ClassicalDecodingProblem, GeneralDecodingProblem

# threshold
export multi_round_qec

# code distance
export code_distance, logical_operator

# error_learning
export TrainningData, error_learning

# stim parser
export parse_stim_file

include("codes/mod2.jl")
include("clifford/paulistring.jl")
include("clifford/cliffordgroup.jl")
include("clifford/paulibasis.jl")

include("nonclifford/tensornetwork.jl")
include("codes/codes.jl")
include("codes/ldpc.jl")

include("codes/encoder.jl")
include("decoding/error_model.jl")
include("decoding/interfaces.jl")

include("decoding/inferenceswithencoder.jl")
include("decoding/measurement.jl")
include("decoding/truthtable.jl")
include("nonclifford/simulation.jl")

include("clifford/tableaux.jl")
include("decoding/threshold.jl")
include("codes/code_distance.jl")
include("nonclifford/error_learning.jl")
include("multiprocessing.jl")
include("codes/gaussian_elimination.jl")
include("nonclifford/correction.jl")

# decoders
include("decoding/dem.jl")
include("decoding/general_decoding.jl")
include("decoding/bposd.jl")
include("decoding/tndecoder.jl")
include("decoding/ipdecoder.jl")
include("decoding/matching.jl")


# deprecate
include("deprecate.jl")

include("yaoblocks.jl")

include("stim_parser/stim_parser.jl")

end

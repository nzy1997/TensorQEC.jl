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

# === Yao Interop ===
export mat

# === Mod2 Arithmetic ===
export Mod2

# === Pauli Algebra ===
export Pauli, PauliString, PauliGroupElement, @P_str
export pauli_decomposition, pauli_basis, pauli_repr
export isanticommute, yaoblock, SumOfPaulis

# === Clifford Simulation ===
export CliffordGate, clifford_simulate, compile_clifford_circuit
export Tableau, tableau_simulate

# === Codes ===
export SurfaceCode, ToricCode, ShorCode, SteaneCode, Code832, Code422, Code1573, Code513
export BivariateBicycleCode, Color488, Color666
export stabilizers, code_distance, logical_operator

# === Encoding ===
export encode_stabilizers, place_qubits

# === Measurement Circuits ===
export measurement_circuit, measure_circuit_fault_tol

# === Error Models & Syndrome ===
export IndependentFlipError, IndependentDepolarizingError, iid_error
export random_error_pattern, CSSErrorPattern
export SimpleSyndrome, CSSSyndrome, syndrome_extraction
export check_logical_error

# === Tanner Graphs ===
export SimpleTannerGraph, CSSTannerGraph
export product_graph, random_ldpc

# === Decoding Pipeline ===
export DecodingProblem, DecodingResult, decode, compile
export BPDecoder, IPDecoder, MatchingDecoder, TableDecoder, TNMAP, TNMMAP

# === Detector Error Model ===
export DetectorErrorModel, detector_error_model

# === Threshold ===
export multi_round_qec

# === STIM Interop ===
export parse_stim_file

# === Tensor Network Simulation (secondary API) ===
export ComplexConj, qc2einsum, QCInfo
export coherent_error_unitary, fidelity_tensornetwork, simulation_tensornetwork

@const_gate CCZ::ComplexF64 = diagm([1, 1, 1, 1, 1, 1, 1, -1])

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

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
using PrettyTables
using Optimisers

using Graphs
using JuMP
using SCIP

# pauli basis
export pauli_basis, pauli_decomposition, pauli_mapping
import Yao.YaoArrayRegister.StaticArrays: SizedVector
import Yao.YaoArrayRegister.LuxurySparse
export arrayreg2sumofpaulis

# Mod
export Mod2

# tensor network
export clifford_network, CliffordNetwork, generate_tensor_network, circuit2tensornetworks
export ExtraTensor, UNITY4, PXY, PIZ, PXY_UNITY2, PIZ_UNITY2
export densitymatrix2sumofpaulis, SumOfPaulis
export PauliString, PauliGroup, isanticommute,paulistring

# inference
export syndrome_inference, measure_syndrome!,correction_pauli_string, generate_syndrome_dict,pauli_string_map_iter, inference, transformed_syndrome_dict

# codes 
export toric_code, stabilizers,ToricCode, SurfaceCode, ShorCode,SteaneCode,Code832, Code422, Code1573, Code513

# encoder
export CSSBimatrix,syndrome_transform, encode_stabilizers,place_qubits

# measurement
export measure_circuit_fault_tol,  measure_circuit_steane,measure_circuit, measure_circuit_steane_single_type

# tablemake
export make_table, save_table, load_table,correct_circuit,show_table,TruthTable,table_inference

# clifford group
export pauli_group, clifford_group, clifford_simulate,to_perm_matrix,perm_of_paulistring,paulistring_annotate,annotate_history,annotate_circuit_pics,perm_of_pauligroup,generate_group

# simulation
export ComplexConj, SymbolRecorder,IdentityRecorder, ein_circ, QCInfo, qc2enisum
export coherent_error_unitary, error_quantum_circuit,toput, error_pairs,fidelity_tensornetwork, simulation_tensornetwork,error_quantum_circuit_pair_replace

# ldpc
export SimpleTannerGraph,syndrome_extraction,product_graph,CSSTannerGraph,plot_graph,dual_graph,get_graph,belief_propagation,random_ldpc,check_decode, check_linear_indepent,ldpc2tensor
export tensor_infer,osd,mod2matrix_inverse,bp_osd,tensor_osd,check_logical_error

# tableaux
export Tableau, new_tableau,tableau_simulate

# error model
export FlipError, DepolarizingError, random_error_qubits
@const_gate CCZ::ComplexF64 = diagm([1, 1,1,1,1,1,1,-1])

# decoder
export BPOSD,decode,BPDecoder,IPDecoder,reduce2general,extract_decoding,general_syndrome

# threshold
export multi_round_qec,threshold_qec

# code distance
export code_distance

# error_learning
export TrainningData,error_learning

# multiprocessing
export multiprocess_run

include("mod2.jl")
include("paulistring.jl")
include("cliffordgroup.jl")
include("paulibasis.jl")
include("tensornetwork.jl")
include("codes.jl")
include("encoder.jl")
include("inferences.jl")
include("measurement.jl")
include("tablemake.jl")
include("simulation.jl")
include("error_model.jl")
include("ldpc.jl")
include("tableaux.jl")
include("decoder.jl")
include("threshold.jl")
include("code_distance.jl")
include("error_learning.jl")
include("multiprocessing.jl")
end

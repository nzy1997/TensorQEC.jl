@deprecate PauliGroup PauliGroupElement
@deprecate densitymatrix2sumofpaulis(dm) pauli_decomposition(dm)
@deprecate arrayreg2sumofpaulis(t) pauli_decomposition(t)
@deprecate paulistring(nq::Int, idx, pos) PauliString(nq, pos => Pauli(idx-1))
import Yao: apply!
@deprecate apply!(reg::ArrayReg, ps::PauliString) apply!(reg, yaoblock(ps))
@deprecate pauli_mapping(m) reshape(pauli_repr(m), fill(4, 2*log2i(size(m, 1)))...)
@deprecate to_perm_matrix(m::AbstractBlock) to_perm_matrix(pauli_repr(m))

@deprecate perm_of_pauligroup(ps, operation::Pair) operation.second(ps, operation.first)

# v0.3 typo fixes
@deprecate check_linear_indepent(args...) check_linear_independent(args...)
@deprecate qc2enisum(args...) qc2einsum(args...)
const TrainningData = TrainingData

# === v0.3 Removed Exports ===
# The following were removed from public API. Users should:
# - syndrome_inference, measure_syndrome!, correction_pauli_string → Use DecodingProblem + decode pipeline
# - make_table, save_table, load_table, correction_circuit → Use TableDecoder via compile/decode
# - CSSBimatrix, syndrome_transform → Use encode_stabilizers directly
# - reduce2general, extract_decoding → Internal, handled by compile/decode
# - generate_syndrome_dict, pauli_string_map_iter → Internal, use TableDecoder
#
# These functions still exist and can be accessed via:
#   using TensorQEC: function_name
# or
#   TensorQEC.function_name
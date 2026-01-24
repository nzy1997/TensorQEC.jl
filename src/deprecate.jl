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
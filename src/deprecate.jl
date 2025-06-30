@deprecate PauliGroup PauliGroupElement
@deprecate densitymatrix2sumofpaulis(dm) pauli_decomposition(dm)
@deprecate arrayreg2sumofpaulis(t) pauli_decomposition(t)
@deprecate PauliString(t::NTuple{N, Int} where N) PauliString(Pauli.(t .- 1))
@deprecate PauliString(t::Int, ts::Int...) PauliString((Pauli(t-1), Pauli.(ts .- 1)...))
@deprecate paulistring(nq::Int, idx, pos) PauliString(nq, pos => Pauli(idx-1))
import Yao: apply!
@deprecate apply!(reg::ArrayReg, ps::PauliString) apply!(reg, yaoblock(ps))
@deprecate pauli_mapping(m) reshape(pauli_repr(m), fill(4, 2*log2i(size(m, 1)))...)
@deprecate to_perm_matrix(m::AbstractBlock) to_perm_matrix(pauli_repr(m))

@deprecate perm_of_pauligroup(ps, operation::Pair) operation.second(ps, operation.first)
@deprecate PauliGroup PauliGroupElement
@deprecate densitymatrix2sumofpaulis(dm) SumOfPaulis(dm)
@deprecate arrayreg2sumofpaulis(t) SumOfPaulis(t)
@deprecate PauliString(t::NTuple{N, Int} where N) PauliString(Pauli.(t .- 1))
@deprecate PauliString(t::Int, ts::Int...) PauliString((Pauli(t-1), Pauli.(ts .- 1)...))
@deprecate paulistring(nq::Int, idx, pos) PauliString(nq, pos => Pauli(idx-1))
import Yao: apply!
@deprecate apply!(reg::ArrayReg, ps::PauliString) apply!(reg, yaoblock(ps))
@deprecate pauli_mapping(m) reshape(pauli_repr(m), fill(4, 2*log2i(size(m, 1)))...)
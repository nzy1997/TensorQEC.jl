struct CSSToGeneralDecodingProblem <: AbstractReductionResult
    qubit_num::Int
end

single_qubit_tensor(px,py,pz) = [1-px-py-pz pz;px py]
function extract_decoding(cgdp::CSSToGeneralDecodingProblem, error_qubits::Vector{Mod2})
    num_qubits = cgdp.qubit_num
    return DecodingResult(true, CSSErrorPattern(error_qubits[1:num_qubits], error_qubits[(num_qubits+1):2*num_qubits]))
end

# pvec=[px,py,pz], i represent x error, i+num_qubits represent z error
"""
    reduce2general(tanner::CSSTannerGraph, pvec::IndependentDepolarizingError)

Reduce a CSS Tanner graph to a general decoding problem.

### Input:
- `tanner::CSSTannerGraph`: the CSS Tanner graph.
- `pvec::IndependentDepolarizingError`: the error model.
### Output:
- `gdp::GeneralDecodingProblem`: the general decoding problem.
- `csg::CSSToGeneralDecodingProblem`: the reduction result.
"""
function reduce2general(tanner::CSSTannerGraph, pvec::IndependentDepolarizingError)
    num_qubits = nq(tanner)
    tn = SimpleTensorNetwork(DynamicEinCode([[i,i+num_qubits] for i in 1:num_qubits],Int[]),[single_qubit_tensor(pvec.px[j],pvec.py[j],pvec.pz[j]) for j in 1:num_qubits])
    return reduce2general(tanner, tn)
end
function reduce2general(tanner::CSSTannerGraph,tn::SimpleTensorNetwork)
    num_qubits = nq(tanner)
    return GeneralDecodingProblem(SimpleTannerGraph(2*num_qubits, [[x.+ num_qubits for x in tanner.stgx.s2q]..., tanner.stgz.s2q...]),tn), CSSToGeneralDecodingProblem(num_qubits)
end
struct CSSToGeneralDecodingProblem <: AbstractReductionResult
    gdp::GeneralDecodingProblem
    qubit_num::Int
end

single_qubit_tensor(p::Float64) = single_qubit_tensor(p,p,p)
function single_qubit_tensor(px::Float64, py::Float64, pz::Float64)
    return [[1-px-py-pz py;px 0.0];;;[pz 0.0;0.0 0.0]]
end

function extract_decoding(cgdp::CSSToGeneralDecodingProblem, error_qubits::Vector{Mod2})
    num_qubits = cgdp.qubit_num
    return CSSDecodingResult(true, CSSErrorPattern(error_qubits[1:num_qubits] .+  error_qubits[(num_qubits+1):2*num_qubits] , error_qubits[(2*num_qubits+1):3*num_qubits] .+  error_qubits[(num_qubits+1):2*num_qubits]))
end

function reduce2general(tanner::CSSTannerGraph, pvec::Vector{Vector{Float64}})
    num_qubits = nq(tanner)
    tn = TensorNetwork(DynamicEinCode([[i,i+num_qubits,i+2*num_qubits] for i in 1:num_qubits],Int[]),[single_qubit_tensor(pvec[j]...) for j in 1:num_qubits])
    return reduce2general(tanner,tn)
end

function reduce2general(tanner::CSSTannerGraph,tn::TensorNetwork)
    num_qubits = nq(tanner)
    return CSSToGeneralDecodingProblem(GeneralDecodingProblem(SimpleTannerGraph(3*num_qubits,[[vcat(x.+ num_qubits,x .+ 2* num_qubits) for x in tanner.stgx.s2q]..., [vcat(x,x .+  num_qubits) for x in tanner.stgz.s2q]...]),tn),num_qubits)
end
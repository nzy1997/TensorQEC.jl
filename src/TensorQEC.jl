module TensorQEC

using TensorInference
using TensorInference: Factor
using Yao
using Base.Iterators: product
using LinearAlgebra

# pauli basis
export pauli_basis, pauli_decomposition, pauli_mapping

# tensor network
export clifford_network, CliffordNetwork, generate_tensor_network, circuit2tensornetworks

include("paulibasis.jl")
include("tensornetwork.jl")

end

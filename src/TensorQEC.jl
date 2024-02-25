module TensorQEC

using TensorInference
using TensorInference: Factor
using Yao
using Yao.ConstGate: PauliGate
using Base.Iterators: product
using LinearAlgebra

# pauli basis
export pauli_basis, pauli_decomposition, pauli_mapping
import Yao.YaoArrayRegister.StaticArrays: SizedVector

# Mod
export Mod2

# tensor network
export clifford_network, CliffordNetwork, generate_tensor_network, circuit2tensornetworks
export ExtraTensor, UNITY4, PXY, PIZ, PXY_UNITY2, PIZ_UNITY2
export densitymatrix2sumofpaulis, SumOfPaulis
export PauliString, PauliGroup, isanticommute

# inference
export syndrome_inference, measure_syndrome!

# encoder
export Bimatrix, toric_code, stabilizers, syndrome_transform
export ToricCode, SurfaceCode


include("mod2.jl")
include("paulistring.jl")
include("paulibasis.jl")
include("tensornetwork.jl")
include("encoder.jl")
include("inferences.jl")

end

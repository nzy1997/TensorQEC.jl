using Test, TensorQEC, TensorQEC.Yao
using Distributions

function random_pauli_string(N::Int,p::Vector)
    values = [1, 2, 3, 4] # 1: I, 2: X, 3: Y, 4: Z
    return PauliString(([values[rand(Categorical(p[i]))] for i in 1:N]...,)) 
end

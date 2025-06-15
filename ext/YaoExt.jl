module YaoExt

using YaoBlocks

# apply and mat
xgates(ps::PauliString{N}) where N = repeat(N, X, (findall(x->x == 2, (getfield.(ps.operators, :id)...,))...,))
ygates(ps::PauliString{N}) where N = repeat(N, Y, (findall(x->x == 3, (getfield.(ps.operators, :id)...,))...,))
zgates(ps::PauliString{N}) where N = repeat(N, Z, (findall(x->x == 4, (getfield.(ps.operators, :id)...,))...,))

"""
    paulistring(n::Int, k::Int, ids::Vector{Int}) -> PauliString

Create a Pauli string with `n` qubits, where the `i`-th qubit is `k` if `i` is in `ids`, otherwise `1`.
`k` = 1 for I2, 2 for X, 3 for Y, 4 for Z.
"""
paulistring(n::Int, k, ids) = PauliString(0, (i ∈ ids ? k : _I(k) for i in 1:n)...)
_I(::Int) = 1
_I(::YaoBlocks.PauliGate) = I2

function Yao.YaoBlocks.Optimise.to_basictypes(ps::PauliString{N}) where N
    return chain(N, [put(N, i=>pauli(q)) for (i, q) in enumerate(ps.operators) if q != 1]...)
end

pauli(i::Int) = (I2, X, Y, Z)[i]
for (i, GT) in enumerate((:I2Gate, :XGate, :YGate, :ZGate))
    @eval begin
        pauli2idx(p::$GT) = $i
    end
end

end
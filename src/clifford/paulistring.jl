pauli(i::Int) = (I2, X, Y, Z)[i]
for (i, GT) in enumerate((:I2Gate, :XGate, :YGate, :ZGate))
    @eval begin
        pauli2idx(p::$GT) = $i
    end
end

"""
    PauliString{N} <: CompositeBlock{2}

A Pauli string is a tensor product of Pauli gates, e.g. `XYZ`.
The matrix representation of a Pauli string is evaluated as
```math
A = \\bigotimes_{i=1}^N \\sigma_{ids[N-i+1]}
```
where `ids` is the array of integers representing the Pauli gates.
Note the order of `ids` is following the little-endian convention, i.e. the first qubit is the least significant qubit.
For example, the Pauli string `XYZ` has matrix representation `Z ⊗ Y ⊗ X`.

### Fields
- `ids::NTuple{N, Int}`: the array of integers (1-4) representing the Pauli gates.
    - 1: I (\$σ_0\$)
    - 2: X (\$σ_1\$)
    - 3: Y (\$σ_2\$)
    - 4: Z (\$σ_3\$)
"""
struct PauliString{N} <: CompositeBlock{2}
    ids::NTuple{N, Int}
end
PauliString(id1::PauliGate, ids::PauliGate...) = PauliString(pauli2idx.((id1, ids...)))
PauliString(id1::Int, ids::Int...) = PauliString((id1, ids...,))
function Base.:(==)(lhs::PauliString{N}, rhs::PauliString{N}) where N
    return all(lhs.ids .== rhs.ids)
end

Yao.nqudits(::PauliString{N}) where N = N
Yao.occupied_locs(ps::PauliString) = (findall(!=(1), ps.ids)...,)
Yao.cache_key(ps::PauliString) = hash(ps.ids)
Yao.subblocks(ps::PauliString) = pauli.(ps.ids)
Yao.chsubblocks(::PauliString, blocks) = PauliString(blocks...)

# properties (faster implementation)
Yao.ishermitian(::PauliString) = true
Yao.isreflexive(::PauliString) = true
Yao.isunitary(::PauliString) = true

"""
    iscommute(a::PauliString, b::PauliString)

Returns `true` if two Pauli strings commute, i.e. ``a b = b a``.
"""
Yao.iscommute(a::PauliString{N}, b::PauliString{N}) where N = _coeff(a, b) ∈ (0, 2)

"""
    isanticommute(a::PauliString, b::PauliString)

Returns `true` if two Pauli strings anticommute, i.e. ``a b + b a = 0``.
"""
isanticommute(a::PauliString{N}, b::PauliString{N}) where N = _coeff(a, b) ∈ (1, 3)

# the pauli string coefficient of the multiplication of a and b
function _coeff(a::PauliString, b::PauliString)
    c = 0
    for (idx, idy) in zip(a.ids, b.ids)
        coeff, idz = _mul(idx, idy)
        c = _mul_coeff(coeff, c)
    end
    return c
end
# TODO: implment pauli matrix multiplication

# iterating and indexing
Base.lastindex(ps::PauliString) = lastindex(ps.ids)
Base.iterate(ps::PauliString) = iterate(pauli.(ps.ids))
Base.iterate(ps::PauliString, st) = iterate(pauli.(ps.ids), st)
Base.length(ps::PauliString) = length(ps.ids)
Base.eachindex(ps::PauliString) = eachindex(ps.ids)
Base.getindex(ps::PauliString, index::Integer) = pauli(getindex(ps.ids, index))

# visualization
Base.show(io::IO, ::MIME"text/plain", ps::PauliString) = show(io, ps)
function Base.show(io::IO, ps::PauliString)
	for (i, q) in enumerate(ps.ids)
		print(io, "IXYZ"[q])
	end
end
Yao.color(::Type{T}) where {T <: PauliString} = :cyan

# overwrite the print_tree to avoid printing subblocks
function YaoBlocks.print_tree(
    io::IO,
    root::AbstractBlock,
    node::PauliString,
    depth::Int = 1,
    islast::Bool = false,
    active_levels = ();
    maxdepth = 5,
    charset = Yao.YaoBlocks.BlockTreeCharSet(),
    title = true,
    compact = false,
)
    Base.show(io, node)
    Yao.YaoBlocks._println(io, node, islast, compact)
    return nothing
end

# compiling and visualization
function Yao.YaoBlocks.Optimise.to_basictypes(ps::PauliString{N}) where N
    return chain(N, [put(N, i=>pauli(q)) for (i, q) in enumerate(ps.ids) if q != 1]...)
end
# apply and mat
xgates(ps::PauliString{N}) where N = repeat(N, X, (findall(x->x == 2, (ps.ids...,))...,))
ygates(ps::PauliString{N}) where N = repeat(N, Y, (findall(x->x == 3, (ps.ids...,))...,))
zgates(ps::PauliString{N}) where N = repeat(N, Z, (findall(x->x == 4, (ps.ids...,))...,))
function Yao.YaoBlocks.unsafe_apply!(reg::AbstractRegister, ps::PauliString{N}) where N
    for pauligates in (xgates, ygates, zgates)
        blk = pauligates(ps)
        Yao.YaoBlocks.unsafe_apply!(reg, blk)
    end
    return reg
end
function Yao.mat(::Type{T}, ps::PauliString) where T
    return mat(T, xgates(ps)) * mat(T, ygates(ps)) * mat(T, zgates(ps))
end

# create a macro for pauli group items
# - coeff ∈ {0, 1, 2, 3} is the coefficient (im^coeff) of the Pauli string,
# - ps is the Pauli string
struct PauliGroupElement{N} <: CompositeBlock{2}
    coeff::Int
    ps::PauliString{N}
end
Yao.subblocks(x::PauliGroupElement) = (x.ps,)
Yao.nqudits(x::PauliGroupElement{N}) where {N} = N
function Base.:(*)(a::PauliGroupElement{N}, b::PauliGroupElement{N}) where {N}
    cc = _mul_coeff(a.coeff, b.coeff)
    pc = ntuple(N) do i
        cci, pci = _mul(a.ps.ids[i], b.ps.ids[i])
        cc = _mul_coeff(cci, cc)
        pci
    end
    return PauliGroupElement(cc, PauliString(pc))
end
_mul_coeff(a, b) = (a + b) % 4
function _mul(idx::Int, idy::Int)
    if idx == idy
        return 0, 1
    elseif idx == 1
        return 0, idy
    elseif idy == 1
        return 0, idx
    elseif idx == 2 && idy ==3
        return 1, 4
    elseif idx == 3 && idy == 2
        return 3, 4
    elseif idx == 2 && idy == 4
        return 3, 3
    elseif idx == 4 && idy == 2
        return 1, 3
    elseif idx == 3 && idy == 4
        return 1, 2
    else # idx == 4 && idy == 3
        return 3, 2
    end
end
Base.show(io::IO, ::MIME"text/plain", pg::PauliGroupElement) = show(io, pg)
function Base.show(io::IO, pg::PauliGroupElement)
    print(io, ("+1", "+i", "-1", "-i")[pg.coeff+1], " * ")
    print(io, pg.ps)
end
function Yao.print_block(io::IO, ::MIME"text/plain", pg::PauliGroupElement)
    print(io, ("+1", "+i", "-1", "-i")[pg.coeff+1], " * ")
end
Yao.iscommute(a::PauliGroupElement, b::PauliGroupElement) = iscommute(a.ps, b.ps)
isanticommute(a::PauliGroupElement, b::PauliGroupElement) = isanticommute(a.ps, b.ps)
Yao.ishermitian(p::PauliGroupElement) = p.coeff ∈ (0, 2)
Yao.isreflexive(p::PauliGroupElement) = p.coeff ∈ (0, 2)
Yao.isunitary(p::PauliGroupElement) = true

# sum of paulis
"""
    SumOfPaulis{T<:Number, N} <: CompositeBlock{2}

A sum of Pauli strings is a linear combination of Pauli strings, e.g. ``c_1 P_1 + c_2 P_2 + \\cdots + c_n P_n``.

### Fields
- `items::Vector{Pair{T, PauliString{N}}}`: the vector of pairs of coefficients and Pauli strings.
"""
struct SumOfPaulis{T<:Number, N} <: CompositeBlock{2}
	items::Vector{Pair{T, PauliString{N}}}
end
Yao.nqudits(::SumOfPaulis{T,N}) where {T,N} = N
Yao.mat(::Type{T}, sp::SumOfPaulis) where T = mat(T, sum([c * kron(pauli(q) for q in p.ids) for (c, p) in sp.items]))
Yao.subblocks(x::SumOfPaulis) = ((x.second for x in x.items)...,)
Yao.chsubblocks(sp::SumOfPaulis{T,N}, blocks) where {T,N} = SumOfPaulis(Pair{T,PauliString{N}}[c.first => p for (c, p) in zip(sp.items, blocks)])

"""
    SumOfPaulis(m::AbstractArray; atol=0)
    SumOfPaulis(dm::DensityMatrix; atol=0)
    SumOfPaulis(reg::ArrayReg; atol=0)

Returns a pauli decomposition representation of a matrix `m`, a density matrix `dm`, or an array register `reg`.

### Keyword Arguments
- `atol::Float64`: the absolute tolerance for the coefficients. If the coefficient is less than `atol`, it will be ignored.
"""
function SumOfPaulis(m::AbstractMatrix; atol=0)
	coeffs = pauli_decomposition(m)
	return SumOfPaulis([coeffs[ci]=>PauliString(ci.I) for ci in CartesianIndices(coeffs) if !isapprox(coeffs[ci], 0; atol=atol)] |> vec)
end
function SumOfPaulis(dm::DensityMatrix)
    res = SumOfPaulis(dm.state)
    # convert the coefficients to real numbers.
    return SumOfPaulis([real(c)=>p for (c, p) in res.items])
end
SumOfPaulis(reg::ArrayReg) = SumOfPaulis(density_matrix(reg))

"""
    paulistring(n::Int, k::Int, ids::Vector{Int}) -> PauliString

Create a Pauli string with `n` qubits, where the `i`-th qubit is `k` if `i` is in `ids`, otherwise `1`.
`k` = 1 for I2, 2 for X, 3 for Y, 4 for Z.
"""
paulistring(n::Int, k, ids) = PauliString((i ∈ ids ? k : _I(k) for i in 1:n)...)
_I(::Int) = 1
_I(::YaoBlocks.PauliGate) = I2

# macro
macro P_str(str::String)
    indices = Int[]
    for c in str
        if c == 'I'
            push!(indices, 1)
        elseif c == 'X'
            push!(indices, 2)
        elseif c == 'Y'
            push!(indices, 3)
        elseif c == 'Z'
            push!(indices, 4)
        else
            return :(error("Token `$($c)` is not a valid Pauli string."))
        end
    end
    return PauliString(indices...)
end
# we redefine the pauli gates to avoid the Yao.jl dependency
"""
    AbstractPauli{N}

An abstract type for Pauli operators, where `N` is the number of qubits.
"""
abstract type AbstractPauli{N} end

"""
    Pauli <: AbstractPauli{1}

A Pauli operator, i.e. ``I``, ``X``, ``Y``, or ``Z``.

### Fields
- `id::Int`: the id of the Pauli operator, which is 0 for ``I``, 1 for ``X``, 2 for ``Y``, or 3 for ``Z``.
"""
struct Pauli <: AbstractPauli{1}
    id::Int
    function Pauli(id::Int)
        @assert 0 <= id <= 3 "Invalid Pauli operator id: $id (not in range 0-3)"
        return new(id)
    end
end
Base.show(io::IO, ::MIME"text/plain", p::Pauli) = show(io, p)
function Base.show(io::IO, p::Pauli)
    c = p.id == 0 ? 'I' : p.id == 1 ? 'X' : p.id == 2 ? 'Y' : 'Z'
    print(io, c)
end
Base.isless(p1::Pauli, p2::Pauli) = isless(p1.id, p2.id)
coeff_type(::Type{Pauli}) = Int

# YaoAPI
LinearAlgebra.ishermitian(p::Pauli) = true
YaoAPI.isreflexive(p::Pauli) = true
YaoAPI.isunitary(p::Pauli) = true
YaoAPI.mat(p::Pauli) = YaoAPI.mat(ComplexF64, p)
function YaoAPI.mat(::Type{T}, p::Pauli) where T
    id = p.id
    if id == 0
        return PermMatrix([1, 2], ones(T, 2))
    elseif id == 1
        return PermMatrix([2, 1], ones(T, 2))
    elseif id == 2
        return PermMatrix([2, 1], T[-im, im])
    else
        return PermMatrix([1, 2], T[1, -1])
    end
end

"""
    PauliString{N} <: AbstractPauli{N}
    PauliString(operators::NTuple{N, Pauli}) where N
    PauliString(pairs::Pair...)

A Pauli string is a tensor product of Pauli operators, e.g. `XYZ`.
The matrix representation of a Pauli string is evaluated as
```math
A = \\bigotimes_{i=1}^N \\sigma_{operators[N-i+1]}
```
where `operators` is the array of Pauli operators.
Note the order of `operators` is following the little-endian convention, i.e. the first qubit is the least significant qubit.
For example, the Pauli string `XYZ` has matrix representation `Z ⊗ Y ⊗ X`.

### Fields
- `operators::NTuple{N, Pauli}`: the array of Pauli operators.

### Arguments
- `pairs::Pair...`: the pairs of locations and Pauli operators.
"""
struct PauliString{N} <: AbstractPauli{N}
    operators::NTuple{N, Pauli}
    function PauliString(operators::NTuple{N, Pauli}) where N
        return new{N}(operators)
    end
end
function PauliString(operators::Pauli...)
    return PauliString(operators)
end
function PauliString(n::Int, pairs::Pair...)
    alllocs = vcat(collect.(first.(pairs))...)
    @assert allunique(alllocs) "Locations must be unique, but got $alllocs"
    content = ntuple(n) do i
        idx = findfirst(pair -> i ∈ pair.first, pairs)
        if idx === nothing
            return Pauli(0)
        else
            return pairs[idx].second
        end
    end
    return PauliString(content)
end
coeff_type(::Type{PauliString{N}}) where N = Int

function Base.:(==)(lhs::PauliString{N}, rhs::PauliString{N}) where N
    return lhs.operators == rhs.operators
end
function Base.isless(p1::PauliString{N}, p2::PauliString{N}) where N
    for k = N:-1:1
        isless(p1.operators[k], p2.operators[k]) && return true
        isless(p2.operators[k], p1.operators[k]) && return false
    end
    return false
end
# iterating and indexing
Base.lastindex(ps::PauliString) = lastindex(ps.operators)
Base.iterate(ps::PauliString) = iterate(ps.operators)
Base.iterate(ps::PauliString, st) = iterate(ps.operators, st)
Base.length(ps::PauliString) = length(ps.operators)
Base.eachindex(ps::PauliString) = eachindex(ps.operators)
Base.getindex(ps::PauliString, index::Integer) = getindex(ps.operators, index)
Base.keys(ps::PauliString) = Base.OneTo(length(ps))

# visualization
Base.show(io::IO, ::MIME"text/plain", ps::PauliString) = show(io, ps)
function Base.show(io::IO, ps::PauliString)
	for q in ps.operators
		print(io, q)
	end
end

# YaoAPI
function YaoAPI.iscommute(a::PauliString{N}, b::PauliString{N}) where N
    c = 0
    for (idx, idy) in zip(a.operators, b.operators)
        coeff, _ = _mul(idx, idy)
        c = _mul_coeff(coeff, c)
    end
    return c ∈ (0, 2)
end

"""
    isanticommute(a::PauliString, b::PauliString)

Returns `true` if two Pauli strings anticommute, i.e. ``a b + b a = 0``.
"""
isanticommute(a::PauliString{N}, b::PauliString{N}) where N = !iscommute(a, b)

LinearAlgebra.ishermitian(ps::PauliString) = true
YaoAPI.isreflexive(ps::PauliString) = true
YaoAPI.isunitary(ps::PauliString) = true

YaoAPI.mat(ps::PauliString) = YaoAPI.mat(ComplexF64, ps)
function YaoAPI.mat(::Type{T}, ps::PauliString{N}) where {T, N}
    isempty(ps.operators) && return PermMatrix(collect(Int, 1:2^N), fill(T(im^ps.coeff), 2^N))  # identity matrix (with phase)
    return reduce(kron, YaoAPI.mat.(T, ps.operators[end:-1:1]))
end

"""
    PauliGroupElement{N} <: AbstractPauli{N}

A Pauli group element is a Pauli string with a phase factor of `im^k` where `k` is in range 0-3.

### Fields
- `coeff::Int`: the coefficient of the Pauli string, i.e. `im^{coeff}`. It should be in range 0-3.
- `ps::PauliString{N}`: the Pauli string.
"""
struct PauliGroupElement{N} <: AbstractPauli{N}
    coeff::Int
    ps::PauliString{N}
    function PauliGroupElement(coeff::Int, ps::PauliString{N}) where N
        @assert 0 <= coeff <= 3 "Invalid Pauli group element coefficient: $coeff (not in range 0-3)"
        return new{N}(coeff, ps)
    end
end
coeff_type(::Type{PauliGroupElement{N}}) where N = Complex{Int}

# Convert between PauliString and PauliGroupElement
PauliGroupElement(ps::PauliString) = PauliGroupElement(0, ps)

Base.length(pg::PauliGroupElement) = length(pg.ps)

# Algebra operations for PauliGroupElement
function Base.:(*)(a::PauliGroupElement{N}, b::PauliGroupElement{N}) where {N}
    cc = _mul_coeff(a.coeff, b.coeff)
    pc = map(a.ps.operators, b.ps.operators) do x, y
        coeff, idz = _mul(x, y)
        (coeff, idz)
    end
    return PauliGroupElement(mapreduce(x -> x[1], _mul_coeff, pc, init=cc), PauliString(ntuple(i->pc[i][2], Val{N}())))
end
Base.one(::Type{PauliGroupElement{N}}) where N = PauliGroupElement(0, PauliString(ntuple(i->Pauli(0), Val{N}())))

# Visualization
Base.show(io::IO, ::MIME"text/plain", ps::PauliGroupElement) = show(io, ps)
function Base.show(io::IO, ps::PauliGroupElement)
    print(io, ("+1", "+i", "-1", "-i")[ps.coeff+1], " * ", ps.ps)
end

# YaoAPI
LinearAlgebra.ishermitian(ps::PauliGroupElement) = ps.coeff ∈ (0, 2)
YaoAPI.isreflexive(ps::PauliGroupElement) = ps.coeff ∈ (0, 2)
YaoAPI.isunitary(ps::PauliGroupElement) = true

YaoAPI.mat(pg::PauliGroupElement) = YaoAPI.mat(ComplexF64, pg)
function YaoAPI.mat(::Type{T}, pg::PauliGroupElement) where T
    isempty(pg.ps.operators) && return PermMatrix(collect(Int, 1:2^N), fill(T(im^pg.coeff), 2^N))  # identity matrix (with phase)
    return im^pg.coeff * reduce(kron, YaoAPI.mat.(T, pg.ps.operators[end:-1:1]))
end

function YaoAPI.iscommute(a::PauliGroupElement{N}, b::PauliGroupElement{N}) where N
    c = 0
    for (idx, idy) in zip(a.ps.operators, b.ps.operators)
        coeff, _ = _mul(idx, idy)
        c = _mul_coeff(coeff, c)
    end
    return c ∈ (0, 2)
end

"""
    isanticommute(a::PauliGroupElement, b::PauliGroupElement)

Returns `true` if two Pauli group elements anticommute, i.e. ``a b + b a = 0``.
"""
isanticommute(a::PauliGroupElement{N}, b::PauliGroupElement{N}) where N = !iscommute(a, b)

# sum of paulis
"""
    SumOfPaulis{T<:Number, N} <: AbstractPauli{N}

A sum of Pauli strings is a linear combination of Pauli strings, e.g. ``c_1 P_1 + c_2 P_2 + \\cdots + c_n P_n``.

### Fields
- `items::Vector{Pair{T, PauliString{N}}}`: the vector of pairs of coefficients and Pauli strings.
"""
struct SumOfPaulis{T<:Number, N} <: AbstractPauli{N}
	items::Vector{Pair{T, PauliString{N}}}
    function SumOfPaulis(items::Vector{Pair{T, PauliString{N}}}) where {T, N}
        return new{T, N}(items)
    end
end
Base.convert(::Type{SumOfPaulis{T, N}}, p::AbstractPauli{N}) where {T, N} = SumOfPaulis{T, N}(p)
SumOfPaulis{T, N}(p::Pauli) where {T, N} = SumOfPaulis([one(T)=>PauliString(p)])
SumOfPaulis{T, N}(p::PauliString) where {T, N} = SumOfPaulis([one(T)=>p])
SumOfPaulis{T, N}(p::SumOfPaulis{T2, N}) where {T, T2, N} = T === T2 ? p : SumOfPaulis([T(c)=>p for (c, p) in p.items])

"""
    sumofpaulis(items::Vector{Pair{T, PauliString{N}}}) where {T, N}

Returns a sum of Pauli strings from a vector of pairs of coefficients and Pauli strings.
Unlike `SumOfPaulis`, it will merge the same Pauli strings and sum up the coefficients.
"""
function sumofpaulis(items::Vector{Pair{T, PauliString{N}}}) where {T, N}
    res = Dict{PauliString{N}, T}()
    for (c, p) in items
        if haskey(res, p)
            res[p] += c
        else
            res[p] = c
        end
    end
    return SumOfPaulis([c=>p for (p, c) in res])
end

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
	return SumOfPaulis([coeffs[ci]=>PauliString(Pauli.(ci.I .- 1)) for ci in CartesianIndices(coeffs) if !isapprox(coeffs[ci], 0; atol=atol)] |> vec)
end
coeff_type(::Type{SumOfPaulis{T, N}}) where {T, N} = T

Base.show(io::IO, ::MIME"text/plain", sp::SumOfPaulis) = show(io, sp)
function Base.show(io::IO, sp::SumOfPaulis)
    for (i, (c, p)) in enumerate(sp.items)
        print(io, c, " * ", p)
        if i < length(sp.items)
            print(io, " + ")
        end
    end
end
function Base.:(==)(a::SumOfPaulis{T1, N}, b::SumOfPaulis{T2, N}) where {T1, T2, N}
    length(a.items) == length(b.items) || return false
    items1 = sort(a.items, by=x->x.second)
    items2 = sort(b.items, by=x->x.second)
    return all(x.first == y.first && x.second == y.second for (x, y) in zip(items1, items2))
end
function Base.isapprox(a::SumOfPaulis{T1, N}, b::SumOfPaulis{T2, N}; kwargs...) where {T1, T2, N}
    length(a.items) == length(b.items) || return false
    items1 = sort(a.items, by=x->x.second)
    items2 = sort(b.items, by=x->x.second)
    return all(isapprox(x.first, y.first; kwargs...) && x.second == y.second for (x, y) in zip(items1, items2))
end


function Base.:(*)(a::SumOfPaulis{T1, N}, b::SumOfPaulis{T2, N}) where {T1, T2, N}
    return sumofpaulis([((coeff, p) = _mul(p1, p2); im^coeff * c1*c2=>p) for (c1, p1) in a.items for (c2, p2) in b.items])
end
Base.one(::Type{SumOfPaulis{T, N}}) where {T, N} = SumOfPaulis([one(T)=>PauliString(ntuple(i->Pauli(0), Val{N}()))])

function Base.:(+)(a::SumOfPaulis{T1, N}, b::SumOfPaulis{T2, N}) where {T1, T2, N}
    return sumofpaulis(vcat(a.items, b.items))
end
Base.:(-)(a::SumOfPaulis{T, N}) where {T, N} = SumOfPaulis([-c=>p for (c, p) in a.items])
Base.:(-)(a::SumOfPaulis{T1, N}, b::SumOfPaulis{T2, N}) where {T1, T2, N} = a + (-b)
Base.zero(::Type{SumOfPaulis{T, N}}) where {T, N} = SumOfPaulis(Pair{T, PauliString{N}}[])

YaoAPI.mat(sp::SumOfPaulis) = YaoAPI.mat(ComplexF64, sp)
YaoAPI.mat(::Type{T}, sp::SumOfPaulis) where T = sum([c * mat(T, p) for (c, p) in sp.items])

# algebra
Base.promote_rule(::Type{PauliString{1}}, ::Type{Pauli}) = PauliString{1}
Base.promote_rule(::Type{PauliGroupElement{1}}, ::Type{Pauli}) = PauliGroupElement{1}
Base.promote_rule(::Type{PauliGroupElement{N}}, ::Type{PauliString{N}}) where N = PauliGroupElement{N}
Base.promote_rule(::Type{SumOfPaulis{T, N}}, ::Type{Pauli}) where {T, N} = SumOfPaulis{T, N}
Base.promote_rule(::Type{SumOfPaulis{T, N}}, ::Type{PauliString{N}}) where {T, N} = SumOfPaulis{T, N}
Base.promote_rule(::Type{SumOfPaulis{T, N}}, ::Type{PauliGroupElement{N}}) where {T, N} = SumOfPaulis{T, N}

function Base.:(*)(a::Pauli, b::Pauli)
    coeff, idz = _mul(a, b)
    return PauliGroupElement(coeff, PauliString((idz,)))
end
function _mul(a::Pauli, b::Pauli)
    idx, idy = a.id, b.id
    if idx == idy
        return (0, Pauli(0))
    elseif idx == 0
        return (0, Pauli(idy))
    elseif idy == 0
        return (0, Pauli(idx))
    elseif idx == 1 && idy == 2
        return (1, Pauli(3))
    elseif idx == 2 && idy == 1
        return (3, Pauli(3))
    elseif idx == 1 && idy == 3
        return (3, Pauli(2))
    elseif idx == 3 && idy == 1
        return (1, Pauli(2))
    elseif idx == 2 && idy == 3
        return (1, Pauli(1))
    else # idx == 3 && idy == 2
        return (3, Pauli(1))
    end
end
function Base.:(*)(a::PauliString{N}, b::PauliString{N}) where {N}
    coeff, ps = _mul(a, b)
    return PauliGroupElement(coeff, ps)
end
function _mul(a::PauliString{N}, b::PauliString{N}) where {N}
    res = PauliGroupElement(0, a) * PauliGroupElement(0, b)
    return (res.coeff, res.ps)
end

function Base.:(*)(a::AbstractPauli, b::AbstractPauli)
    pa, pb = promote(a, b)
    return pa * pb
end
Base.:(*)(a::Number, b::AbstractPauli{N}) where N = SumOfPaulis([a * c => p for (c, p) in SumOfPaulis{promote_type(typeof(a), coeff_type(typeof(b))), N}(b).items])
Base.:(*)(a::AbstractPauli, b::Number) = b * a
Base.:(/)(a::AbstractPauli, b::Number) = a * inv(b)

function Base.:(+)(a::AbstractPauli{N}, b::AbstractPauli{N}) where {N}
    T = promote_type(coeff_type(typeof(a)), coeff_type(typeof(b)))
    return SumOfPaulis{T, N}(a) + SumOfPaulis{T, N}(b)
end
function Base.:(-)(a::AbstractPauli{N}, b::AbstractPauli{N}) where {N}
    T = promote_type(coeff_type(typeof(a)), coeff_type(typeof(b)))
    return SumOfPaulis{T, N}(a) - SumOfPaulis{T, N}(b)
end
function Base.:(-)(a::AbstractPauli{N}) where {N}
    T = coeff_type(typeof(a))
    return -SumOfPaulis{T, N}(a)
end

# YaoAPI
"""
    yaoblock(x::Pauli)

Returns the Yao block corresponding to a Pauli operator.
"""
yaoblock(x::Pauli) = x.id == 0 ? I2 : x.id == 1 ? X : x.id == 2 ? Y : Z
yaoblock(x::PauliString) = kron(yaoblock.(x.operators)...)
yaoblock(x::PauliGroupElement) = im^x.coeff * yaoblock(x.ps)
yaoblock(x::SumOfPaulis) = sum([c * yaoblock(p) for (c, p) in x.items])
PauliString(gate::YaoBlocks.PauliGate, gates...) = PauliString(Pauli(gate), Pauli.(gates)...)
Pauli(gate::YaoBlocks.PauliGate) = gate == I2 ? Pauli(0) : gate == X ? Pauli(1) : gate == Y ? Pauli(2) : gate == Z ? Pauli(3) : error("Invalid Pauli gate: $gate")

function SumOfPaulis(dm::DensityMatrix)
    res = SumOfPaulis(dm.state)
    # convert the coefficients to real numbers.
    return SumOfPaulis([real(c)=>p for (c, p) in res.items])
end
SumOfPaulis(reg::ArrayReg) = SumOfPaulis(density_matrix(reg))

# used for multiplication of phase factors
_mul_coeff(a::Int, b::Int) = (a + b) % 4

# macro
"""
    @P_str(str::String)

A macro to convert a string to a Pauli string.

### Example
```jldoctest; setup=:(using TensorQEC)
julia> P"IX"
IX
```
"""
macro P_str(str::String)
    paulis = Pauli[]
    for c in str
        if c == 'I'
            push!(paulis, Pauli(0))
        elseif c == 'X'
            push!(paulis, Pauli(1))
        elseif c == 'Y'
            push!(paulis, Pauli(2))
        elseif c == 'Z'
            push!(paulis, Pauli(3))
        else
            return :(error("Token `$($c)` is not a valid Pauli string."))
        end
    end
    return PauliString(paulis...)
end

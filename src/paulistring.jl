pauli(i::Int) = (I2, X, Y, Z)[i]
for (i, GT) in enumerate((:I2Gate, :XGate, :YGate, :ZGate))
    @eval begin
        pauli2idx(p::$GT) = $i
    end
end
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
# Yao.iscommute(a::PauliString, b::PauliString) = a * b == b * a
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
	for (i, q) in enumerate(ps.ids[end:-1:1])
		print(io, "IXYZ"[q])
		if i < length(ps.ids)
			print(io, " ⊗ ")
		end
	end
end
Yao.color(::Type{T}) where {T <: PauliString} = :cyan

# overwrite the print_tree to avoid printing subblocks
function Yao.YaoBlocks.print_tree(
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

# sum of paulis
struct SumOfPaulis{T<:Number, N} <: Yao.CompositeBlock{2}
	items::Vector{Pair{T, PauliString{N}}}
end
Yao.nqudits(::SumOfPaulis{T,N}) where {T,N} = N
Yao.mat(::Type{T}, sp::SumOfPaulis) where T = mat(T, sum([c * kron(pauli(q) for q in p.ids) for (c, p) in sp.items]))
Yao.subblocks(x::SumOfPaulis) = ((x.second for x in x.items)...,)
Yao.chsubblocks(sp::SumOfPaulis{T,N}, blocks) where {T,N} = SumOfPaulis(Pair{T,PauliString{N}}[c.first => p for (c, p) in zip(sp.items, blocks)])

function tensor2sumofpaulis(t::AbstractArray)
	return SumOfPaulis([t[ci]=>PauliString(ci.I) for ci in CartesianIndices(t)] |> vec)
end
densitymatrix2sumofpaulis(dm::DensityMatrix) = tensor2sumofpaulis(real.(pauli_decomposition(dm.state)))
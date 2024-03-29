# generate Clifford group members
# https://www.nature.com/articles/s41534-022-00583-7
clifford_group(n::Int) = generate_group(clifford_generators(n))

function clifford_generators(n::Int)
    @assert n > 0
    if n == 1
        return to_perm_matrix.(Int8, UInt8, pauli_repr.([H, ConstGate.S]))
    else
        return to_perm_matrix.(Int8, UInt8, pauli_repr.(vcat(
            [put(n, i=>H) for i=1:n],
            [put(n, i=>ConstGate.S) for i=1:n],
            [put(n, (i, j)=>ConstGate.CNOT) for j=1:n for i=j+1:n],
            [put(n, (j, i)=>ConstGate.CNOT) for j=1:n for i=j+1:n]
        )))
    end
end
function to_perm_matrix(::Type{T}, ::Type{Ti}, m::AbstractMatrix; atol=1e-8) where {T, Ti}
    @assert all(j -> count(i->abs(i) > atol, view(m, :, j)) == 1, 1:size(m, 2))
    perm = [findfirst(i->abs(i) > atol, view(m, :, j)) for j=1:size(m, 2)]
    vals = [_safe_convert(T, m[perm[j], j]) for j=1:size(m, 2)]
    @assert size(m, 1) <= typemax(Ti)
    return PermMatrix{T, Ti}(perm, vals) |> LuxurySparse.staticize
end
function _safe_convert(::Type{T}, x::Complex) where T
    return _safe_convert(T, real(x)) + _safe_convert(T, imag(x)) * im
end
function _safe_convert(::Type{T}, x::Real) where T
    y = round(T, x)
    @assert x ≈ y "fail to convert target to type: $T"
    return y
end

function generate_group(v::Vector; max_size=Inf)
    # loop until no new elements are added
    keep_loop = true
    items_vector = copy(v)
    items = Dict(zip(items_vector, 1:length(items_vector)))
    while keep_loop && length(items_vector) < max_size
        keep_loop = false
        for m in v, k in 1:length(items_vector)
            candidate = m * items_vector[k]
            if !haskey(items, candidate)
                keep_loop = true
                items[candidate] = k + 1
                push!(items_vector, candidate)
            end
        end
    end
    return items_vector
end

# integer type should fit the size of the matrix
struct CliffordTable{N, Ti}
    basis::Vector{PauliString{N}}
    table::Vector{PermMatrix{Int8, Ti, Vector{Int8}, Vector{Ti}}}
end

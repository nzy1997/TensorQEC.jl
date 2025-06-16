# generate Clifford group members
# https://www.nature.com/articles/s41534-022-00583-7
"""
    clifford_group(n::Int)

Generate the n-qubit Clifford group.
"""
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

"""
    to_perm_matrix([::Type{T}, ::Type{Ti}, ]matrix_or_yaoblock; atol=1e-8)

Convert a Clifford gate to its permutation representation.

### Arguments
- `T`: Element type of phase factor.
- `Ti`: Element type of the permutation matrix.
- `m`: The matrix representation of the gate.
- `atol`: The tolerance to zeros in the matrix.

### Returns
- `pm`: The permutation matrix. pm.perm is the permutation vector, pm.vals is the phase factor.
"""
to_perm_matrix(m::AbstractBlock; atol=1e-8) = to_perm_matrix(Int8, Int, m; atol)
to_perm_matrix(::Type{T}, ::Type{Ti}, m::AbstractBlock; atol=1e-8) where {T, Ti} = to_perm_matrix(T, Ti, pauli_repr(m); atol)
function to_perm_matrix(::Type{T}, ::Type{Ti}, m::AbstractMatrix; atol=1e-8) where {T, Ti}
    @assert all(j -> count(i->abs(i) > atol, view(m, :, j)) == 1, 1:size(m, 2))
    @warn "TODO: fix?"
    perm = [findfirst(i->abs(i) > atol, view(m, :, j)) for j=1:size(m, 2)]
    vals = [_safe_convert(T, m[perm[j], j]) for j=1:size(m, 2)]
    @assert size(m, 1) <= typemax(Ti)
    return PermMatrixCSC{T, Ti}(perm, vals) |> LuxurySparse.staticize
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
    table::Vector{PermMatrixCSC{Int8, Ti, Vector{Int8}, Vector{Ti}}}
end

"""
    perm_of_paulistring(ps::PauliString, operation::Pair{NTuple{M, Int}, <:PermMatrixCSC}) where {M}

Map the Pauli string `ps` by a permutation matrix `pm`. Return the mapped Pauli string and the phase factor.

### Arguments
- `ps`: The Pauli string.
- `operation`: A pair of the positions to apply the permutation and the permutation matrix.

### Returns
- `ps`: The mapped Pauli string.
- `val`: The phase factor.
"""
function perm_of_paulistring(ps::PauliString{N}, operation::Pair{NTuple{M, Int}, <:PermMatrixCSC}) where {N, M}
    pos, pm = operation
    @assert 4^M == length(pm.perm)
    idx = pauli_c2l(Val(M), ntuple(k->ps.operators[pos[k]].id + 1, Val(M)))
    ci = pauli_l2c(Val(M), pm.perm[idx])
    paulis = ntuple(Val{N}()) do k
        loc = findfirst(==(k), operation.first)
        loc === nothing ? ps[k] : Pauli(ci[loc]-1)
    end
    return PauliString(paulis), pm.vals[idx]
end
_complex2int(x) = x==1+0im ? 0 : x==0+1im ? 1 : x==-1+0im ? 2 : 3
function perm_of_pauligroup(pg::PauliGroupElement, operation::Pair{NTuple{M, Int}, <:PermMatrixCSC}) where {M}
    ps, val = perm_of_paulistring(pg.ps, operation)
    return PauliGroupElement(_mul_coeff(pg.coeff,_complex2int(val)), ps)
end

"""
    CliffordSimulateResult{N}

The result of simulating a Pauli string by a Clifford circuit.

### Fields
- `output::PauliString{N}`: A mapped Pauli string as the output.
- `phase::ComplexF64`: The phase factor.
- `circuit::ChainBlock`: The circuit (simplified, with linear structure).
- `history::Vector{PauliString{N}}`: The history of Pauli strings, its length is `length(circuit)+1`.
"""
struct CliffordSimulateResult{N}
    output::PauliString{N}
    phase::Complex{Int}
    circuit::ChainBlock
    history::Vector{PauliString{N}}
    function CliffordSimulateResult(output::PauliString{N}, phase::Complex{Int}, circuit::ChainBlock, history::Vector{PauliString{N}}) where N
        @assert length(history) == length(circuit) + 1
        new{N}(output, phase, circuit, history)
    end
end

"""
    clifford_simulate(ps::PauliString, qc::ChainBlock) 
    
Map the Pauli string `ps` by a quantum circuit `qc`. 

### Arguments
- `ps`: The Pauli string.
- `qc`: The quantum circuit.

### Returns
- `result`: A [`CliffordSimulateResult`](@ref) records the output Pauli string, the phase factor, the simplified circuit, and the history of Pauli strings.
"""
function clifford_simulate(ps::PauliString{N}, qc::ChainBlock) where N
    ps_history = PauliString{N}[]
    qc = simplify(qc; rules=[to_basictypes, Optimise.eliminate_nested])
    gatedict=Dict{UInt64, PermMatrixCSC}()
    valf = 1 + 0im
    push!(ps_history, ps)
    for _gate in qc
        gate = toput(_gate)
        key = hash(gate.content)
        if haskey(gatedict, key) 
            ps, val = perm_of_paulistring(ps, gate.locs=>gatedict[key])
        else 
            pm = to_perm_matrix(Int8, UInt8, pauli_repr(mat(gate.content)))
            push!(gatedict, key => pm)
            ps,val = perm_of_paulistring(ps, gate.locs=>pm)
        end
        valf *= val
        push!(ps_history, ps)
    end
    return CliffordSimulateResult(ps, valf, qc, ps_history)
end

function paulistring_annotate(ps::PauliString{N};color = "red") where N
    return paulistring_annotate(ps, N;color)
end
function paulistring_annotate(ps::PauliString{N},num_qubits::Int;color = "red") where N
    qc = chain(num_qubits)
    for (loc, p) in enumerate(ps.operators)
        p == Pauli(0) && continue
        push!(qc, put(num_qubits, loc=>line_annotation("$p"; color)))
    end
    return qc
end

"""
    annotate_history(res::CliffordSimulateResult{N})

Annotate the history of Pauli strings in the result of `clifford_simulate`.

### Arguments
- `res`: The result of `clifford_simulate`.

### Returns
- `draw`: The visualization of the history.
"""
function annotate_history(res::CliffordSimulateResult{N}) where N
    qcf,_= generate_annotate_circuit(res)
    return annotate_circuit(qcf)
end

function generate_annotate_circuit(res::CliffordSimulateResult{N};color = "red") where N
    qcf = chain(N)
    pos = [1]
    push!(qcf, paulistring_annotate(res.history[1];color))
    for i in 1:length(res.circuit)
        block = res.circuit[i]
        push!(qcf, block)
        if !isempty(occupied_locs(block) ∩ findall(x -> x != Pauli(0), res.history[i+1]))
            push!(qcf, paulistring_annotate(res.history[i+1];color))
            push!(pos, length(qcf))
        end
    end
    return qcf,pos
end

function annotate_circuit(qcf::ChainBlock;filename = nothing)
    CircuitStyles.barrier_for_chain[], temp = true, CircuitStyles.barrier_for_chain[]
    draw = vizcircuit(qcf;filename)
    CircuitStyles.barrier_for_chain[] = temp
    return draw
end

function replace_block_color(qc::ChainBlock,color::String)
    for i in 1:length(qc)
        qc[i] = replace_block_color(qc[i],color)
    end
    return qc
end

replace_block_color(qc::PutBlock,color::String) = put(qc.n,qc.locs=> line_annotation(qc.content.name; color))

function annotate_circuit_pics(res::CliffordSimulateResult{N};foldername=nothing) where N
    qcf,pos = generate_annotate_circuit(res;color = "transparent")
    filename = nothing
    (foldername === nothing) || (filename = "$foldername/0.png")
    annotate_circuit(qcf;filename)
    for i in 1:length(pos)
        qcf[pos[i]] = replace_block_color(qcf[pos[i]],"red")
        (foldername === nothing) || (filename = "$foldername/$i.png")
        annotate_circuit(qcf;filename)
    end
end
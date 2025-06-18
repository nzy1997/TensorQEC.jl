# generate Clifford group members
# https://www.nature.com/articles/s41534-022-00583-7

"""
    CliffordGate{PM<:PermMatrixCSC{Int, Int}}

CliffordGate represented as a permutation matrix.
It is a callable object that can be applied to a [`PauliString`](@ref) or a [`PauliGroupElement`](@ref).

### Fields
- `mat::PM`: The permutation matrix.

### Examples
```jldoctest
julia> using TensorQEC.Yao

julia> hgate = CliffordGate(H)
CliffordGate(nqubits = 1)
 I → I
 X → Z
 Y → -Y
 Z → X

julia> hgate(P"IX", (2,))
+1 * IZ

julia> hgate(PauliGroupElement(1, P"IX"), (2,))
+i * IZ
```
"""
struct CliffordGate{PM<:PermMatrixCSC{Complex{Int}, Int}}
    mat::PM
end
# TODO: improve performance
CliffordGate(g::AbstractBlock) = CliffordGate(to_perm_matrix(pauli_repr(g)))

Yao.mat(c::CliffordGate) = c.mat
Base.:(*)(c1::CliffordGate, c2::CliffordGate) = CliffordGate(c1.mat * c2.mat)
YaoAPI.nqubits(c::CliffordGate) = log2i(size(c.mat, 1)) ÷ 2

Base.show(io::IO, ::MIME"text/plain", c::CliffordGate) = show(io, c)
function Base.show(io::IO, c::CliffordGate)
    n = nqubits(c)
    basis = pauli_basis(Val(n))
    println(io, "CliffordGate(nqubits = $n)")
    for (j, b) in enumerate(basis)
        bi = basis[c.mat.perm[j]]
        coeff = c.mat.vals[j]
        print(io, " $(b) → ")
        coeff == 1 ? print(io, bi) : coeff == -1 ? print(io, "-$(bi)") : real(coeff) ≈ 0 ? print(io, imag(coeff), "im * $(bi)") : imag(coeff) ≈ 0 ? print(io, real(coeff), " * $(bi)") : print(io, coeff, " * $(bi)")
        j < length(basis) && println(io)
    end
end

"""
    clifford_group(n::Int)

Generate the n-qubit Clifford group.
"""
clifford_group(n::Int) = generate_group(clifford_generators(n))

function clifford_generators(n::Int)
    @assert n > 0
    if n == 1
        return CliffordGate.([H, ConstGate.S])
    else
        return CliffordGate.(vcat(
            [put(n, i=>H) for i=1:n],
            [put(n, i=>ConstGate.S) for i=1:n],
            [put(n, (i, j)=>ConstGate.CNOT) for j=1:n for i=j+1:n],
            [put(n, (j, i)=>ConstGate.CNOT) for j=1:n for i=j+1:n]
        ))
    end
end

"""
    to_perm_matrix(matrix; atol=1e-8)

Convert a general matrix to a permutation matrix.

### Arguments
- `m`: The matrix representation of the gate.
- `atol`: The tolerance to zeros in the matrix.

### Returns
- `pm`: The permutation matrix. pm.perm is the permutation vector, pm.vals is the leading coefficient.
"""
function to_perm_matrix(m::AbstractMatrix; atol=1e-8)
    @assert all(j -> count(i->abs(i) > atol, view(m, :, j)) == 1, 1:size(m, 2))
    perm = [findfirst(i->abs(i) > atol, view(m, :, j)) for j=1:size(m, 2)]
    vals = [_safe_convert(Complex{Int}, m[perm[j], j]) for j=1:size(m, 2)]
    return PermMatrixCSC(perm, vals) |> LuxurySparse.staticize
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
    (gate::CliffordGate)(ps::PauliString, pos::NTuple{M, Int}) where {M}

Map the Pauli string `ps` by a Clifford gate `gate`. Return the mapped Pauli group element.

### Arguments
- `ps`: The Pauli string.
- `pos`: The positions to apply the permutation.

### Returns
- `pg`: The mapped Pauli group element.
"""
function (gate::CliffordGate)(ps::PauliString{N}, pos::NTuple{M, Int}) where {N, M}
    pm = mat(gate)
    @assert 4^M == length(pm.perm)
    idx = pauli_c2l(Val(M), ntuple(k->ps.operators[pos[k]].id + 1, Val(M)))
    ci = pauli_l2c(Val(M), pm.perm[idx])
    paulis = ntuple(Val{N}()) do k
        loc = findfirst(==(k), pos)
        loc === nothing ? ps[k] : Pauli(ci[loc]-1)
    end
    return PauliGroupElement(_complex2int(pm.vals[idx]), PauliString(paulis))
end
_complex2int(x) = x==1+0im ? 0 : x==0+1im ? 1 : x==-1+0im ? 2 : 3
function (gate::CliffordGate)(pg::PauliGroupElement, pos::NTuple{M, Int}) where {M}
    elem = gate(pg.ps, pos)
    return PauliGroupElement(_add_phase(pg.phase, elem.phase), elem.ps)
end

"""
    CliffordSimulateResult{N}

The result of simulating a Pauli string by a Clifford circuit.

### Fields
- `output::PauliGroupElement{N}`: A mapped Pauli group element as the output.
- `circuit::ChainBlock`: The circuit (simplified, with linear structure).
- `history::Vector{PauliGroupElement{N}}`: The history of Pauli group elements, its length is `length(circuit)+1`.
"""
struct CliffordSimulateResult{N}
    output::PauliGroupElement{N}
    circuit::ChainBlock
    history::Vector{PauliGroupElement{N}}
    function CliffordSimulateResult(output::PauliGroupElement{N}, circuit::ChainBlock, history::Vector{PauliGroupElement{N}}) where N
        @assert length(history) == length(circuit) + 1
        new{N}(output, circuit, history)
    end
end

struct CompiledCliffordCircuit{M1, M2}
    sequence::Vector{Tuple{Int, Int, Int}}  # (howmany qubits, gate-idx, locs-idx)
    single_qubit_gates::Vector{CliffordGate{M1}}
    single_qubit_locs::Vector{Tuple{Int}}
    two_qubit_gates::Vector{CliffordGate{M2}}
    two_qubit_locs::Vector{Tuple{Int, Int}}
end

function compile_clifford_circuit(qc::ChainBlock)
    sequence = Tuple{Int, Int, Int}[]
    single_qubit_gates = typeof(CliffordGate(X))[]
    single_qubit_locs = Tuple{Int}[]
    two_qubit_gates = typeof(CliffordGate(ConstGate.CNOT))[]
    two_qubit_locs = Tuple{Int, Int}[]

    qc = simplify(qc; rules=[to_basictypes, Optimise.eliminate_nested])
    gatedict = Dict{UInt64, Int}()
    for _gate in qc
        gate = toput(_gate)
        key = hash(gate.content)
        if haskey(gatedict, key) 
            cgate_idx = gatedict[key]
        else 
            if nqubits(gate.content) == 1
                push!(single_qubit_gates, CliffordGate(gate.content))
                cgate_idx = length(single_qubit_gates)
            else
                push!(two_qubit_gates, CliffordGate(gate.content))
                cgate_idx = length(two_qubit_gates)
            end
            gatedict[key] = cgate_idx
        end
        if nqubits(gate.content) == 1
            push!(single_qubit_locs, gate.locs)
            push!(sequence, (1, cgate_idx, length(single_qubit_locs)))
        else
            push!(two_qubit_locs, gate.locs)
            push!(sequence, (2, cgate_idx, length(two_qubit_locs)))
        end
    end
    return CompiledCliffordCircuit(sequence, single_qubit_gates, single_qubit_locs, two_qubit_gates, two_qubit_locs)
end
function (cl::CompiledCliffordCircuit)(pg::PauliGroupElement{N}) where {N}
    for i in 1:length(cl.sequence)
        pg = _step(cl, pg, i)
    end
    return pg
end
function _step(cl::CompiledCliffordCircuit, pg::PauliGroupElement{N}, i::Int) where {N}
    howmany, gate_idx, locs_idx = cl.sequence[i]
    if howmany == 1
        return cl.single_qubit_gates[gate_idx](pg, cl.single_qubit_locs[locs_idx])
    else
        return cl.two_qubit_gates[gate_idx](pg, cl.two_qubit_locs[locs_idx])
    end
end

"""
    clifford_simulate(paulistring, qc::ChainBlock)
    
Perform Clifford simulation with a Pauli string `paulistring` as the input.

### Arguments
- `paulistring`: The input Pauli string, which can be a [`PauliString`](@ref) or a [`PauliGroupElement`](@ref).
- `qc`: The quantum circuit represented as a Yao's `ChainBlock`, its elements should be Clifford gates.

### Returns
- `result`: A [`CliffordSimulateResult`](@ref) records the output Pauli string, the phase factor, the simplified circuit, and the history of Pauli strings.
"""
clifford_simulate(ps::PauliString{N}, qc::ChainBlock) where {N} = clifford_simulate(PauliGroupElement(0, ps), qc)
function clifford_simulate(pg::PauliGroupElement{N}, qc::ChainBlock) where {N}
    history = PauliGroupElement{N}[]
    cl = compile_clifford_circuit(qc)
    push!(history, pg)
    for i in 1:length(cl.sequence)
        pg = _step(cl, pg, i)
        push!(history, pg)
    end
    # TODO: remove the quantum circuit from the result
    return CliffordSimulateResult(pg, qc, history)
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
    push!(qcf, paulistring_annotate(res.history[1].ps;color))
    for i in 1:length(res.circuit)
        block = res.circuit[i]
        push!(qcf, block)
        if !isempty(occupied_locs(block) ∩ findall(x -> x != Pauli(0), res.history[i+1].ps))
            push!(qcf, paulistring_annotate(res.history[i+1].ps;color))
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
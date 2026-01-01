struct TToricDecoder <: AbstractGeneralDecoder 
    row_transformation::Matrix{Mod2}
    column_transformation::Matrix{Mod2}
    x_product_pos::Vector{Tuple{Int,Int}}
    z_product_pos::Vector{Tuple{Int,Int}}
    toric_qubit_pos::Vector{Vector{Int}}
    toric_x_check_pos::Vector{Vector{Int}}
    toric_z_check_pos::Vector{Vector{Int}}
    cg_size::Int
end

Base.show(io::IO, ::MIME"text/plain", p::TToricDecoder) = show(io, p)
Base.show(io::IO, p::TToricDecoder) = print(io, "TToricDecoder(Matching)")

struct PackedMod2Matrix
    data::Matrix{UInt64}
    nrows::Int
    ncols::Int
end

function pack_mod2_rows(A::AbstractMatrix{Mod2})
    nrows, ncols = size(A)
    chunks = cld(ncols, 64)
    data = zeros(UInt64, nrows, chunks)
    @inbounds for i in 1:nrows
        for chunk in 1:chunks
            word = UInt64(0)
            base = (chunk - 1) * 64
            maxj = min(64, ncols - base)
            for k in 1:maxj
                if A[i, base + k].x
                    word |= UInt64(1) << (k - 1)
                end
            end
            data[i, chunk] = word
        end
    end
    return PackedMod2Matrix(data, nrows, ncols)
end

function pack_mod2_vec!(buf::Vector{UInt64}, x::AbstractVector{Mod2})
    n = length(x)
    chunks = cld(n, 64)
    if length(buf) < chunks
        resize!(buf, chunks)
    end
    @inbounds for chunk in 1:chunks
        word = UInt64(0)
        base = (chunk - 1) * 64
        maxj = min(64, n - base)
        for k in 1:maxj
            if x[base + k].x
                word |= UInt64(1) << (k - 1)
            end
        end
        buf[chunk] = word
    end
    return buf
end

function mul_packed!(y::Vector{Mod2}, A::PackedMod2Matrix, x_bits::Vector{UInt64})
    chunks = size(A.data, 2)
    @inbounds for i in 1:A.nrows
        count = 0
        for chunk in 1:chunks
            count += count_ones(A.data[i, chunk] & x_bits[chunk])
        end
        y[i] = Mod2(isodd(count))
    end
    return y
end

struct CompiledTTD <: CompiledDecoder
    ttd::TToricDecoder
    qubit_num::Int
    x_product_pos::Vector{NTuple{2,Int}}
    z_product_pos::Vector{NTuple{2,Int}}
    toric_qubit_pos::Vector{Vector{Int}}
    toric_x_check_rows::Vector{Vector{Int}}
    toric_z_check_rows::Vector{Vector{Int}}
    matching_graphx::Vector{MatchingGraph}
    edge2qubitx::Vector{Matrix{Int}}
    matching_graphz::Vector{MatchingGraph}
    edge2qubitz::Vector{Matrix{Int}}
    row_packed::PackedMod2Matrix
    col_packed::PackedMod2Matrix
    s_buf::Vector{Mod2}
    synd_buf::Vector{Mod2}
    ep_buf::Vector{Mod2}
    ct_buf::Vector{Mod2}
    out_buf::Vector{Mod2}
    s_bits_buf::Vector{UInt64}
    ct_bits_buf::Vector{UInt64}
    xevents_buf::Vector{Int}
    zevents_buf::Vector{Int}
end

function compile(decoder::TToricDecoder, iddp::IndependentDepolarizingDecodingProblem)
    qubit_num = iddp.tanner.stgx.nq
    syd_num = qubit_num ÷ 2
    @assert syd_num % decoder.cg_size == 0 "syndrome block size mismatch"
    blocks_per_half = syd_num ÷ decoder.cg_size
    check_block(pos::Int) = (1 <= pos <= 2 * blocks_per_half) ? pos : error("invalid check position: $pos")
    x_qubit_block(pos::Int) = (isodd(pos) && (pos + 1) ÷ 2 <= blocks_per_half) ? (pos + 1) ÷ 2 : error("invalid x qubit position: $pos")
    z_qubit_block(pos::Int) = (iseven(pos) && pos ÷ 2 <= blocks_per_half) ? pos ÷ 2 : error("invalid z qubit position: $pos")
    expand_rows(pos::Vector{Int}) = begin
        rows = Int[]
        for p in pos
            block = check_block(p)
            append!(rows, (block - 1) * decoder.cg_size + 1:block * decoder.cg_size)
        end
        return rows
    end
    x_product_pos = Vector{NTuple{2,Int}}(undef, length(decoder.x_product_pos))
    for (i, (check_pos, qubit_pos)) in enumerate(decoder.x_product_pos)
        x_product_pos[i] = (check_block(check_pos), x_qubit_block(qubit_pos))
    end
    z_product_pos = Vector{NTuple{2,Int}}(undef, length(decoder.z_product_pos))
    for (i, (check_pos, qubit_pos)) in enumerate(decoder.z_product_pos)
        z_product_pos[i] = (check_block(check_pos), z_qubit_block(qubit_pos))
    end
    toric_num = length(decoder.toric_qubit_pos)
    @assert toric_num == length(decoder.toric_x_check_pos) "toric_qubit_pos and toric_x_check_pos must have the same length"
    @assert toric_num == length(decoder.toric_z_check_pos) "toric_qubit_pos and toric_z_check_pos must have the same length"
    toric_qubit_pos = Vector{Vector{Int}}(undef, toric_num)
    toric_x_check_rows = Vector{Vector{Int}}(undef, toric_num)
    toric_z_check_rows = Vector{Vector{Int}}(undef, toric_num)
    for i in 1:toric_num
        toric_qubit_pos[i] = expand_rows(Int.(decoder.toric_qubit_pos[i]))
        toric_x_check_rows[i] = expand_rows(Int.(decoder.toric_x_check_pos[i]))
        toric_z_check_rows[i] = expand_rows(Int.(decoder.toric_z_check_pos[i]))
    end
    for pos in toric_qubit_pos
        for q in pos
            @assert 1 <= q <= qubit_num "invalid toric qubit position: $q"
        end
    end
    H = [iddp.tanner.stgx.H zeros(Mod2,syd_num,qubit_num); zeros(Mod2,syd_num,qubit_num) iddp.tanner.stgz.H];
    H2 = decoder.row_transformation* H*decoder.column_transformation
    matching_graphx = Vector{MatchingGraph}(undef, toric_num)
    edge2qubitx = Vector{Matrix{Int}}(undef, toric_num)
    matching_graphz = Vector{MatchingGraph}(undef, toric_num)
    edge2qubitz = Vector{Matrix{Int}}(undef, toric_num)
    for i in 1:toric_num
        x_rows = toric_x_check_rows[i]
        x_cols = toric_qubit_pos[i]
        matching_graphx[i], edge2qubitx[i] = parity_matrix2matching_graph(Matrix(@view H2[x_rows, x_cols]))
        z_rows = toric_z_check_rows[i]
        z_cols = [q + qubit_num for q in toric_qubit_pos[i]]
        matching_graphz[i], edge2qubitz[i] = parity_matrix2matching_graph(Matrix(@view H2[z_rows, z_cols]))
    end
    row_packed = pack_mod2_rows(decoder.row_transformation)
    col_packed = pack_mod2_rows(decoder.column_transformation)
    s_buf = Vector{Mod2}(undef, size(decoder.row_transformation, 2))
    synd_buf = Vector{Mod2}(undef, size(decoder.row_transformation, 1))
    ep_buf = Vector{Mod2}(undef, 2 * qubit_num)
    ct_buf = Vector{Mod2}(undef, 2 * qubit_num)
    out_buf = Vector{Mod2}(undef, 2 * qubit_num)
    s_bits_buf = Vector{UInt64}(undef, cld(size(decoder.row_transformation, 2), 64))
    ct_bits_buf = Vector{UInt64}(undef, cld(size(decoder.column_transformation, 2), 64))
    xevents_buf = Int[]
    zevents_buf = Int[]
    sizehint!(xevents_buf, decoder.cg_size)
    sizehint!(zevents_buf, decoder.cg_size)
    return CompiledTTD(
        decoder,
        qubit_num,
        x_product_pos,
        z_product_pos,
        toric_qubit_pos,
        toric_x_check_rows,
        toric_z_check_rows,
        matching_graphx,
        edge2qubitx,
        matching_graphz,
        edge2qubitz,
        row_packed,
        col_packed,
        s_buf,
        synd_buf,
        ep_buf,
        ct_buf,
        out_buf,
        s_bits_buf,
        ct_bits_buf,
        xevents_buf,
        zevents_buf,
    )
end


function decode(ci::CompiledTTD, syndrome::CSSSyndrome)
    num_qubits = ci.qubit_num
    cg_size = ci.ttd.cg_size
    s = ci.s_buf
    copyto!(s, 1, syndrome.sx, 1, length(syndrome.sx))
    copyto!(s, length(syndrome.sx) + 1, syndrome.sz, 1, length(syndrome.sz))
    synd = ci.synd_buf
    pack_mod2_vec!(ci.s_bits_buf, s)
    mul_packed!(synd, ci.row_packed, ci.s_bits_buf)
    # @show synd
    ep = ci.ep_buf
    fill!(ep, zero(Mod2))
    @inbounds for (check_block, qubit_block) in ci.x_product_pos
        row_start = (check_block - 1) * cg_size + 1
        col_start = (qubit_block - 1) * 2 * cg_size + 1
        for j in 1:cg_size
            ep[num_qubits + col_start + j - 1] = synd[row_start + j - 1]
        end
    end
    @inbounds for (check_block, qubit_block) in ci.z_product_pos
        row_start = (check_block - 1) * cg_size + 1
        col_start = (qubit_block - 1) * 2 * cg_size + 1
        for j in 1:cg_size
            ep[col_start + cg_size + j - 1] = synd[row_start + j - 1]
        end
    end
    # @show ep

    xevents = ci.xevents_buf
    zevents = ci.zevents_buf
    for i in 1:length(ci.toric_qubit_pos)
        fill_events_mod2_indices!(xevents, synd, ci.toric_x_check_rows[i])
        error_vecx = decode_to_edges(ci.matching_graphx[i], xevents)
        @inbounds for j in 1:length(error_vecx) ÷ 2
            col = ci.toric_qubit_pos[i][ci.edge2qubitx[i][error_vecx[2 * j - 1], error_vecx[2 * j]]]
            ep[
                num_qubits +
                col
            ] = Mod2(1)
        end

        fill_events_mod2_indices!(zevents, synd, ci.toric_z_check_rows[i])
        error_vecz = decode_to_edges(ci.matching_graphz[i], zevents)
        @inbounds for j in 1:length(error_vecz) ÷ 2
            col = ci.toric_qubit_pos[i][ci.edge2qubitz[i][error_vecz[2 * j - 1], error_vecz[2 * j]]]
            ep[
                col
            ] = Mod2(1)
        end
        # ep[2*ci.ttd.cg_size*ci.ttd.product_num+1+2*ci.ttd.cg_size*(i-1):2*ci.ttd.cg_size*ci.ttd.product_num+2*ci.ttd.cg_size*i] .= res.error_pattern.xerror
        # ep[num_qubits+2*ci.ttd.cg_size*ci.ttd.product_num+1+2*ci.ttd.cg_size*(i-1):num_qubits+2*ci.ttd.cg_size*ci.ttd.product_num+2*ci.ttd.cg_size*i] .= res.error_pattern.zerror
    end

    # @show [ep[num_qubits+1:end]...,ep[1:num_qubits]...]
    ct = ci.ct_buf
    @inbounds for i in 1:num_qubits
        ct[i] = ep[num_qubits + i]
        ct[num_qubits + i] = ep[i]
    end
    out = ci.out_buf
    pack_mod2_vec!(ci.ct_bits_buf, ct)
    mul_packed!(out, ci.col_packed, ci.ct_bits_buf)
    xerror = copy(@view out[num_qubits+1:end])
    zerror = copy(@view out[1:num_qubits])
    return DecodingResult(true, CSSErrorPattern(xerror, zerror))
end

function parity_matrix2matching_graph(H::Matrix{Mod2})
    n = max(size(H, 1), size(H, 2))
    if isodd(n)
        n += 1
    end
    m = MatchingGraph(n, 0)
    edge2qubit = zeros(Int, size(H, 1),size(H,1))
    for i in 1:size(H, 2)
        a,b = findall(v->v.x,H[:,i])
        SparseBlossom.add_edge!(m, a, b)
        edge2qubit[a,b] = i
        edge2qubit[b,a] = i
    end
    return m, edge2qubit
end

function fill_events!(buf::Vector{Int}, syndrome::AbstractVector{Bool})
    empty!(buf)
    for i in eachindex(syndrome)
        if syndrome[i]
            push!(buf, i)  # 1-based
        end
    end
    return buf
end

function fill_events_mod2!(buf::Vector{Int}, synd::AbstractVector{Mod2}, start::Int, len::Int)
    if length(buf) < len
        resize!(buf, len)
    end
    count = 0
    @inbounds for j in 1:len
        if synd[start + j - 1].x
            count += 1
            buf[count] = j  # 1-based in the slice
        end
    end
    resize!(buf, count)
    return buf
end

function fill_events_mod2_indices!(buf::Vector{Int}, synd::AbstractVector{Mod2}, indices::AbstractVector{Int})
    if length(buf) < length(indices)
        resize!(buf, length(indices))
    end
    count = 0
    @inbounds for j in 1:length(indices)
        if synd[indices[j]].x
            count += 1
            buf[count] = j
        end
    end
    resize!(buf, count)
    return buf
end

function _mod2_matrix_to_ints(mat::AbstractMatrix{Mod2})
    rows, cols = size(mat)
    out = Vector{Vector{Int}}(undef, rows)
    @inbounds for i in 1:rows
        row = Vector{Int}(undef, cols)
        for j in 1:cols
            row[j] = mat[i, j].x ? 1 : 0
        end
        out[i] = row
    end
    return out
end

function _mod2_matrix_from_json(arr, label::String)
    rows = length(arr)
    cols = rows == 0 ? 0 : length(arr[1])
    mat = Matrix{Mod2}(undef, rows, cols)
    @inbounds for i in 1:rows
        row = arr[i]
        @assert length(row) == cols "$label is ragged"
        for j in 1:cols
            mat[i, j] = Mod2(row[j] != 0)
        end
    end
    return mat
end

function save_ttoric_decoder(filename::AbstractString, decoder::TToricDecoder)
    data = Dict{String, Any}(
        "format" => "TToricDecoder",
        "cg_size" => decoder.cg_size,
        "x_product_pos" => [[a, b] for (a, b) in decoder.x_product_pos],
        "z_product_pos" => [[a, b] for (a, b) in decoder.z_product_pos],
        "toric_qubit_pos" => [Int.(pos) for pos in decoder.toric_qubit_pos],
        "toric_x_check_pos" => [Int.(pos) for pos in decoder.toric_x_check_pos],
        "toric_z_check_pos" => [Int.(pos) for pos in decoder.toric_z_check_pos],
        "row_transformation" => _mod2_matrix_to_ints(decoder.row_transformation),
        "column_transformation" => _mod2_matrix_to_ints(decoder.column_transformation),
    )
    open(filename, "w") do io
        JSON.print(io, data, 2)
    end
    return nothing
end

function load_ttoric_decoder(filename::AbstractString;switch_xz_part = false)
    data = JSON.parsefile(filename)
    format = get(data, "format", "TToricDecoder")
    @assert format == "TToricDecoder" "unsupported format: $format"
    row = _mod2_matrix_from_json(data["row_transformation"], "row_transformation")
    col = _mod2_matrix_from_json(data["column_transformation"], "column_transformation")
    x_product_pos = [(Int(pos[1]), Int(pos[2])) for pos in data["x_product_pos"]]
    z_product_pos = [(Int(pos[1]), Int(pos[2])) for pos in data["z_product_pos"]]
    toric_qubit_pos = [Int.(pos) for pos in data["toric_qubit_pos"]]
    toric_x_check_pos = [Int.(pos) for pos in data["toric_x_check_pos"]]
    toric_z_check_pos = [Int.(pos) for pos in data["toric_z_check_pos"]]
    cg_size = Int(data["cg_size"])
    if switch_xz_part
        x_product_pos, z_product_pos = z_product_pos, x_product_pos
        toric_x_check_pos, toric_z_check_pos = toric_z_check_pos, toric_x_check_pos
    end
    return TToricDecoder(
        row,
        col,
        x_product_pos,
        z_product_pos,
        toric_qubit_pos,
        toric_x_check_pos,
        toric_z_check_pos,
        cg_size,
    )
end

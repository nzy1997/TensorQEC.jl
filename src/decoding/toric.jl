struct TToricDecoder <: AbstractGeneralDecoder 
    row_transformation::Matrix{Mod2}
    column_transformation::Matrix{Mod2}
    product_num::Int
    toric_num::Int
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
    matching_graphx::MatchingGraph
    edge2qubitx::Matrix{Int}
    matching_graphz::MatchingGraph
    edge2qubitz::Matrix{Int}
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
    @assert syd_num == (decoder.toric_num + decoder.product_num )* decoder.cg_size "a"
    # @show syd_num
    # @show decoder.toric_num
    # @show decoder.product_num
    # @show decoder.cg_size
    # @show qubit_num
    # @show size(iddp.tanner.stgx.H)
    H = [iddp.tanner.stgx.H zeros(Mod2,syd_num,qubit_num); zeros(Mod2,syd_num,qubit_num) iddp.tanner.stgz.H];
    H2 = decoder.row_transformation* H*decoder.column_transformation
    matching_graphx, edge2qubitx = parity_matrix2matching_graph(H2[decoder.product_num*decoder.cg_size+1:decoder.product_num*decoder.cg_size+decoder.cg_size,2*decoder.product_num*decoder.cg_size+1:2*decoder.product_num*decoder.cg_size+2*decoder.cg_size])
    matching_graphz, edge2qubitz = parity_matrix2matching_graph(H2[syd_num+decoder.product_num*decoder.cg_size+1:syd_num+decoder.product_num*decoder.cg_size+decoder.cg_size,qubit_num+2*decoder.product_num*decoder.cg_size+1:2*decoder.product_num*decoder.cg_size+2*decoder.cg_size+qubit_num])
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
    product_len = ci.ttd.cg_size * ci.ttd.product_num
    s = ci.s_buf
    copyto!(s, 1, syndrome.sx, 1, length(syndrome.sx))
    copyto!(s, length(syndrome.sx) + 1, syndrome.sz, 1, length(syndrome.sz))
    synd = ci.synd_buf
    pack_mod2_vec!(ci.s_bits_buf, s)
    mul_packed!(synd, ci.row_packed, ci.s_bits_buf)
    # @show synd
    xstab_num = length(synd) ÷ 2
    ep = ci.ep_buf
    fill!(ep, zero(Mod2))
    @inbounds for i in 1:product_len
        ep[i + num_qubits] = synd[i]
        ep[i + product_len] = synd[i + xstab_num]
    end
    # @show ep

    xevents = ci.xevents_buf
    zevents = ci.zevents_buf
    for i in 1:ci.ttd.toric_num
        fill_events_mod2!(
            xevents,
            synd,
            product_len + 1 + ci.ttd.cg_size * (i - 1),
            ci.ttd.cg_size,
        )
        error_vecx = decode_to_edges(ci.matching_graphx, xevents)
        @inbounds for j in 1:length(error_vecx) ÷ 2
            ep[
                num_qubits +
                2 * product_len +
                2 * ci.ttd.cg_size * (i - 1) +
                ci.edge2qubitx[error_vecx[2 * j - 1], error_vecx[2 * j]]
            ] = Mod2(1)
        end

        fill_events_mod2!(
            zevents,
            synd,
            xstab_num + product_len + 1 + ci.ttd.cg_size * (i - 1),
            ci.ttd.cg_size,
        )
        error_vecz = decode_to_edges(ci.matching_graphz, zevents)
        @inbounds for j in 1:length(error_vecz) ÷ 2
            ep[
                2 * product_len +
                2 * ci.ttd.cg_size * (i - 1) +
                ci.edge2qubitz[error_vecz[2 * j - 1], error_vecz[2 * j]]
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
    m = MatchingGraph(size(H, 2), 0)
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

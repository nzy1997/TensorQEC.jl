struct TToricDecoder <: AbstractGeneralDecoder 
    row_transformation::Matrix{Mod2}
    column_transformation::Matrix{Mod2}
    product_num::Int
    toric_num::Int
    cg_size::Int
end

Base.show(io::IO, ::MIME"text/plain", p::TToricDecoder) = show(io, p)
Base.show(io::IO, p::TToricDecoder) = print(io, "TToricDecoder(Matching)")

struct CompiledTTD <: CompiledDecoder
    ttd::TToricDecoder
    qubit_num::Int
    matching_graphx::MatchingGraph
    edge2qubitx::Matrix{Int}
    matching_graphz::MatchingGraph
    edge2qubitz::Matrix{Int}
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
    return CompiledTTD(decoder, qubit_num, matching_graphx, edge2qubitx, matching_graphz, edge2qubitz)
end


function decode(ci::CompiledTTD, syndrome::CSSSyndrome)
    num_qubits = ci.qubit_num
    s = [syndrome.sx...,syndrome.sz...]
    synd = ci.ttd.row_transformation * s
    # @show synd
    xstab_num = length(synd) ÷ 2
    ep = zeros(Mod2,2*num_qubits)
    for i in 1:ci.ttd.cg_size*ci.ttd.product_num
        ep[i+num_qubits] = synd[i]
        ep[i+ci.ttd.cg_size*ci.ttd.product_num] = synd[i+xstab_num]
    end
    # @show ep

    for i in 1:ci.ttd.toric_num
        xsyndrome = findall(v -> v.x,synd[ci.ttd.cg_size*ci.ttd.product_num+1+ci.ttd.cg_size*(i-1):ci.ttd.cg_size*ci.ttd.product_num+ci.ttd.cg_size*i])
        error_vecx = decode_to_edges(ci.matching_graphx, xsyndrome)
        for j in 1:length(error_vecx) ÷ 2
            ep[num_qubits+2*ci.ttd.cg_size*ci.ttd.product_num+2*ci.ttd.cg_size*(i-1)+ci.edge2qubitx[error_vecx[2*j-1],error_vecx[2*j]]] = Mod2(1)
        end

        zsyndrome = findall(v -> v.x,synd[xstab_num+ci.ttd.cg_size*ci.ttd.product_num+1+ci.ttd.cg_size*(i-1):xstab_num+ci.ttd.cg_size*ci.ttd.product_num+ci.ttd.cg_size*i])
        error_vecz = decode_to_edges(ci.matching_graphz, zsyndrome)
        for j in 1:length(error_vecz) ÷ 2
            ep[2*ci.ttd.cg_size*ci.ttd.product_num+2*ci.ttd.cg_size*(i-1)+ci.edge2qubitz[error_vecz[2*j-1],error_vecz[2*j]]] = Mod2(1)
        end
        # ep[2*ci.ttd.cg_size*ci.ttd.product_num+1+2*ci.ttd.cg_size*(i-1):2*ci.ttd.cg_size*ci.ttd.product_num+2*ci.ttd.cg_size*i] .= res.error_pattern.xerror
        # ep[num_qubits+2*ci.ttd.cg_size*ci.ttd.product_num+1+2*ci.ttd.cg_size*(i-1):num_qubits+2*ci.ttd.cg_size*ci.ttd.product_num+2*ci.ttd.cg_size*i] .= res.error_pattern.zerror
    end

    # @show [ep[num_qubits+1:end]...,ep[1:num_qubits]...]
    error_pattern = ci.ttd.column_transformation * [ep[num_qubits+1:end]...,ep[1:num_qubits]...]
    return DecodingResult(true, CSSErrorPattern(error_pattern[num_qubits+1:end], error_pattern[1:num_qubits]))
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

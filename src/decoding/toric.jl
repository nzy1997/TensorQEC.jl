struct TToricDecoder{DT} <: AbstractGeneralDecoder 
    toric_decoder::DT
    row_transformation::Matrix{Mod2}
    column_transformation::Matrix{Mod2}
    product_num::Int
    toric_num::Int
    cg_size::Int
end

Base.show(io::IO, ::MIME"text/plain", p::TToricDecoder) = show(io, p)
Base.show(io::IO, p::TToricDecoder) = print(io, "TToricDecoder($(p.toric_decoder))")

struct CompiledTTD{DT} <: CompiledDecoder
    ttd::TToricDecoder{DT}
    tanner::SimpleTannerGraph
    toric_tanner::CSSTannerGraph
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
    stgx = SimpleTannerGraph(H2[decoder.product_num*decoder.cg_size+1:decoder.product_num*decoder.cg_size+decoder.cg_size,2*decoder.product_num*decoder.cg_size+1:2*decoder.product_num*decoder.cg_size+2*decoder.cg_size])
    stgz = SimpleTannerGraph(H2[syd_num+decoder.product_num*decoder.cg_size+1:syd_num+decoder.product_num*decoder.cg_size+decoder.cg_size,qubit_num+2*decoder.product_num*decoder.cg_size+1:2*decoder.product_num*decoder.cg_size+2*decoder.cg_size+qubit_num])
    toric_tanner = CSSTannerGraph(stgx, stgz)
    return CompiledTTD(decoder, SimpleTannerGraph(H),toric_tanner)
end


function decode(ci::CompiledTTD, syndrome::CSSSyndrome)
    num_qubits = ci.tanner.nq ÷ 2
    s = [syndrome.sx...,syndrome.sz...]
    synd = ci.ttd.row_transformation * s
    # @show synd
    xstab_num = length(synd) ÷ 2
    ep = zeros(Mod2,ci.tanner.nq)
    for i in 1:ci.ttd.cg_size*ci.ttd.product_num
        ep[i+num_qubits] = synd[i]
        ep[i+ci.ttd.cg_size*ci.ttd.product_num] = synd[i+xstab_num]
    end
    # @show ep

    for i in 1:ci.ttd.toric_num
        res = decode(ci.ttd.toric_decoder, ci.toric_tanner,CSSSyndrome(synd[ci.ttd.cg_size*ci.ttd.product_num+1+ci.ttd.cg_size*(i-1):ci.ttd.cg_size*ci.ttd.product_num+ci.ttd.cg_size*i],synd[xstab_num+ci.ttd.cg_size*ci.ttd.product_num+1+ci.ttd.cg_size*(i-1):xstab_num+ci.ttd.cg_size*ci.ttd.product_num+ci.ttd.cg_size*i]))
        ep[2*ci.ttd.cg_size*ci.ttd.product_num+1+2*ci.ttd.cg_size*(i-1):2*ci.ttd.cg_size*ci.ttd.product_num+2*ci.ttd.cg_size*i] .= res.error_pattern.xerror
        ep[num_qubits+2*ci.ttd.cg_size*ci.ttd.product_num+1+2*ci.ttd.cg_size*(i-1):num_qubits+2*ci.ttd.cg_size*ci.ttd.product_num+2*ci.ttd.cg_size*i] .= res.error_pattern.zerror
    end

    # @show [ep[num_qubits+1:end]...,ep[1:num_qubits]...]
    error_pattern = ci.ttd.column_transformation * [ep[num_qubits+1:end]...,ep[1:num_qubits]...]
    return DecodingResult(true, CSSErrorPattern(error_pattern[num_qubits+1:end], error_pattern[1:num_qubits]))
end


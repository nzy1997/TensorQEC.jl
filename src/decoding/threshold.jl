function multi_round_qec(tanner::CSSTannerGraph,decoder::AbstractDecoder,em::AbstractQuantumErrorModel;rounds = 10)
    logical_xerror = 0
    logical_zerror = 0
    logical_error = 0
    ct = compile(decoder, tanner)
    for i in 1:rounds
        ep = random_error_qubits(nq(tanner), em)
        synd = syndrome_extraction(ep, tanner)
        res = decode(ct, synd)
        resx = check_logical_error(ep.xerror, res.error_qubits.xerror, tanner.stgx.H)
        resz = check_logical_error(ep.zerror, res.error_qubits.zerror, tanner.stgz.H)
        resx || (logical_xerror += 1)
        resz || (logical_zerror += 1)
        (resx && resz) || (logical_error += 1)
        @info "round $i, logical x error: $logical_xerror, logical z error: $logical_zerror, logical error: $(logical_error)"
        @info "error model: $(em)"
    end
    return logical_xerror/rounds,logical_zerror/rounds,logical_error/rounds
end

function multi_round_qec(tanner::SimpleTannerGraph,decoder::AbstractDecoder,em::AbstractClassicalErrorModel,tanner_check::SimpleTannerGraph;rounds = 10)
    logical_xerror = 0
    ct = compile(decoder, tanner)

    for i in 1:rounds
        ex = random_error_qubits(nq(tanner), em)
        sydz = syndrome_extraction(ex, tanner)
        res = decode(ct, sydz)
        check_logical_error(ex, res.error_qubits, tanner_check.H) || (logical_xerror += 1)
        @info "round $i, error model: $(em), logical error number: $logical_xerror"
    end
    return logical_xerror/rounds
end
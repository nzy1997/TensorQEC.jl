struct SimulationResult
    rounds::Int
    logical_xerror_rate::Float64
    logical_zerror_rate::Float64
    logical_error_rate::Float64
end

function multi_round_qec(tanner::CSSTannerGraph,decoder,em::AbstractErrorModel;rounds = 100)
    logical_xerror = 0
    logical_zerror = 0
    logical_error = 0
    for i in 1:rounds
        resx,resz = single_round_qec(tanner,decoder,em)
        resx || (logical_xerror += 1)
        resz || (logical_zerror += 1)
        (resx && resz) || (logical_error += 1)
        @info "round $i, logical x error: $logical_xerror, logical z error: $logical_zerror, logical error: $(logical_error)"
        @info "error model: $(em)"
    end
    return SimulationResult(rounds,logical_xerror/rounds,logical_zerror/rounds,logical_error/rounds)
end

function single_round_qec(tanner::CSSTannerGraph, deocder::AbstractDecoder,em::AbstractQuantumErrorModel)
    ex,ez = random_error_qubits(nq(tanner), em)

    sydz = sydrome_extraction(ex, tanner.stgz)
    res = decode(deocder, tanner.stgz, em.px, sydz)
    ex_app = res.error_qubits

    sydx = sydrome_extraction(ez, tanner.stgx)
    res = decode(deocder, tanner.stgx, em.pz, sydx)
    ez_app = res.error_qubits

    return  check_logical_error(ex, ex_app, tanner.stgx.H), check_logical_error(ez, ez_app, tanner.stgz.H)
end

function threshold_qec(tanner::CSSTannerGraph,decoder,error_model_vec::Vector;rounds = 100)
    return [multi_round_qec(tanner,decoder, em; rounds) for em in error_model_vec]
end


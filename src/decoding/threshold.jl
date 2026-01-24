"""
    multi_round_qec(tanner::CSSTannerGraph, decoder::AbstractDecoder, em::AbstractQuantumErrorModel; rounds=10)
    multi_round_qec(tanner::SimpleTannerGraph, decoder::AbstractDecoder, em::AbstractClassicalErrorModel, tanner_check::SimpleTannerGraph; rounds=10)

Run multiple rounds of quantum error correction and estimate the logical error rate.

Generates random errors according to the error model, extracts syndromes, decodes,
and checks for logical errors over the specified number of rounds.

# Arguments
- `tanner`: The Tanner graph of the code.
- `decoder::AbstractDecoder`: The decoder to use.
- `em::AbstractErrorModel`: The error model.
- `rounds::Int = 10`: Number of QEC rounds to simulate.

# Returns
- For CSS codes: `(x_error_rate, z_error_rate, total_error_rate)` as a tuple.
- For classical codes: `error_rate` as a scalar.
"""
function multi_round_qec(tanner::CSSTannerGraph,decoder::AbstractDecoder,em::AbstractQuantumErrorModel;rounds = 10)
    logical_xerror = 0
    logical_zerror = 0
    logical_error = 0
    ct = compile(decoder, tanner)
    for i in 1:rounds
        ep = random_error_pattern(em)
        synd = syndrome_extraction(ep, tanner)
        res = decode(ct, synd)
        resx = check_logical_error(ep.xerror, res.error_pattern.xerror, tanner.stgx.H)
        resz = check_logical_error(ep.zerror, res.error_pattern.zerror, tanner.stgz.H)
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
        ex = random_error_pattern(em)
        sydz = syndrome_extraction(ex, tanner)
        res = decode(ct, sydz)
        check_logical_error(ex, res.error_pattern, tanner_check.H) || (logical_xerror += 1)
        @info "round $i, error model: $(em), logical error number: $logical_xerror"
    end
    return logical_xerror/rounds
end
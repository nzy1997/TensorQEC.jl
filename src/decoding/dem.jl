struct DetectorErrorModel
    error_rates::Vector{Float64}
    flipped_detectors::Vector{Vector{Int}}
    detector_list::Vector{Int}
    logical_list::Vector{Int}
end

# visualization
Base.show(io::IO, ::MIME"text/plain", dem::DetectorErrorModel) = show(io, dem)
function Base.show(io::IO, dem::DetectorErrorModel)
    fd = Vector{Vector{Int}}()
    fl = Vector{Vector{Int}}()
    for i in 1:length(dem.error_rates)
        push!(fd, dem.flipped_detectors[i] ∩ dem.detector_list)
        push!(fl, dem.flipped_detectors[i] ∩ dem.logical_list)
    end
	header = ["Error Index","Probability", "Detectors", "Logicals"]
	pt = pretty_table(io, hcat(1:length(dem.error_rates), dem.error_rates, fd, fl); header)
	return nothing
end

function detector_error_model(qc::ChainBlock)
    qc = simplify(qc; rules=[to_basictypes, Optimise.eliminate_nested])
    cqc = compile_clifford_circuit(qc)
    @assert length(cqc.sequence) == length(qc)
    num_qubits = nqubits(qc)
    dl = Vector{Int}()
    ll = Vector{Int}()
    detector_error_map = Dict{Vector{Int}, Float64}()
    for (i, gate) in enumerate(qc)
        if gate isa PutBlock && gate.content isa MixedUnitaryChannel
            for (j, op) in enumerate(gate.content.operators)
                if op == I2
                elseif op == X
                    ps = PauliString(num_qubits,gate.locs[1]=>Pauli(1))
                    push_to_dict!(detector_error_map, forward_analysis(ps, cqc, i, qc), gate.content.probs[j])
                elseif op == Z
                    ps = PauliString(num_qubits,gate.locs[1]=>Pauli(3))
                    push_to_dict!(detector_error_map, forward_analysis(ps, cqc, i, qc), gate.content.probs[j])
                end
            end
        elseif gate isa PutBlock && gate.content isa DepolarizingChannel
            @assert gate.content.n == 1 "Depolarizing channel is not supported for multiple qubits"
            @assert gate.content.p <= 0.75 "Can't analyze single-qubit over-mixing depolarizing errors (probability > 3/4)"
            ps = PauliString(num_qubits,gate.locs[1]=>Pauli(1))
            xdetectors = forward_analysis(ps, cqc, i, qc)

            ps = PauliString(num_qubits,gate.locs[1]=>Pauli(2))
            ydetectors = forward_analysis(ps, cqc, i, qc)

            ps = PauliString(num_qubits,gate.locs[1]=>Pauli(3))
            zdetectors = forward_analysis(ps, cqc, i, qc)

            if isempty(xdetectors)
                @assert ydetectors == zdetectors "X and Z detectors are not consistent"
                push_to_dict!(detector_error_map, ydetectors, gate.content.p*2/3)
            elseif isempty(ydetectors)
                @assert xdetectors == zdetectors "X and Z detectors are not consistent"
                push_to_dict!(detector_error_map, xdetectors, gate.content.p*2/3)
            elseif isempty(zdetectors)
                @assert xdetectors == ydetectors "X and Y detectors are not consistent"
                push_to_dict!(detector_error_map, xdetectors, gate.content.p*2/3)
            else
                push_to_dict!(detector_error_map, xdetectors, (1-sqrt(1-4*gate.content.p/3))/2)
                push_to_dict!(detector_error_map, ydetectors, (1-sqrt(1-4*gate.content.p/3))/2)
                push_to_dict!(detector_error_map, zdetectors, (1-sqrt(1-4*gate.content.p/3))/2)
            end
        elseif gate isa PutBlock && gate.content isa DetectorBlock && gate.content.detector_type == 0 
            push!(dl, gate.content.num)
        elseif gate isa PutBlock && gate.content isa DetectorBlock && gate.content.detector_type == 1
            push!(ll, gate.content.num)
        end
    end
    ers = Float64[]
    fds = Vector{Int}[]
    
    for (detectors, error_rate) in detector_error_map
        push!(ers, error_rate)
        push!(fds, detectors)
    end
    return DetectorErrorModel(ers,fds,dl,ll)
end

function forward_analysis(ps::PauliString, cqc::CompiledCliffordCircuit, i::Int, qc::ChainBlock)
    pg = PauliGroupElement(0, ps)
    flipped_measure = Int[]
    flipped_detectors = Int[]
    for j in i+1:length(qc)
        if qc[j] isa Measure
            if qc[j].postprocess isa ResetTo
                ps_vec = collect(pg.ps.operators)
                for pos in qc[j].locations
                    ps_vec[pos] = Pauli(0)
                end
                pg = PauliGroupElement(0, PauliString(ps_vec...))
            end
        elseif qc[j].content isa NumberedMeasure
            if qc[j].content.m.operator == ComputationalBasis()
                p = Pauli(3)
            elseif  qc[j].content.m.operator == X
                p = Pauli(1)
            elseif qc[j].content.m.operator == Y
                p = Pauli(2)
            end
            pgm = PauliGroupElement(0, PauliString(nqubits(qc), qc[j].locs=>p))

            if isanticommute(pg, pgm)
                push!(flipped_measure, qc[j].content.num)
            end

            if qc[j].content.m.postprocess isa ResetTo
                ps_vec = collect(pg.ps.operators)
                ps_vec[qc[j].locs[1]] = Pauli(0)
                pg = PauliGroupElement(0, PauliString(ps_vec...))
            end
        elseif qc[j].content isa DetectorBlock
            count_m = 0
            for k in qc[j].content.vm
                if k.num in flipped_measure
                    count_m += 1
                end
            end
            if count_m % 2 == 1
                push!(flipped_detectors, qc[j].content.num)
            end
        else
            pg = _step(cqc, pg, j)
        end
    end
    return flipped_detectors
end

function push_to_dict!(detector_error_map::Dict{Vector{Int}, Float64}, detectors::Vector{Int}, error_rate::Float64)
    if isempty(detectors)
        return detector_error_map
    end
    sorted_detectors = sort(detectors)
    if haskey(detector_error_map, sorted_detectors)
        detector_error_map[sorted_detectors] = (1-detector_error_map[sorted_detectors]) * error_rate + detector_error_map[sorted_detectors] * (1- error_rate)
    else
        detector_error_map[sorted_detectors] = error_rate
    end
    return detector_error_map
end

# TODO:
# 1. Compile measurement and detectors

function dem2tanner(dem::DetectorErrorModel)
    nq = length(dem.error_rates)
    q2s = [setdiff(v,dem.logical_list) for v in dem.flipped_detectors]
    ns = maximum(dem.detector_list)
    s2q = [findall(x-> i ∈ x , q2s) for i in 1:ns]
    H = zeros(Mod2, ns, nq)
    for i in 1:ns, j in s2q[i]
        H[i, j] = 1
    end
    return SimpleTannerGraph(nq, ns, q2s, s2q, H)
end

function random_error_qubits(dem::DetectorErrorModel)
    return random_error_qubits(IndependentFlipError(dem.error_rates))
end
struct DetectorErrorModel
    error_rates::Vector{Float64}
    flipped_detectors::Vector{Vector{Int}}
end

function detector_error_model(qc::ChainBlock)
    qc = simplify(qc; rules=[to_basictypes, Optimise.eliminate_nested])
    cqc = compile_clifford_circuit(qc)
    @assert length(cqc.sequence) == length(qc)
    num_qubits = nqubits(qc)
    ers = Vector{Float64}()
    fds = Vector{Vector{Int}}()
    for (i, gate) in enumerate(qc)
        if gate isa PutBlock && gate.content isa MixedUnitaryChannel
            for (j, op) in enumerate(gate.content.operators)
                if op == I2
                elseif op == X
                    ps = PauliString(num_qubits,gate.locs[1]=>Pauli(1))
                    push!(fds,forward_analysis(ps, cqc, i, qc))
                    push!(ers, gate.content.probs[j])
                elseif op == Z
                    ps = PauliString(num_qubits,gate.locs[1]=>Pauli(3))
                    forward_analysis(ps, cqc, i, qc)
                end
            end
        elseif gate isa PutBlock && gate.content isa DepolarizingChannel
            @assert gate.content.n == 1 "Depolarizing channel is not supported for multiple qubits"
            ps = PauliString(num_qubits,gate.locs[1]=>Pauli(1))
            push!(fds,forward_analysis(ps, cqc, i, qc))
            push!(ers, gate.content.p*2/3)

            ps = PauliString(num_qubits,gate.locs[1]=>Pauli(3))
            push!(fds,forward_analysis(ps, cqc, i, qc))
            push!(ers, gate.content.p*2/3)
        end
    end
    return DetectorErrorModel(ers,fds)
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

function postprocess(ers,fds)

end

# TODO:
# 1. Compile measurement and detectors
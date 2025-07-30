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
        end
    end
    return DetectorErrorModel(ers,fds)
end

function forward_analysis(ps::PauliString, cqc::CompiledCliffordCircuit, i::Int, qc::ChainBlock)
    pg = PauliGroupElement(0, ps)
    flipped_measure = Int[]
    flipped_detectors = Int[]
    for j in i+1:length(qc)
        if qc[j] isa NumberedMeasure
            if qc[j].m.operator == ComputationalBasis()
                p = Pauli(3)
            elseif  qc[j].m.operator == X
                p = Pauli(1)
            elseif qc[j].m.operator == Y
                p = Pauli(2)
            end
            pgm = PauliGroupElement(0, PauliString(nqubits(qc), qc[j].m.locations=>p))

            if isanticommute(pg, pgm)
                push!(flipped_measure, qc[j].num)
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

# TODO:
# 1. Compile measurement and detectors
# 2. measurement block to put
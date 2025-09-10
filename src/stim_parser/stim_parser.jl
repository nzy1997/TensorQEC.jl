"""
    parse_stim_file(file_path::String, qubit_number::Int)

Parse a Stim file and return a Yao circuit.

# Arguments
- `file_path::String`: The path to the Stim file.
- `qubit_number::Int`: The number of qubits in the circuit.
"""
function parse_stim_file(file_path::String, qubit_number::Int)
    content = read(file_path, String)
    return parse_stim_string(content, qubit_number)
end
function parse_stim_string(content::String, qubit_number::Int)
    measure_list = Vector{NumberedMeasure}()
    measure_pos_list = Vector{Int}()
    detector_num = [0]
    return _parse_stim_string!(content, qubit_number, measure_list, measure_pos_list, detector_num)
end

function _parse_stim_string!(content::String, qubit_number::Int, measure_list::Vector{NumberedMeasure}, measure_pos_list::Vector{Int}, detector_num::Vector{Int})
    lines = split(content, '\n')
    qc = chain(qubit_number)
    
    i = 1
    while i <= length(lines)
        line = strip(lines[i])
        # @show line
        # @show i
        if startswith(line, "#") || isempty(line)
            i += 1
            continue
        end
        
        # Remove inline comments (everything after #)
        if occursin("#", line)
            comment_start = findfirst("#", line)
            line = strip(line[1:(comment_start.start-1)])
        end
        
        # Parse the instruction
        parts = split(line)
        if isempty(parts)
            i += 1
            continue
        end
        
        instruction_name = uppercase(parts[1])
        
        # Check if this is a REPEAT block
        if instruction_name == "REPEAT"
            # Parse REPEAT block
            repeat_count = parse(Int, parts[2])
            
            # Check if the opening brace is on the same line
            if endswith(line, "{")
                # Brace is on the same line, start collecting from next line
                block_content = String[]
                i += 1  # Move to next line
                brace_count = 1
            else
                # Brace should be on the next line
                block_content = String[]
                i += 1  # Move to next line
                brace_count = 0
            end
            
            while i <= length(lines)
                block_line = strip(lines[i])
                if startswith(block_line, "#") || isempty(block_line)
                    i += 1
                    continue
                end
                
                if block_line == "{" || endswith(block_line, "{")
                    brace_count += 1
                elseif block_line == "}"
                    brace_count -= 1
                    if brace_count == 0
                        break
                    end
                end
               
                if brace_count > 0
                    push!(block_content, block_line)
                end
                
                i += 1
            end
            # Convert block content back to string for recursive parsing
            block_str = join(block_content, "\n")
            
            # block_circuit = _parse_stim_string!(block_str, qubit_number, measure_list, measure_pos_list, detector_num)
            for _ in 1:repeat_count
                push!(qc, _parse_stim_string!(block_str, qubit_number, measure_list, measure_pos_list, detector_num))
            end
            
            i += 1
            continue
        end
        
        # Extract targets (skip instruction name)
        targets = parts[2:end]
        
        # Parse arguments if present
        arguments = Float64[]
        target_start = 2
        
        # Check if there are arguments in parentheses
        if '(' in parts[1]
           start_pos = findfirst('(', parts[1])
           end_pos = findlast(')', parts[1])
           instruction_name = uppercase(parts[1][1:start_pos-1])
           if !isnothing(end_pos)
                temp_string = parts[1][start_pos+1:end_pos-1]
                target_start = 2
           else
                temp_string = parts[1][start_pos+1:end]
                target_start = 2
                while true
                    if ')' in parts[target_start]
                        temp_string *= parts[target_start][1:end-1]
                        target_start += 1
                        break
                    else
                        temp_string *= parts[target_start]
                        target_start += 1
                    end
                end
           end
           arguments = [parse(Float64, strip(arg)) for arg in split(temp_string, ",") if !isempty(strip(arg))]
        end
     
        targets = parts[target_start:end]
        
        # Clean up targets that might have trailing parentheses
        cleaned_targets = String[]
        for target in targets
            if isempty(target)
                continue
            end
            # Remove any trailing parentheses that might have been left over
            if endswith(target, ")")
                target = target[1:end-1]
            end
            if !isempty(target)
                push!(cleaned_targets, target)
            end
        end
        
        # Parse targets and apply gates
        qubit_indices = Int[]
        record_idx = Int[]
        for target in cleaned_targets
            if isempty(target)
                continue
            end
            
            # Handle different target types
            if target == "*"
                # Combiner - skip for now
                continue
            elseif startswith(target, "rec[") && endswith(target, "]")
                record_content = target[5:end-1]  # Remove "rec[" and "]"
                push!(record_idx, parse(Int, record_content))
            elseif startswith(target, "sweep[") && endswith(target, "]")
                # Sweep bit target - skip for now
                continue
            elseif startswith(target, "!")
                # Inverted target
                try
                    qubit_idx = parse(Int, target[2:end])
                    push!(qubit_indices, qubit_idx)
                catch
                    error("Invalid inverted target: $target")
                end
            elseif length(target) >= 2 && target[1] in ['X', 'Y', 'Z']
                # Pauli target
                try
                    qubit_idx = parse(Int, target[2:end])
                    push!(qubit_indices, qubit_idx)
                catch
                    error("Invalid Pauli target: $target")
                end
            else
                # Regular qubit target
                try
                    qubit_idx = parse(Int, target)
                    push!(qubit_indices, qubit_idx)
                catch
                    error("Invalid qubit target: $target")
                end
            end
        end
        
        # Apply the appropriate gate based on instruction name
        apply_gate!(qc, qubit_number, instruction_name, qubit_indices, arguments, measure_list, record_idx, measure_pos_list, detector_num)
        
        i += 1
    end
    
    return qc
end

# Helper function to apply gates
function apply_gate!(qc, qubit_number::Int, instruction_name::String, qubit_indices::Vector{Int}, arguments::Vector{Float64}, measure_list::Vector{NumberedMeasure}, record_idx::Vector{Int}, measure_pos_list::Vector{Int},detector_num::Vector{Int})
    if instruction_name == "H"
        for qubit in qubit_indices
            push!(qc, put(qubit_number,qubit+1 => H))
        end
    elseif instruction_name == "X"
        for qubit in qubit_indices
            push!(qc, put(qubit_number, qubit+1 => X))
        end
    elseif instruction_name == "Y"
        for qubit in qubit_indices
            push!(qc, put(qubit_number, qubit+1 => Y))
        end
    elseif instruction_name == "Z"
        for qubit in qubit_indices
            push!(qc, put(qubit_number, qubit+1 => Z))
        end
    elseif instruction_name == "S"
        for qubit in qubit_indices
            push!(qc, put(qubit_number, qubit+1 => ConstGate.S))
        end
    elseif instruction_name == "T"
        for qubit in qubit_indices
            push!(qc, put(qubit_number, qubit+1 => ConstGate.T))
        end
    elseif instruction_name == "C_XYZ"
        for qubit in qubit_indices
            push!(qc, put(qubit_number, qubit+1 => matblock(ComplexF64[1-im  -1-im; 1-im  1+im]/2)))
        end
    elseif instruction_name == "CNOT" || instruction_name == "CX"
        # CNOT requires pairs of qubits
        if !isempty(record_idx)
            c = condition(measure_list[end + record_idx[1]+1], X, nothing)
            push!(qc, put(qubit_number, qubit_indices.+1 => c))
        else
            for i in 1:2:length(qubit_indices)
                if i + 1 <= length(qubit_indices)
                    control_qubit = qubit_indices[i]
                    target_qubit = qubit_indices[i + 1]
                    push!(qc, control(qubit_number, control_qubit+1, (target_qubit+1) => X))
                end
            end
        end
    elseif instruction_name == "CZ"
        # CZ requires pairs of qubits
        if !isempty(record_idx)
            c = condition(measure_list[end + record_idx[1]+1], Z, nothing)
            push!(qc, put(qubit_number, qubit_indices.+1 => c))
        else
            for i in 1:2:length(qubit_indices)
                if i + 1 <= length(qubit_indices)
                    control_qubit = qubit_indices[i]
                    target_qubit = qubit_indices[i + 1]
                    push!(qc, control(qubit_number, control_qubit+1, (target_qubit+1) => Z))
                end
            end
        end
    elseif instruction_name in ["M", "MZ", "MX", "MY", "MR"]
        # Measurement operations
        for qubit in qubit_indices
            if instruction_name == "MX"
                op = X
            elseif instruction_name == "MY"
                op = Y
            else 
                op = ComputationalBasis()
            end
            if instruction_name == "MR"
                resetto = bit"0"
            else
                resetto = nothing
            end
            if isempty(arguments)
                m = NumberedMeasure(Measure(1; operator = op, resetto = resetto), length(measure_list)+1)
            else
                m = NumberedMeasure(Measure(1; operator = op, error_prob = arguments[1], resetto = resetto), length(measure_list)+1)
            end
            push!(qc, put(qubit_number, qubit+1 => m))
            push!(measure_list, m)
            push!(measure_pos_list, qubit+1)
        end
    elseif instruction_name == "R"
        # Reset operations
        for qubit in qubit_indices
            m = Measure(1;resetto=bit"0")
            push!(qc, put(qubit_number, qubit+1 => m))
        end
    elseif instruction_name == "RX"
        # Reset operations
        for qubit in qubit_indices
            m = Measure(1;resetto=bit"0")
            push!(qc, put(qubit_number, qubit+1 => m))
            push!(qc,put(qubit_number, qubit+1 => H))
        end
    elseif instruction_name == "TICK"
        # TICK is just a timing marker - no quantum operation
        return
    elseif instruction_name == "DEPOLARIZE1"
        for qubit in qubit_indices
            push!(qc, put(qubit_number, qubit+1 => DepolarizingChannel(1, arguments[1])))
        end
    elseif instruction_name == "DEPOLARIZE2"
        for i in 1:2:length(qubit_indices)
            if i + 1 <= length(qubit_indices)
                control_qubit = qubit_indices[i]
                target_qubit = qubit_indices[i + 1]
                push!(qc, put(qubit_number, (control_qubit+1, target_qubit+1) => DepolarizingChannel(2, arguments[1])))
            end
        end
    elseif instruction_name == "X_ERROR"
        for qubit in qubit_indices
            push!(qc, put(qubit_number, qubit+1 => quantum_channel(BitFlipError(arguments[1]))))
        end
    elseif instruction_name == "DETECTOR"
        detector_num[1] += 1
        db = DetectorBlock{2}(measure_list[end + 1 .+ record_idx], detector_num[1], 0)
        push!(qc, put(qubit_number, measure_pos_list[end + 1 + record_idx[1]] => db))
    elseif instruction_name == "OBSERVABLE_INCLUDE"
        detector_num[1] += 1
        ld = DetectorBlock{2}(measure_list[end + 1 .+ record_idx], detector_num[1], 1)
        push!(qc, put(qubit_number, measure_pos_list[end + 1 + record_idx[1]] => ld))
    elseif instruction_name in ["QUBIT_COORDS", "SHIFT_COORDS"]
        # Annotations - skip for now
        return
    elseif instruction_name in ["{", "}"]
        return
    else
        error("Unknown instruction: $instruction_name")
    end
end

function parse_dem_file(file_path::String)
    content = read(file_path, String)
    return parse_dem_string(content)
end

function parse_dem_string(content::String)
    lines = split(content, '\n')
    error_rates = Vector{Float64}()
    flipped_detectors = Vector{Vector{Int}}()
    flipped_logicals = Vector{Vector{Int}}()
    for line in lines
        if isempty(line)
            continue
        end
        s = split(line)
        if startswith(s[1], "error(")
            num = parse(Float64, s[1][7:end-1])
            push!(error_rates, num)
            detectors = Vector{Int}()
            logicals = Vector{Int}()
            for j in 2:length(s)
                if startswith(s[j], "D")
                    push!(detectors, parse(Int, s[j][2:end])+1)
                end
                if startswith(s[j], "L")
                    push!(logicals, parse(Int, s[j][2:end])+1)
                end
            end
            push!(flipped_detectors, detectors)
            push!(flipped_logicals, logicals)
        elseif startswith(s[1], "repeat")
            error("Repeat is not supported, use `circuit.detector_error_model(flatten_loops=True)` to flatten the loops in stim")
        end
    end
    largest_detector = maximum(maximum.(flipped_detectors))
    flipped_logicals = broadcast(x -> x .+ largest_detector, flipped_logicals)
    largest_logical = maximum(broadcast(x -> isempty(x) ? 0 : maximum(x), flipped_logicals))
    return DetectorErrorModel(error_rates, flipped_detectors .∪ flipped_logicals, collect(1:largest_detector), collect(largest_detector+1:largest_logical))
end

# Warning: This function is not fully tested.
function dump_stim_file(qc::ChainBlock, filename::String)
    qc= YaoBlocks.Optimise.simplify(qc; rules=[to_basictypes, Optimise.eliminate_nested])
    measure_list = Int[]
    open(filename, "w") do io
        for gate in qc
            # For each gate in the ChainBlock, write its STIM representation to the file.
            # This is a minimal implementation for common gates; extend as needed.
            if gate isa PutBlock
                # Single-qubit gate
                @assert length(gate.locs) == 1
                loc = gate.locs[1]
                g = gate.content
                if g isa ConstGate.HGate
                    println(io, "H ", loc-1)
                elseif g isa ConstGate.XGate
                    println(io, "X ", loc-1)
                elseif g isa ConstGate.YGate
                    println(io, "Y ", loc-1)
                elseif g isa ConstGate.ZGate
                    println(io, "Z ", loc-1)
                elseif g isa NumberedMeasure
                    println(io, "M ", loc-1)
                    push!(measure_list, g.num)
                elseif g isa DetectorBlock
                    if g.detector_type == 0
                        print(io, "DETECTOR ")
                        for m in g.vm
                            print(io, "rec[", findfirst(==(m.num), measure_list)-length(measure_list)-1, "] ")
                        end
                        println(io)
                    elseif g.detector_type == 1
                        print(io, "OBSERVABLE_INCLUDE(0) ")
                        for m in g.vm
                            print(io, "rec[", findfirst(==(m.num), measure_list)-length(measure_list)-1, "] ")
                        end
                        println(io)
                    end
                else
                    error("Unsupported gate: $(typeof(g))")
                end
            elseif gate isa ControlBlock
                # Two-qubit controlled gates (assume CNOT for now)
                ctrl = gate.ctrl_locs[1] - 1
                tgt = gate.locs[1] - 1
                if gate.content isa ConstGate.XGate
                    println(io, "CX $ctrl $tgt")
                else
                    error("Unsupported controlled gate: $(typeof(gate))")
                end
            elseif gate isa MeasureBlock
                # Measurement
                for loc in gate.locations
                    println(io, "M ", loc-1)
                end
            else
                error("Unsupported block: $(typeof(gate))")
            end
        end
    end
end

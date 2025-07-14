function parse_stim_string(content::String, qubit_number::Int)
    lines = split(content, '\n')
    qc = chain(qubit_number)
    
    i = 1
    while i <= length(lines)
        line = strip(lines[i])
        @show line
        if startswith(line, "#") || isempty(line)
            i += 1
            continue
        end
        
        # Remove inline comments (everything after #)
        if occursin("#", line)
            comment_start = findfirst("#", line)
            @show comment_start
            line = strip(line[1:(comment_start.start-1)])
        end
        
        # Parse the instruction
        parts = split(line)
        if isempty(parts)
            i += 1
            continue
        end
        
        instruction_name = uppercase(parts[1])
        
        @show parts
        
        # Check if this is a REPEAT block
        if instruction_name == "REPEAT"
            # Parse REPEAT block
            repeat_count = parse(Int, parts[2])
            @show repeat_count
            
            # Find the block content
            block_lines = String[]
            i += 1  # Move to next line
            brace_count = 0
            
            while i <= length(lines)
                block_line = strip(lines[i])
                
                if startswith(block_line, "#") || isempty(block_line)
                    i += 1
                    continue
                end
                
                # Remove inline comments
                if occursin("#", block_line)
                    comment_start = findfirst("#", block_line)
                    block_line = strip(block_line[1:(comment_start.start-1)])
                end
                
                if block_line == "{"
                    brace_count += 1
                    i += 1
                    continue
                elseif block_line == "}"
                    brace_count -= 1
                    if brace_count == 0
                        break
                    end
                end
                
                if brace_count > 0
                    push!(block_lines, block_line)
                end
                
                i += 1
            end
            
            # Apply the block instructions repeat_count times
            for _ in 1:repeat_count
                for block_line in block_lines
                    if isempty(block_line)
                        continue
                    end
                    
                    # Parse block instruction
                    block_parts = split(block_line)
                    if isempty(block_parts)
                        continue
                    end
                    
                    block_instruction_name = uppercase(block_parts[1])
                    
                    # Extract targets (skip instruction name)
                    block_targets = block_parts[2:end]
                    
                    # Parse arguments if present
                    block_arguments = Float64[]
                    block_target_start = 2
                    
                    # Check if there are arguments in parentheses
                    if length(block_parts) > 1 && startswith(block_parts[2], "(")
                        # Find the closing parenthesis
                        arg_parts = String[]
                        paren_count = 0
                        j = 2
                        
                        while j <= length(block_parts)
                            part = block_parts[j]
                            
                            if startswith(part, "(")
                                paren_count += 1
                            end
                            
                            if endswith(part, ")")
                                paren_count -= 1
                                # Remove closing parenthesis
                                if paren_count == 0
                                    part = part[1:end-1]
                                    if !isempty(part)
                                        push!(arg_parts, part)
                                    end
                                    break
                                end
                            end
                            
                            push!(arg_parts, part)
                            j += 1
                        end
                        
                        # Parse arguments
                        for part in arg_parts
                            for arg_str in split(part, ",")
                                arg_str = strip(arg_str)
                                if !isempty(arg_str)
                                    try
                                        arg = parse(Float64, arg_str)
                                        push!(block_arguments, arg)
                                    catch
                                        error("Invalid argument: $arg_str")
                                    end
                                end
                            end
                        end
                        
                        block_target_start = j + 1
                    end
                    
                    # Extract targets after arguments
                    block_targets = block_parts[block_target_start:end]
                    
                    # Parse targets and apply gates
                    block_qubit_indices = Int[]
                    
                    for target in block_targets
                        if isempty(target)
                            continue
                        end
                        
                        # Handle different target types
                        if target == "*"
                            # Combiner - skip for now
                            continue
                        elseif startswith(target, "rec[") && endswith(target, "]")
                            # Measurement record target - skip for now
                            continue
                        elseif startswith(target, "sweep[") && endswith(target, "]")
                            # Sweep bit target - skip for now
                            continue
                        elseif startswith(target, "!")
                            # Inverted target
                            try
                                qubit_idx = parse(Int, target[2:end])
                                push!(block_qubit_indices, qubit_idx)
                            catch
                                error("Invalid inverted target: $target")
                            end
                        elseif length(target) >= 2 && target[1] in ['X', 'Y', 'Z']
                            # Pauli target
                            try
                                qubit_idx = parse(Int, target[2:end])
                                push!(block_qubit_indices, qubit_idx)
                            catch
                                error("Invalid Pauli target: $target")
                            end
                        else
                            # Regular qubit target
                            try
                                qubit_idx = parse(Int, target)
                                push!(block_qubit_indices, qubit_idx)
                            catch
                                error("Invalid qubit target: $target")
                            end
                        end
                    end
                    
                    # Apply the appropriate gate based on instruction name
                    apply_gate!(qc, qubit_number, block_instruction_name, block_qubit_indices)
                end
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
        if length(parts) > 1 && startswith(parts[2], "(")
            # Find the closing parenthesis
            arg_parts = String[]
            paren_count = 0
            j = 2
            
            while j <= length(parts)
                part = parts[j]
                
                if startswith(part, "(")
                    paren_count += 1
                end
                
                if endswith(part, ")")
                    paren_count -= 1
                    # Remove closing parenthesis
                    if paren_count == 0
                        part = part[1:end-1]
                        if !isempty(part)
                            push!(arg_parts, part)
                        end
                        break
                    end
                end
                
                push!(arg_parts, part)
                j += 1
            end
            
            # Parse arguments
            for part in arg_parts
                for arg_str in split(part, ",")
                    arg_str = strip(arg_str)
                    if !isempty(arg_str)
                        try
                            arg = parse(Float64, arg_str)
                            push!(arguments, arg)
                        catch
                            error("Invalid argument: $arg_str")
                        end
                    end
                end
            end
            
            target_start = j + 1
        end
        
        # Extract targets after arguments
        targets = parts[target_start:end]
        
        # Parse targets and apply gates
        qubit_indices = Int[]
        
        for target in targets
            if isempty(target)
                continue
            end
            
            # Handle different target types
            if target == "*"
                # Combiner - skip for now
                continue
            elseif startswith(target, "rec[") && endswith(target, "]")
                # Measurement record target - skip for now
                continue
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
        apply_gate!(qc, qubit_number, instruction_name, qubit_indices)
        
        i += 1
    end
    
    return qc
end

# Helper function to apply gates
function apply_gate!(qc, qubit_number::Int, instruction_name::String, qubit_indices::Vector{Int})
    if instruction_name == "H"
        for qubit in qubit_indices
            @show qubit
            push!(qc, put(qubit_number,qubit+1 => H))
        end
    elseif instruction_name == "X"
        for qubit in qubit_indices
            qc = put(qc, qubit, X)
        end
    elseif instruction_name == "Y"
        for qubit in qubit_indices
            qc = put(qc, qubit, Y)
        end
    elseif instruction_name == "Z"
        for qubit in qubit_indices
            qc = put(qc, qubit, Z)
        end
    elseif instruction_name == "S"
        for qubit in qubit_indices
            push!(qc, put(qubit_number, qubit+1 => ConstGate.S))
        end
    elseif instruction_name == "T"
        for qubit in qubit_indices
            qc = put(qc, qubit, T)
        end
    elseif instruction_name == "CNOT" || instruction_name == "CX"
        # CNOT requires pairs of qubits
        for i in 1:2:length(qubit_indices)
            if i + 1 <= length(qubit_indices)
                control_qubit = qubit_indices[i]
                target_qubit = qubit_indices[i + 1]
                @show control_qubit, target_qubit
                push!(qc, control(qubit_number, control_qubit+1, (target_qubit+1) => X))
            end
        end
    elseif instruction_name == "CZ"
        # CZ requires pairs of qubits
        for i in 1:2:length(qubit_indices)
            if i + 1 <= length(qubit_indices)
                control_qubit = qubit_indices[i]
                target_qubit = qubit_indices[i + 1]
                push!(qc, control(qubit_number, control_qubit+1, (target_qubit+1) => Z))
            end
        end
    elseif instruction_name == "M" || instruction_name == "MR"
        # Measurement operations
        push!(qc, Measure(qubit_number; locs = qubit_indices.+1))
    elseif instruction_name == "R"
        # Reset operations
        for qubit in qubit_indices
            qc = put(qc, qubit, Reset)
        end
    elseif instruction_name == "TICK"
        # TICK is just a timing marker - no quantum operation
        return
    elseif startswith(instruction_name, "DEPOLARIZE")
        # Noise operations - skip for now
        return
    elseif instruction_name in ["DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"]
        # Annotations - skip for now
        return
    else
        @warn "Unknown instruction: $instruction_name"
    end
end
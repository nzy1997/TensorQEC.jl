function parse_stim_string(content::String,qubit_number::Int)
    lines = split(content, '\n')
    qc = chain(qubit_number)
    
    for line in lines
        line = strip(line)
        @show line
        # if !startswith(line, "#")
        #     continue
        # end
        
        # Remove # and trim
        comment = strip(line[2:end])
        @show comment
        if !isempty(comment)
            push!(circuit.comments, comment)
            
            # Parse key-value pairs like "task: memory"
            if occursin(":", comment)
                parts = split(comment, ":", limit=2)
                if length(parts) == 2
                    key = strip(parts[1])
                    value = strip(parts[2])
                    circuit.metadata[key] = value
                end
            end
        end
    end

    # Parse metadata from comments at the beginning
    # parse_metadata!(circuit, lines)
    
    # Parse instructions
    # i = 1
    # while i <= length(lines)
    #     line = strip(lines[i])
        
    #     # Skip empty lines and comments
    #     if isempty(line) || startswith(line, "#")
    #         i += 1
    #         continue
    #     end
        
    #     # Parse instruction or block
    #     instruction_or_block, new_i = parse_line(lines, i)
    #     if instruction_or_block !== nothing
    #         push!(circuit.instructions, instruction_or_block)
    #     end
    #     i = new_i
    # end
    
    return qc
end
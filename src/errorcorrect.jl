function correction_pauli_string(qubit_num::Int, syn::Dict{Int,Bool}, prob::Dict{Int,Vector{Float64}})
    ps = ones(Int, qubit_num)
    for (k, v) in prob
        @show syn[k]
        if syn[k] #syn[k] is true, measure outcome is -1, use X or Y
            ps[k] = (findmax(v)[2]) == 1 ? 2 : 3
        elseif findmax(v)[2] == 2
            ps[k] = 4
        end
    end
    return PauliString(ps[1:end]...)
end



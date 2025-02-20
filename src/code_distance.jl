function code_distance(H::Matrix{Int}; verbose = false)
    # H : m x n

    m,n = size(H)
    model = Model(HiGHS.Optimizer)
    !verbose && set_silent(model)

    @variable(model, 0 <= z[i = 1:n] <= 1, Int)
    @variable(model, 0 <= k[i = 1:m], Int)
    
    for i in 1:m
        @constraint(model, sum(z[j] for j in 1:n if H[i,j] == 1) == 2 * k[i])
    end
    @constraint(model, sum(z[j] for j in 1:n) >= 1)

    @objective(model, Min, sum(z[j] for j in 1:n))
    optimize!(model)
    @assert is_solved_and_feasible(model) "The problem is infeasible!"
    return  objective_value(model)
end

function code_distance(H::AbstractMatrix; verbose = false)
    return code_distance(Matrix{Int}(H); verbose = verbose)
end

function row_echelon_form(H::Matrix{Bool})
    bimat = SimpleBimatrix(H)
    gaussian_elimination!(bimat,1:size(H, 1), 0, 0;allow_col_operation = false)
    return bimat.matrix
end
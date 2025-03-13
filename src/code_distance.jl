function code_distance(H::Matrix{Int}; verbose = false,ipsolver = SCIP.Optimizer)
    m,n = size(H)
    model = Model(ipsolver)
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

function null_space(H::Matrix{Bool})
    reH = row_echelon_form(H)
    m,n = size(H)

    local x
    pivots = [x for i in 1:m if begin x = findfirst(!iszero, reH[i,:]); !isnothing(x) end]

    rank = length(pivots)
    null_space = zeros(Bool, n-rank, n)

    row_count = 0
    for j in 1:n
        if j ∉ pivots
            row_count += 1
            null_space[row_count, j] = 1
            for i in 1:length(pivots)
                reH[i,j] && (null_space[row_count, pivots[i]] = true)
            end
        end
    end
    return null_space
end

function logical_oprator(Hx::Matrix{Bool}, Hz::Matrix{Bool})
    kerHx = null_space(Hx)

    H = [Hz; kerHx]
    bimat = gaussian_elimination!(SimpleBimatrix(H), 1:size(H, 1), 0, 0)

    lz = bimat.matrix[1+size(Hz, 1):end, :]

    lz = lz[any.(!iszero, eachrow(lz)), :]
    lz[:,bimat.ordering] = lz
    return lz
end

function logical_oprator(tanner::CSSTannerGraph)
    lz = logical_oprator([a.x for a in tanner.stgx.H], [a.x for a in tanner.stgz.H])
    lx = logical_oprator([a.x for a in tanner.stgz.H], [a.x for a in tanner.stgx.H])
    return lx, lz
end

function code_distance(Hz::Matrix{Int},lz::Matrix{Int}; verbose = false,ipsolver = SCIP.Optimizer)
    m,n = size(Hz)
    num_lz = size(lz, 1)
    model = Model(ipsolver)
    !verbose && set_silent(model)

    @variable(model, 0 <= z[i = 1:n] <= 1, Int)
    @variable(model, 0 <= k[i = 1:m], Int)
    @variable(model, 0 <= l[i = 1:num_lz], Int)
    @variable(model, 0 <= r[i = 1:num_lz] <= 1, Int)
    
    for i in 1:m
        @constraint(model, sum(z[j] for j in 1:n if Hz[i,j] == 1) == 2 * k[i])
    end

    for i in 1:num_lz
        @constraint(model, sum(z[j] for j in 1:n if lz[i,j] == 1) == 2*l[i] + r[i])
    end
    @constraint(model, sum(r[i] for i in 1:num_lz) >= 1)

    @objective(model, Min, sum(z[j] for j in 1:n))
    optimize!(model)
    @assert is_solved_and_feasible(model) "The problem is infeasible!"
    return  objective_value(model)
end

function code_distance(tanner::CSSTannerGraph)
    lx,lz = logical_oprator(tanner)
    dx = code_distance(Int.(tanner.stgz.H), Int.(lz))
    dz = code_distance(Int.(tanner.stgx.H), Int.(lx))
    return min(dx,dz)
end
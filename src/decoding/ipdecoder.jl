"""
    IPDecoder <: AbstractDecoder

An integer programming decoder.
"""
Base.@kwdef struct IPDecoder <: AbstractDecoder 
    optimizer = SCIP.Optimizer
    verbose::Bool = false
end

function decode(decoder::IPDecoder, prob::SimpleDecodingProblem, syndrome::Vector{Mod2})
    H = [a.x for a in prob.tanner.H]
    return DecodingResult(true,_mixed_integer_programming(decoder, H, [s.x for s in syndrome], prob.pvec))
end

function _mixed_integer_programming(decoder::IPDecoder, H::Matrix{Bool}, syndrome::Vector{Bool}, p_vec::Vector{Float64})
    m,n = size(H)
    model = Model(decoder.optimizer)
    !decoder.verbose && set_silent(model)

    @variable(model, 0 <= z[i = 1:n] <= 1, Int)
    @variable(model, 0 <= k[i = 1:m], Int)
    
    for i in 1:m
        @constraint(model, sum(z[j] for j in 1:n if H[i,j] == 1) == 2 * k[i] + (syndrome[i] ? 1 : 0))
    end
    obj = 0.0
    for j in 1:n
        if p_vec[j] == 0.0
            @constraint(model, z[j] == 0)
        else
            obj += log(p_vec[j]) * z[j] + log(1-p_vec[j]) * (1 - z[j])
        end
    end
    @objective(model, Max, obj)
    optimize!(model)
    @assert is_solved_and_feasible(model) "The problem is infeasible!"
    return Mod2.(value.(z) .> 0.5)
end

function decode(decoder::IPDecoder, prob::CSSDecodingProblem, syndrome::CSSSyndrome)
    tanner = prob.tanner
    p_vec = prob.pvec
    Hx = [a.x for a in tanner.stgx.H]
    Hz = [a.x for a in tanner.stgz.H]
    mx,n = size(Hx)
    mz = size(Hz, 1)

    model = Model(decoder.optimizer)
    !decoder.verbose && set_silent(model)

    @variable(model, 0 <= x[i = 1:n] <= 1, Int)
    @variable(model, 0 <= y[i = 1:n] <= 1, Int)
    @variable(model, 0 <= z[i = 1:n] <= 1, Int)
    @variable(model, 0 <= k[i = 1:mx], Int)
    @variable(model, 0 <= l[i = 1:mz], Int)
    
    for i in 1:mx
        @constraint(model, sum(z[j]+y[j] for j in 1:n if Hx[i,j] == 1) == 2 * k[i] + (syndrome.sx[i].x ? 1 : 0))
    end
    for i in 1:mz
        @constraint(model, sum(x[j]+y[j] for j in 1:n if Hz[i,j] == 1) == 2 * l[i] + (syndrome.sz[i].x ? 1 : 0))
    end

    for i in 1:n
        @constraint(model, x[i] + y[i] +z[i] <= 1)
    end

    obj = 0.0
    for j in 1:n
        if p_vec[j].px == 0.0
            @constraint(model, x[j] == 0)
        else
            obj += log(p_vec[j].px) * x[j]
        end
        if p_vec[j].py == 0.0
            @constraint(model, y[j] == 0)
        else
            obj += log(p_vec[j].py) * y[j]
        end
        if p_vec[j].pz == 0.0
            @constraint(model, z[j] == 0)
        else
            obj += log(p_vec[j].pz) * z[j]
        end
        obj += log(1- p_vec[j].px - p_vec[j].py - p_vec[j].pz) * (1 - x[j] - y[j] - z[j])
    end
    @objective(model, Max, obj)
    optimize!(model)
    @assert is_solved_and_feasible(model) "The problem is infeasible!"
    return CSSDecodingResult(true, Mod2.(value.(x) .> 0.5) .+ Mod2.(value.(y) .> 0.5), Mod2.(value.(z) .> 0.5) .+ Mod2.(value.(y) .> 0.5))
end
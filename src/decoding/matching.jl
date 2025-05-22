abstract type MatchingSolver end

"""
    MatchingDecoder{T<:MatchingSolver} <: AbstractDecoder

A decoder that uses matching algorithm.
Fields:
- `solver::T`: the matching solver
"""
struct MatchingDecoder{T<:MatchingSolver} <: AbstractClassicalDecoder 
    solver::T
end
struct IPMatchingSolver <: MatchingSolver end

# The last element is boundary
struct FWSWeightedGraph{T}
    edges::Vector{SimpleWeightedEdge{Int,T}}
    v2e::Vector{Vector{Int}}
    dists::Matrix{T}
    error_path:: Matrix{Vector{Int}}
end
Graphs.nv(fwg::FWSWeightedGraph) = length(fwg.v2e)


function fws_edges(tanner::SimpleTannerGraph, p::Vector{Float64})
    g = SimpleWeightedGraph(tanner.ns + 1)
    edge_mat = zeros(Int,tanner.ns + 1,tanner.ns + 1) # edge_mat[i,j] is the qubit of the edge from i to j
    for (i,q2s_vec) in enumerate(tanner.q2s)
       if length(q2s_vec) == 1
            min_add_edge!(g, q2s_vec[1], tanner.ns + 1, log((1-p[i])/p[i]),edge_mat,i)
       else
            min_add_edge!(g, q2s_vec[1], q2s_vec[2],  log((1-p[i])/p[i]),edge_mat,i)
       end
    end
    fws = floyd_warshall_shortest_paths(g)
    return fws, collect(edges(g)),edge_mat
end

function tanner2fwswg(tanner::SimpleTannerGraph, p::Vector{Float64})
    @assert maximum(length.(tanner.q2s)) <= 2 "Each error causes either one or two syndromes"
    fws,edge_vec,edge_mat_qubit = fws_edges(tanner,p)
    v2e = [Vector{Int}() for _ in 1:tanner.ns + 1]
    qubit_vec = Int[]
    edge_mat = zeros(Int,tanner.ns + 1,tanner.ns + 1)
    for (i,e) in enumerate(edge_vec)
        push!(v2e[e.src], i)
        push!(v2e[e.dst], i)
        push!(qubit_vec, edge_mat_qubit[e.src,e.dst])
        edge_mat[e.src,e.dst] = i
        edge_mat[e.dst,e.src] = i
    end
    parent_path = [Vector{Int}() for _ in 1:tanner.ns + 1, _ in 1:tanner.ns + 1]

    for i in 1:(tanner.ns + 1)
        for j in 1:(tanner.ns + 1)
            (i == j) && continue
            parent_path[i,j] = path_vec(edge_mat,qubit_vec,fws.parents,i,j)
        end
    end
    return FWSWeightedGraph(edge_vec,v2e , fws.dists,parent_path)
end

function path_vec(edge_mat,qubit_vec, parents,s,d)
    ans_vec = Vector{Int}()
    while d != s
        d_new = parents[s,d]
        e = edge_mat[d,d_new]
        push!(ans_vec,qubit_vec[e])
        d = d_new
    end
    return ans_vec
end

function min_add_edge!(g, s, d, w,edge_mat,i)
    if iszero(g.weights[s,d]) || g.weights[s,d] > w
        add_edge!(g,s,d,w)
        edge_mat[s,d] = i
        edge_mat[d,s] = i
    end
end

# The last element is boundary
struct MatchingWithBoundary{MT <: AbstractMatrix}
    adj_mat::MT
end

function FWSWGtoMWB(fwg::FWSWeightedGraph,syndrome::Vector{Mod2})
    syndrome_vertex = [findall(v->v.x,syndrome)...,nv(fwg)]
    return MatchingWithBoundary(view(fwg.dists,syndrome_vertex,syndrome_vertex))
end

function extract_decoding(fwg::FWSWeightedGraph{T},  edge_vec::Vector,qubit_num::Int) where T
    edge_vec_long = fill(Mod2(0), qubit_num)
    for edge in edge_vec
        vec = fwg.error_path[edge[1],edge[2]]
        edge_vec_long[vec] .= edge_vec_long[vec] .+ Mod2(1)
    end
    return edge_vec_long
end


function solve_matching(mwb::MatchingWithBoundary, matching_solver::IPMatchingSolver)
    num_vertices = size(mwb.adj_mat,1)

    model = Model(SCIP.Optimizer)
    set_silent(model)
    
    @variable(model, 0 <= z[i = 1:num_vertices - 1,j = (i+1) : num_vertices] <= 1, Int)

    for i in 1:(num_vertices-1)
        @constraint(model, sum(i < j ? z[i,j] : z[j,i] for j in 1:(num_vertices) if i != j) == 1)
    end

    obj = 0.0
    for i in 1:num_vertices
        for j in (i+1):num_vertices
            obj += mwb.adj_mat[i,j] * z[i,j]
        end
    end

    @objective(model, Min, obj)
    optimize!(model)
    @assert is_solved_and_feasible(model) "The problem is infeasible!"

    ans_pairs = collect((value.(z).> 0.5).data)
    sydrome_vec = mwb.adj_mat.indices[1]
    return [sydrome_vec[collect(pair)] for pair in getfield.(ans_pairs[findall(x -> x.second, ans_pairs)],:first)]
end

struct GreedyMatchingSolver <: MatchingSolver end

function solve_matching(mwb::MatchingWithBoundary, matching_solver::GreedyMatchingSolver)
    view_mat = mwb.adj_mat
    indices_vec = mwb.adj_mat.indices[1]
    boundary_ind = indices_vec[end]
    ans_vec = Vector{Vector{Int}}()
    while length(indices_vec) > 1
        _,ind = findmax(view_mat)
        inds = indices_vec[collect(ind.I)]
        push!(ans_vec,inds)
        setdiff!(indices_vec,setdiff(inds, boundary_ind))
        view_mat = view(mwb.adj_mat.parent,indices_vec,indices_vec)
    end
    return ans_vec
end

struct CompiledMatching{ET} <: CompiledDecoder
    solver::ET
    qubit_num::Int
    fwg::FWSWeightedGraph
end

function compile(decoder::MatchingDecoder, prob::ClassicalDecodingProblem)
    fwg = tanner2fwswg(prob.tanner,prob.pvec)
    return CompiledMatching(decoder.solver,prob.tanner.nq,fwg)
end

function decode(cm::CompiledMatching,syndrome::SimpleSyndrome)
    mwb = FWSWGtoMWB(cm.fwg,syndrome.s)
    ev = solve_matching(mwb,cm.solver)
    return DecodingResult(true, extract_decoding(cm.fwg,ev,cm.qubit_num))
end

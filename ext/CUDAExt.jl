module CUDAExt

using CUDA; CUDA.allowscalar(false)
using CUDA.GPUArrays: @kernel, get_backend, @index
using TensorQEC
using TensorQEC: SpinConfig, SpinGlassSA, Mod2
using TensorQEC.Graphs

function TensorQEC.anneal_run!(config::SpinConfig{<:CuVector}, sap::SpinGlassSA{T}, betas::Vector{T}; num_trials) where T
	zero_config, one_config = TensorQEC.get01configs(config, sap.logical_qubits, sap.logical_qubits_check)
    batched_config = map(x->x.x, repeat(config.config, 1, num_trials))
    s2q = vcat(sap.s2q..., sap.logical_qubits)
    s2q_ptr = cumsum([1, length.(sap.s2q)..., length(sap.logical_qubits)])
    partitions = partite_stabilizers([sap.s2q..., sap.logical_qubits])
    # assert each partition does not share any qubits
    @assert all(p -> length(union(p...)) == sum(length.(p)), partitions)
    @assert sum(length, partitions) == length(sap.s2q) + 1

    _anneal_kernel!(batched_config, betas, CuVector{Int32}(s2q), CuVector{Int32}(s2q_ptr), CuVector(sap.logp_vector_error), CuVector(sap.logp_vector_noerror), CuVector{Int32}.(partitions))
    one_count = sum(reduce(⊻, batched_config[sap.logical_qubits_check, :]; dims=1, init=false))
    return (; mostlikely = one_count / num_trials > 0.5 ? one_config : zero_config, p1 = one_count / num_trials)
end

function _anneal_kernel!(batched_config::CuMatrix{Bool}, betas::Vector{T}, s2q::CuVector{Int32}, s2q_ptr::CuVector{Int32}, logp_vector_error::CuVector{T}, logp_vector_noerror::CuVector{T}, partitions::Vector{<:CuVector{Int32}}) where T
    # (blockIdx() shi 当前block的索引, 
    # blockDim() shi 当前block中thread的数量
    # threadIdx() shi 当前thread的索引

    function anneal_kernel!(batched_config, beta, s2q, s2q_ptr, logp_vector_error, logp_vector_noerror, partition, maxpart)
        index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        (i, ic) = divrem(index - Int32(1), maxpart) .+ Int32(1)
        if ic <= length(partition) && i <= size(batched_config, Int32(2))
            icheck = partition[ic]
            ΔE = zero(T)
            @inbounds for idx in s2q_ptr[icheck]:s2q_ptr[icheck+Int32(1)]-Int32(1)
                loc = s2q[idx]
                ΔE += batched_config[loc, i] ? logp_vector_error[loc] - logp_vector_noerror[loc] : logp_vector_noerror[loc] - logp_vector_error[loc]
            end
            if CUDA.Random.rand(T) < exp(-beta * ΔE)  # accept
                @inbounds for idx in s2q_ptr[icheck]:s2q_ptr[icheck+Int32(1)]-Int32(1)  # flip the qubits
                    loc = s2q[idx]
                    batched_config[loc, i] = !batched_config[loc, i]
                end
            end
        end
    end
    maxpart = maximum(length.(partitions))
    kernel = @cuda launch=false anneal_kernel!(batched_config, first(betas), s2q, s2q_ptr, logp_vector_error, logp_vector_noerror, first(partitions), maxpart)
    config = launch_configuration(kernel.fun)
    threads = min(maxpart * size(batched_config, 2), config.threads)
    blocks = cld(maxpart * size(batched_config, 2), threads)
    for beta in betas
        #for partition in partitions
        for partition in partitions
            CUDA.@sync kernel(batched_config, beta, s2q, s2q_ptr, logp_vector_error, logp_vector_noerror, partition, maxpart; threads, blocks)
        end
    end
end

function partite_stabilizers(s2q::Vector{Vector{Int}})
    # construct a graph from s2q, each stabilizer is a vertex, two vertices are connected if the stabilizers share the same qubit
    # Create a graph where stabilizers are vertices
    g = SimpleGraph(length(s2q))

    # Add edges between stabilizers that share qubits
    for i in 1:length(s2q)
        for j in i+1:length(s2q)
            if any(in(s2q[i]), s2q[j])
                add_edge!(g, i, j)
            end
        end
    end
    
    # Partition the stabilizers into independent sets
    return partite_vertices(g)
end

function greedy_coloring(g::SimpleGraph)
    coloring = zeros(Int, nv(g))
    for node in vertices(g)
        used_neighbour_colors = unique!([coloring[nbr] for nbr in neighbors(g, node) if coloring[nbr] != 0])
        coloring[node] = findfirst(∉(used_neighbour_colors), 1:nv(g))
    end
    return coloring
end

function dual_graph(g::SimpleGraph)
    gdual = SimpleGraph(ne(g))
    for (i, e1) in enumerate(edges(g)), (j, e2) in enumerate(edges(g))
        if src(e1) == dst(e2) || src(e1) == src(e2) || dst(e1) == dst(e2) || dst(e1) == src(e2)
            add_edge!(gdual, i, j)
        end
    end
    return gdual
end

function is_valid_coloring(g::SimpleGraph, coloring::Vector{Int})
    !any(e -> coloring[src(e)] == coloring[dst(e)], edges(g))
end

function partite_vertices(g::SimpleGraph)
    coloring = greedy_coloring(g)
    return [findall(==(i), coloring) for i in 1:maximum(coloring)]
end
partite_edges(g::SimpleGraph) = partite_vertices(dual_graph(g))

function togpu(sap::SpinGlassSA)
    return SpinGlassSA(
        CuVector{Int32}(sap.s2qx),
        CuVector{Int32}(sap.s2q_ptrx),
        CuVector{Int32}(sap.s2qz),
        CuVector{Int32}(sap.s2q_ptrz),
        CuMatrix(sap.lx),
        CuMatrix(sap.lz),
        CuVector{Int32}(sap.xlogical_qubits),
        CuVector{Int32}(sap.xlogical_qubits_ptr),
        CuVector{Int32}(sap.zlogical_qubits),
        CuVector{Int32}(sap.zlogical_qubits_ptr),
        CuMatrix(sap.logpx_diff),
        CuMatrix(sap.logpz_diff),
        sap.betas,
        sap.num_trials,
        sap.tanner
    )
end

end


module CUDAExt

using CUDA; CUDA.allowscalar(false)
using CUDA.GPUArrays: @kernel, get_backend, @index
using TensorQEC
using TensorQEC: SpinGlassSA, Mod2, VecPtr, getview, read_tensor_deltaE
using TensorQEC.Graphs

function TensorQEC.anneal_run!(config::CuVector{Mod2}, sap::SpinGlassSA)
    batched_config = map(x->x.x, repeat(config, 1, sap.num_trials))

    partitions = partite_stabilizers(sap.vecvecops)
    # assert each partition does not share any qubits
    @assert all(p -> length(union(p...)) == sum(length.(p)), partitions)
    T = eltype(sap.logp.vec)
    _anneal_kernel!(batched_config, sap.betas, CuVector{Int32}(sap.ops.vec),CuVector{Int32}(sap.ops.ptr), CuVector{T}(sap.logp.vec),CuVector{Int32}(sap.logp.ptr), CuVector{Int32}(sap.logp2bit.vec),CuVector{Int32}(sap.logp2bit.ptr), CuVector{Int32}(sap.bit2logp.vec),CuVector{Int32}(sap.bit2logp.ptr), CuVector{Int32}.(partitions))
    logical_num = length(sap.ops_check)
    vec = zeros(Int,2^logical_num)
    for trial in 1:sap.num_trials
        possum = 1
        for j in 1:logical_num
            CUDA.@allowscalar possum += reduce(⊻, batched_config[getview(sap.ops_check,j), trial]; dims=1, init=false)[] ? (1 << (j-1)) : 0
        end
        vec[possum] += 1
    end
    return vec./sap.num_trials
end

function _anneal_kernel!(batched_config::CuMatrix{Bool}, betas::Vector{T}, ops_vec,ops_ptr, logp_vec,logp_ptr,logp2bit_vec,logp2bit_ptr,bit2logp_vec,bit2logp_ptr, partitions::Vector{<:CuVector{Int32}}) where T
    function anneal_kernel!(batched_config, beta, ops_vec,ops_ptr, logp_vec,logp_ptr,logp2bit_vec,logp2bit_ptr,bit2logp_vec,bit2logp_ptr, partition, maxpart)
        index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        (i, ic) = divrem(index - Int32(1), maxpart) .+ Int32(1)
        if ic <= length(partition) && i <= size(batched_config, Int32(2))
            icheck = partition[ic]
            ΔE = zero(T)
            @inbounds for loc_index in ops_ptr[icheck]:ops_ptr[icheck+Int32(1)]-Int32(1)
                loc = ops_vec[loc_index]
                for j_index in bit2logp_ptr[loc]:bit2logp_ptr[loc+Int32(1)]-Int32(1)
                    j = bit2logp_vec[j_index]
                    possum = logp_ptr[j]
                    possum_trans = Int32(0)
                    for (bit_order,bit_index) in enumerate(logp2bit_ptr[j]:logp2bit_ptr[j+Int32(1)]-Int32(1))
                        bit_num = logp2bit_vec[bit_index]
                        batched_config[bit_num,i] && (possum += (Int32(1) << (bit_order-1)))
                        if bit_num == loc
                            possum_trans = batched_config[loc,i] ? -(Int32(1) << (bit_order-1)) : (Int32(1) << (bit_order-1))
                        end
                    end
                    ΔE += logp_vec[possum] - logp_vec[possum + possum_trans]
                end
            end
            
            if CUDA.Random.rand(T) < exp(-beta * ΔE)  # accept
                @inbounds for loc_index in ops_ptr[icheck]:ops_ptr[icheck+Int32(1)]-Int32(1)
                    loc = ops_vec[loc_index]
                    batched_config[loc, i] = !batched_config[loc, i]
                end
            end
        end
    end
    maxpart = maximum(length.(partitions))
    kernel = @cuda launch=false anneal_kernel!(batched_config, first(betas), ops_vec,ops_ptr, logp_vec,logp_ptr,logp2bit_vec,logp2bit_ptr,bit2logp_vec,bit2logp_ptr, first(partitions), maxpart)
    config = launch_configuration(kernel.fun)
    threads = min(maxpart * size(batched_config, 2), config.threads)
    blocks = cld(maxpart * size(batched_config, 2), threads)
    for beta in betas
        for partition in partitions
            CUDA.@sync kernel(batched_config, beta, ops_vec,ops_ptr, logp_vec,logp_ptr,logp2bit_vec,logp2bit_ptr,bit2logp_vec,bit2logp_ptr, partition, maxpart; threads, blocks)
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
TensorQEC.togpu(config::Vector{Mod2}) = CUDA.CuVector(config)
end
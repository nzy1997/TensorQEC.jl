module CUDAExt

using CUDA; CUDA.allowscalar(false)
using CUDA.GPUArrays: @kernel, get_backend, @index
using TensorQEC
using TensorQEC: SpinConfig, SpinGlassSA, Mod2

function TensorQEC.anneal_singlerun!(config::SpinConfig{<:CuVector}, sap::SpinGlassSA{T}, betas::Vector{T}; num_trials) where T
	zero_config, one_config = TensorQEC.get01configs(config, sap.logical_qubits, sap.logical_qubits_check)
    batched_config = map(x->x.x, repeat(config.config, 1, num_trials))
    s2q = vcat(sap.s2q..., sap.logical_qubits)
    s2q_ptr = cumsum([1, length.(sap.s2q)..., length(sap.logical_qubits)])
    _anneal_kernel!(batched_config, CuVector(betas), CuVector(s2q), CuVector(s2q_ptr), CuVector(sap.logp_vector_error), CuVector(sap.logp_vector_noerror))
    one_count = sum(reduce(⊻, batched_config[sap.logical_qubits_check, :]; dims=1, init=false))
    return (; mostlikely = one_count / num_trials > 0.5 ? one_config : zero_config, p1 = one_count / num_trials)
end

function _anneal_kernel!(batched_config::CuMatrix{Bool}, betas::CuVector{T}, s2q::CuVector{Int}, s2q_ptr::CuVector{Int}, logp_vector_error::CuVector{T}, logp_vector_noerror::CuVector{T}) where T
    function anneal_kernel!(batched_config, betas, s2q, s2q_ptr, logp_vector_error, logp_vector_noerror)
        i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
        if i <= size(batched_config, 2)
            for beta in betas
                for icheck in 1:length(s2q_ptr)-1
                    ΔE = zero(T)
                    @inbounds for idx in s2q_ptr[icheck]:s2q_ptr[icheck+1]-1
                        loc = s2q[idx]
                        ΔE += batched_config[loc, i] ? logp_vector_error[loc] - logp_vector_noerror[loc] : logp_vector_noerror[loc] - logp_vector_error[loc]
                    end
                    prob = ΔE <= 0 ? 1.0 : exp(-beta * ΔE)
                    if CUDA.Random.rand() < prob  # accept
                        @inbounds for idx in s2q_ptr[icheck]:s2q_ptr[icheck+1]-1  # flip the qubits
                            loc = s2q[idx]
                            batched_config[loc, i] = !batched_config[loc, i]
                        end
                    end
                end
            end
        end
    end
    kernel = @cuda launch=false anneal_kernel!(batched_config, betas, s2q, s2q_ptr, logp_vector_error, logp_vector_noerror)
    config = launch_configuration(kernel.fun)
    threads = min(size(batched_config, 2), config.threads)
    blocks = cld(size(batched_config, 2), threads)
    CUDA.@sync kernel(batched_config, betas, s2q, s2q_ptr, logp_vector_error, logp_vector_noerror; threads, blocks)
end

end


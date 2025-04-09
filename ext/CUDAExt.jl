module CUDAExt

using CUDA; CUDA.allowscalar(false)
using CUDA.GPUArrays: @kernel, get_backend, @index
using TensorQEC
using TensorQEC: SpinConfig, SpinGlassSA

function TensorQEC.anneal_singlerun!(config::SpinConfig{<:CuVector}, sap::SpinGlassSA{T}, betas::Vector{T}; num_trials) where T
	zero_config, one_config = TensorQEC.get01configs(config, sap.logical_qubits, sap.logical_qubits_check)
    batched_config = repeat(config.config, 1, num_trials)
    s2q = vcat(sap.s2q..., sap.logical_qubits)
    s2q_ptr = cumsum([1, length.(sap.s2q)...]).+1
    @kernel function anneal_kernel!(batched_config, betas, s2q, s2q_ptr, logp_vector_error, logp_vector_noerror)
        i = @index(Global, Linear)
        if i <= num_trials
            for beta in betas
                for ispin in 1:length(s2q)+1
                    ΔE = zero(T)
                    fliprange = s2q_ptr[ispin]:s2q_ptr[ispin+1]-1
                    @inbounds for loc in fliprange
                        loc = s2q[loc]
                        ΔE += batched_config[loc, i].x ? logp_vector_error[loc] - logp_vector_noerror[loc] : logp_vector_noerror[loc] - logp_vector_error[loc]
                    end
                    prob = ΔE <= 0 ? 1.0 : CUDA.@fastmath exp(-beta * ΔE)
                    if CUDA.Random.rand(Float32) < prob  # accept
                        for loc in fliprange
                            loc = s2q[loc]
                            batched_config[loc, i] = batched_config[loc, i] + Mod2(1)
                        end
                    end
                end
            end
        end
    end
    backend = get_backend(batched_config)
    CUDA.@sync anneal_kernel!(backend)(batched_config, CuVector(betas), CuVector(s2q), CuVector(s2q_ptr), CuVector(sap.logp_vector_error), CuVector(sap.logp_vector_noerror); ndrange=size(batched_config, 2))
    one_count = sum(getfield.(sum(batched_config[:, sap.logical_qubits_check]; dims=1), :x))
    return (; mostlikely = one_count / num_trials > 0.5 ? one_config : zero_config, p1 = one_count / num_trials)
end

end


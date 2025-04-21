struct VecPtr{VT <: AbstractVector, VIT <: AbstractVector}
	vec::VT
	ptr::VIT
end
getview(vptr::VecPtr, i::Integer) = view(vptr.vec, vptr.ptr[i]:vptr.ptr[i+1]-1)
Base.length(vptr::VecPtr) = length(vptr.ptr) - 1

struct SpinGlassSA{VT, VIT, T} 
	ops::VecPtr{VIT,VIT}
	ops_check::VecPtr{VIT,VIT}
	logp::VecPtr{VT,VIT}
	logp2bit::VecPtr{VIT,VIT}
	bit2logp::VecPtr{VIT,VIT}
	betas::Vector{T}
	num_trials::Int
	partitions::Vector{Vector{Int}}
end

function generate_spin_glass_sa(tanner::CSSTannerGraph, ide::IndependentDepolarizingError, betas::Vector{T}, num_trials::Int,use_cuda::Bool; IT::Type{<:Integer} = Int32) where {T}
	qubit_num = nq(tanner)

	lx,lz = logical_operator(tanner)
	xlogical_qubits = [findall(i->i.x,row) for row in eachrow(lx)]
	zlogical_qubits = [findall(i->i.x,row) for row in eachrow(lz)]

	vecvecops = vcat(tanner.stgx.s2q, broadcast.(+,tanner.stgz.s2q,qubit_num),xlogical_qubits, broadcast.(+,zlogical_qubits,qubit_num))
	ops = _vecvec2vecptr(vecvecops, IT,IT)
	ops_check = _vecvec2vecptr(vcat(zlogical_qubits, broadcast.(+,xlogical_qubits,qubit_num)), IT,IT)
	ops_correct = _vecvec2vecptr(vcat(xlogical_qubits, broadcast.(+,zlogical_qubits,qubit_num)), IT,IT)
	logp = _vecvec2vecptr([[log(one(T)-px-py-pz),log(px),log(pz),log(py)] for (px,py,pz) in zip(ide.px,ide.py,ide.pz)], IT,T)
	logp2bit = _vecvec2vecptr([[i,i+qubit_num] for i in 1:qubit_num], IT,IT)
	bit_vec = [[i] for i in 1:qubit_num]
	bit2logp = _vecvec2vecptr(vcat(bit_vec,bit_vec), IT,IT)
	partitions = use_cuda ? gpu_partite!(vecvecops,logp2bit) : [[0]]
	return SpinGlassSA(ops, ops_check, logp, logp2bit, bit2logp, betas, num_trials,partitions),ops_correct
end
gpu_partite!(vecvecops,logp2bit) = error("CUDA extension not loaded, try `using CUDA`")

"""
	anneal_run!(config, sap, betas::Vector{Float64}, num_sweep::Int)

Perform Simulated Annealing using Metropolis updates for the run.

# Arguments
* `config`: configuration that can be updated.
* `sap`: problem with `energy`, `flip!` and `random_config` interfaces.
* `betas`: inverse temperature scales, i.e. it can be [1.0, 0.5, 0.0]

# Keyword arguments
* `num_sweep`: the number of sweeps.
* `num_sweep_thermalize`: the number of sweeps for thermalization.
* `ptemp`: the probability of proposing a new beta.
* `rng`: the random number generator.
* `eta_lr`: the learning rate of eta.

Returns a named tuple: (; optimal_cost, optimal_configuration, p1, mostlikely_configuration, acceptance_rate, beta_acceptance_rate, valid_samples, etas).
"""
function anneal_run!(config::Vector{Mod2}, sap::SpinGlassSA; rng = Random.Xoshiro())
	betas = sap.betas
	num_trials = sap.num_trials
	logical_num = length(sap.ops_check)

	vec = zeros(Int,2^logical_num)
	for trial in 1:num_trials
		for beta in betas
			try_flip!(config, sap.logp, sap.logp2bit, sap.bit2logp, sap.ops, rng, beta)
		end

		possum = 1
		for j in 1:logical_num
			possum += sum(config[getview(sap.ops_check,j)]).x ? (1 << (j-1)) : 0
		end
		vec[possum] += 1
		@show sa_energy(config, sap)
	end
	return vec./num_trials
end

"""
	flip!(config::AnnealingConfig, ispin::Proposal, ap::AnnealingProblem) -> SpinConfig

Apply the change to the configuration.
"""
function flip!(config,fliplocs)
	for q in fliplocs
		@inbounds config[q] = config[q] .+ Mod2(1)
	end
	return config
end
struct SimulatedAnnealing{T} <: AbstractDecoder
	betas::Vector{T}
	num_trials::Int
	use_cuda::Bool
end
Base.show(io::IO, ::MIME"text/plain", ps::SimulatedAnnealing) = show(io, ps)
function Base.show(io::IO, ps::SimulatedAnnealing)
	print(io, "$(ps.use_cuda ? "CUDA" : "CPU")SA")
end

struct CompiledSpinGlassSA{VT, VIT, T} <: CompiledDecoder
	sap::SpinGlassSA{VT, VIT, T}
	tanner::CSSTannerGraph
	use_cuda::Bool
	ops_correct::VecPtr{VIT,VIT}
end

function compile(decoder::SimulatedAnnealing, problem::CSSDecodingProblem)
	prob,ops_correct = generate_spin_glass_sa(problem.tanner, problem.pvec, decoder.betas, decoder.num_trials,decoder.use_cuda)
	return CompiledSpinGlassSA(prob, problem.tanner, decoder.use_cuda,ops_correct)
end

function decode(cm::CompiledSpinGlassSA, syndrome::CSSSyndrome)
	config = CSSErrorPattern(TensorQEC._mixed_integer_programming_for_one_solution(cm.tanner, syndrome)...)
	config = vcat(config.xerror,config.zerror)
	cu_config = cm.use_cuda ? togpu(config) : config
    res = anneal_run!(cu_config, cm.sap)
	_, pos = findmax(res)

	for i in 1:length(cm.sap.ops_check)
		(reduce(+, config[getview(cm.sap.ops_check,i)]).x == (readbit(pos-1,i) == 1)) || (config[getview(cm.ops_correct,i)] .+= Mod2(1))
    end
	qubit_num = nq(cm.tanner)
	return SADecodingResult(true,CSSErrorPattern(config[1:qubit_num],config[qubit_num+1:end]),res)
end
togpu(config) = error("CUDA extension not loaded, try `using CUDA`")

function _vecvec2vecptr(vecvec::Vector, IT::Type{<:Integer},T2::Type)
	vec = T2.(vcat(vecvec...))
    ptr = IT.(cumsum([1, length.(vecvec)...]))
	return VecPtr(vec, ptr)
end

function try_flip!(config, logp::VecPtr{Vector{T}, Vector{IT}}, logp2bit, bit2logp, ops, rng, beta) where {T, IT}
	for index in 1:length(ops)
		fliplocs = getview(ops, index)
		ΔE = zero(T)
		@inbounds for i in fliplocs
			for j in getview(bit2logp,i)
				bit_nums = getview(logp2bit,j)
				ΔE += read_tensor_deltaE(getview(logp,j),findfirst(==(i),bit_nums),view(config,bit_nums))
			end
		end

		if ΔE <= 0
			flip!(config, fliplocs)
		else
			prob = @fastmath exp(-beta * ΔE)
			if rand(rng) < prob
				flip!(config, fliplocs)
			end
		end
	end 
end

function read_tensor_deltaE(vec,i,config)
	possum = 1
	for j in 1:length(config)
		possum += config[j].x ? (1 << (j-1)) : 0
	end
	deltaE = config[i].x ? vec[possum] - vec[possum - (1 << (i-1))] : vec[possum] - vec[possum + (1 << (i-1))]
	return deltaE
end

struct SADecodingResult{VT}
    success_tag::Bool
    error_qubits::CSSErrorPattern
    mar::VT
end

function sa_energy(config, sap::SpinGlassSA)
	energy = 0
	for j in 1:length(sap.logp)
		vec = getview(sap.logp,j)
		possum = 1 
		for (pos,i) in enumerate(getview(sap.logp2bit,j))
			possum += config[i].x ? (1 << (pos-1)) : 0
		end
		energy += vec[possum]
	end
	return energy
end
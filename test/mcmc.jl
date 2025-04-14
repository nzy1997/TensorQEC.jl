using TensorQEC
using Test
using TensorQEC: SpinGlassSA, SpinConfig, energy, propose, flip!,anneal_singlerun!
using TensorQEC.TensorInference
using Random

# @testset "energy" begin
#     tanner = CSSTannerGraph(SurfaceCode(3,3)).stgz
#     error_qubits = Mod2[0,0,0,1,0,0,0,0,0]
#     syd = syndrome_extraction(error_qubits, tanner)

#     p_vector = fill(0.1, 9)
#     prob = SpinGlassSA(tanner.s2q, syd.s, [1,2,3,10], p_vector)
#     config = SpinConfig(Mod2[0,0,0,0,0,0,0,0,0,0])
#     @show energy(config, prob)

#     config = SpinConfig(Mod2[0,0,0,1,0,0,0,0,0,1])
#     @show energy(config, prob)

#     config = SpinConfig(Mod2[0,0,0,1,0,0,0,0,0,0])
#     @show energy(config, prob)

#     proposal, ΔE = propose(config, prob)
#     config_new = flip!(deepcopy(config), proposal, prob)
#     @show config_new
#     @show energy(config, prob) - energy(config_new, prob)
#     @show ΔE
# end

@testset "anneal" begin
    tanner = CSSTannerGraph(SurfaceCode(3,3))
    error_qubits = Mod2[1,0,0,0,0,0,0,0,0]
    syd = syndrome_extraction(error_qubits, tanner.stgz)
    T = Float32

    p_vector = fill(T(0.1), 9)
    p_vector[2] = 0.26
    p_vector[3] = 0.26

    lx,lz = TensorQEC.logical_operator(tanner)
    @show lx
    @show lz

    prob = SpinGlassSA(tanner.stgx.s2q, findall(i -> i.x, lx[1,:]), p_vector, findall(i -> i.x, lz[1,:]))
    config = SpinConfig(Mod2[1,0,0,0,0,0,0,0,0])
    nsweeps = 1000
    res = anneal_singlerun!(config, prob, collect(T, 0:1e-4:1.0);num_trials=nsweeps)

    # config_new = copy(res.mostlikely.config)
    # config_new = Mod2[1,0,0,0,0,0,0,0,0]
    # (sum(lz[1,:].* config_new).x == (res.p1 > 0.5)) || (config_new += lx[1,:])
    @show res
    # @show config_new
end

using CUDA
@testset "anneal - CUDA" begin
    d = 21
    tanner = CSSTannerGraph(SurfaceCode(d,d))
    em = FlipError(0.1)
    error_qubits = random_error_qubits(d^2, em)
    syd = syndrome_extraction(error_qubits, tanner.stgz)
    T = Float32
    p_vector = fill(T(0.1), d^2)

    lx,lz = TensorQEC.logical_operator(tanner)

    prob = SpinGlassSA(tanner.stgx.s2q, findall(i -> i.x, lx[1,:]), p_vector, findall(i -> i.x, lz[1,:]))
    hz = tanner.stgz.H
    config = SpinConfig(TensorQEC._mixed_integer_programming_for_one_solution(hz, syd.s))
    num_trials = 1000
    @time CUDA.@sync res = anneal_singlerun!(SpinConfig(CUDA.CuVector(config.config)), prob, collect(T, 0:1e-6:1.0);num_trials)
    # @time res = anneal_singlerun!(SpinConfig(config.config), prob, collect(T, 0:1e-6:1.0);num_trials)
    @show res
end



@testset "marginals" begin
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d,d)).stgz
    error_qubits = Mod2[1,0,0,0,0,0,0,0,0]
    syd = syndrome_extraction(error_qubits, tanner)
    p_vector = fill(0.1, d*d)
    p_vector[2] = 0.26
    p_vector[3] = 0.26
    lx,lz = TensorQEC.logical_operator(CSSTannerGraph(SurfaceCode(d,d)))
    uaimodel = compile(TNMAP(), tanner, p_vector).uaimodel
    factors = uaimodel.factors
    newfactor_pos= length(factors)+1
    push!(uaimodel.factors, TensorInference.Factor((findall(i-> i.x, lz[1,:])...,newfactor_pos), TensorQEC.parity_check_matrix(d)))

    tn_new = TensorNetworkModel(
		1:newfactor_pos,
		fill(2,newfactor_pos),
		factors;
		mars = [[newfactor_pos]],
        evidence=Dict([(i+d*d,s.x ? 1 : 0) for (i,s) in enumerate(syd.s)])
	)

    res = TensorQEC._mixed_integer_programming_for_one_solution(tanner.H, syd.s)

    @show sum(res .* lz[1,:])
    @show sum((res.+lx[1,:]) .* lz[1,:])
    @show marginals(tn_new)

    @show res
end

@testset "mcmc d = 9" begin
    d = 9
    p = 0.15
    tanner = CSSTannerGraph(SurfaceCode(d,d))  
    p_vector = fill(p, d*d)
    Random.seed!(798899)
    error_qubits = random_error_qubits(d^2, FlipError(p))
    syd = syndrome_extraction(error_qubits, tanner.stgz)
    ct = compile(TNMAP(), tanner, [DepolarizingError(p_vector[i],0.0,0.0) for i in 1:d*d])
    css_syd = CSSSyndrome(zeros(Mod2,(d*d-1)÷2),syd.s)
    tnres = decode(ct, css_syd)
    @show tnres
#    Random.seed!(1234)  0.5914153505916218
#   Random.seed!(798899) 0.7758626199730365
    T = Float32
    lx,lz = TensorQEC.logical_operator(tanner)

    prob = SpinGlassSA(tanner.stgx.s2q, findall(i -> i.x, lx[1,:]), T.(p_vector), findall(i -> i.x, lz[1,:]))

    config = SpinConfig(TensorQEC._mixed_integer_programming_for_one_solution(tanner.stgz.H, syd.s))
    nsweeps = 1000
    res = anneal_singlerun!(config, prob, collect(T, 0:1e-5:1.0);num_trials=nsweeps)
    @show res
end 
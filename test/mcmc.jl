using TensorQEC
using Test
using TensorQEC: SpinGlassSA, SpinConfig, energy, propose, flip!,anneal_singlerun!
using TensorQEC.TensorInference
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

    prob = SpinGlassSA(tanner.stgx.s2q, findall(lx[1,:]), p_vector, findall(lz[1,:]))
    config = SpinConfig(Mod2[1,0,0,0,0,0,0,0,0])
    nsweeps = 1000
    res = anneal_singlerun!(config, prob, collect(T, 0:1e-4:1.0);num_trials=nsweeps)

    @show res
end

using CUDA
@testset "anneal - CUDA" begin
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

    prob = SpinGlassSA(tanner.stgx.s2q, findall(lx[1,:]), p_vector, findall(lz[1,:]))
    config = SpinConfig(Mod2[1,0,0,0,0,0,0,0,0])
    nsweeps = 1000
    res = anneal_singlerun!(SpinConfig(CUDA.CuVector(config.config)), prob, collect(T, 0:1e-4:1.0);num_trials=nsweeps)

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
    push!(uaimodel.factors, TensorInference.Factor((findall(lz[1,:])...,newfactor_pos), TensorQEC.parity_check_matrix(d)))

    tn_new = TensorNetworkModel(
		1:newfactor_pos,
		fill(2,newfactor_pos),
		factors;
		mars = [[newfactor_pos]],
        evidence=Dict([(i+d*d,s.x ? 1 : 0) for (i,s) in enumerate(syd.s)])
	)

    res = TensorQEC._mixed_integer_programming_for_one_solution(getfield.(tanner.H, :x), syd.s)

    @show sum(res .* lz[1,:])
    @show sum((res.+lx[1,:]) .* lz[1,:])
    @show marginals(tn_new)

    @show res
end

@testset "mcmc d = 5" begin
    T = Float32
    error_qubits = Mod2[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    tanner = CSSTannerGraph(SurfaceCode(5,5))
    syd = syndrome_extraction(error_qubits, tanner.stgz)
    p_vector = fill(T(0.1), 25)

    lx,lz = TensorQEC.logical_operator(tanner)
    prob = SpinGlassSA(tanner.stgx.s2q, findall(lx[1,:]), p_vector, findall(lz[1,:]))
    config = SpinConfig(TensorQEC._mixed_integer_programming_for_one_solution(getfield.(tanner.stgz.H, :x), syd.s))
    nsweeps = 100000
    res = anneal_singlerun!(config, prob, T[1.0, 0.8, 0.5, 0.0]; num_sweep=nsweeps,num_sweep_thermalize=100, ptemp=0.1)
    @show res
    @show res.mostlikely.config
    @show sum(lz.*res.mostlikely.config)
    @show sum(lz.*error_qubits)

    # ct = compile(TNMAP(), tanner, fill(DepolarizingError(0.15,0.0,0.0), 25))
    # css_syd = CSSSyndrome(zeros(Mod2,25),syd.s)
    # tnres = decode(ct, css_syd)
    # @show tnres
end 

@testset "marginals" begin
    d = 5
    tanner = CSSTannerGraph(SurfaceCode(d,d)).stgz
    error_qubits = Mod2[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    syd = syndrome_extraction(error_qubits, tanner)
    p_vector = fill(0.1, d*d)
    lx,lz = TensorQEC.logical_operator(CSSTannerGraph(SurfaceCode(d,d)))
    uaimodel = compile(TNMAP(), tanner, p_vector).uaimodel
    factors = uaimodel.factors
    newfactor_pos= length(factors)+1
    push!(uaimodel.factors, TensorInference.Factor((findall(lz[1,:])...,newfactor_pos), TensorQEC.parity_check_matrix(d)))

    tn_new = TensorNetworkModel(
		1:newfactor_pos,
		fill(2,newfactor_pos),
		factors;
		mars = [[newfactor_pos]],
        evidence=Dict([(i+d*d,s.x ? 1 : 0) for (i,s) in enumerate(syd.s)])
	)

    res = TensorQEC._mixed_integer_programming_for_one_solution(getfield.(tanner.H, :x), syd.s)

    @show sum(res .* lz[1,:])
    @show sum((res.+lx[1,:]) .* lz[1,:])
    @show marginals(tn_new)

    @show res
end
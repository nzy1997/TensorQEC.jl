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

    lx,lz = TensorQEC.logical_oprator(tanner)
    @show lx
    @show lz

    prob = SpinGlassSA(tanner.stgx.s2q, findall(lx[1,:]), p_vector, findall(lz[1,:]))
    config = SpinConfig(Mod2[1,0,0,0,0,0,0,0,0])
    nsweeps = 100000
    res = anneal_singlerun!(config, prob, T[1.0, 0.5, 0.0]; num_sweep=nsweeps, ptemp=0.1)
    #res = anneal_singlerun!(config, prob, T[1.0], T[0.0], 10000000; ptemp=0.1)
    @show res
    @test res.accept_rate > 0.05
    @test res.beta_accpet_rate > 0.1
    @test res.valid_samples > nsweeps
    @test isapprox(res.p1, 0.431, atol=0.3)
    @test res.mostlikely.config[1].x == 1

    @show  syndrome_extraction(res.mostlikely.config, tanner.stgz) == syd
    @show sum(lz.*res.mostlikely.config)
    @show sum(lz.*error_qubits)
end

@testset "marginals" begin
    tanner = CSSTannerGraph(SurfaceCode(3,3)).stgz
    error_qubits = Mod2[1,0,0,0,0,0,0,0,0]
    syd = syndrome_extraction(error_qubits, tanner)
    p_vector = fill(0.1, 9)
    p_vector[2] = 0.26
    p_vector[3] = 0.26
    lx,lz = TensorQEC.logical_oprator(CSSTannerGraph(SurfaceCode(3,3)))
    uaimodel = compile(TNMAP(), tanner, p_vector).uaimodel
    factors = uaimodel.factors
    push!(uaimodel.factors, TensorInference.Factor((findall(lz[1,:])...,14), TensorQEC.parity_check_matrix(3)))

    tn_new = TensorNetworkModel(
		1:14,
		fill(2,14),
		factors;
		mars = [[14]],
        evidence=Dict([(i+9,s.x ? 1 : 0) for (i,s) in enumerate(syd.s)])
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
    p_vector = fill(T(0.15), 25)

    lx,lz = TensorQEC.logical_oprator(tanner)
    prob = SpinGlassSA(tanner.stgx.s2q, findall(lx[1,:]), p_vector, findall(lz[1,:]))
    config = SpinConfig(TensorQEC._mixed_integer_programming_for_one_solution(getfield.(tanner.stgz.H, :x), syd.s))
    res = anneal_singlerun!(config, prob, T[1.0, 0.1], T[0.0, -5.0], 1000000; ptemp=0.1)
    @show res
    @show res.mostlikely.config
    @show sum(lz.*res.mostlikely.config)
    @show sum(lz.*error_qubits)
end 
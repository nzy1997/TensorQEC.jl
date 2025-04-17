using TensorQEC
using Test
using TensorQEC: generate_spin_glass_sa, flip!,anneal_singlerun!
using TensorQEC.TensorInference
using Random

@testset "bitflip probability" begin
    d = 3
    p_vector = fill(0.1, d*d)
    p_vector[2] = 0.2
    p_vector[3] = 0.2
    em = IndependentDepolarizingError(p_vector,p_vector,p_vector)
    tanner = CSSTannerGraph(SurfaceCode(d,d))
    Random.seed!(1234)
    error_qubits = random_error_qubits(em)
    syd = syndrome_extraction(error_qubits, tanner)
  
    ct = compile(TNMAP(), tanner, em)
    tnres = decode(ct, syd)
    @show tnres
end 

@testset "anneal" begin
    T = Float32
    d = 3
    p_vector = fill(0.1, d*d)
    em = IndependentDepolarizingError(T.(p_vector),T.(p_vector),T.(p_vector))
    tanner = CSSTannerGraph(SurfaceCode(d,d))
    Random.seed!(1234)
    error_qubits = random_error_qubits(em)
    syd = syndrome_extraction(error_qubits, tanner)

    config = CSSErrorPattern(TensorQEC._mixed_integer_programming_for_one_solution(tanner, syd)...)
    nsweeps = 1000
    prob = generate_spin_glass_sa(tanner, em)
    res = anneal_singlerun!(config, prob, collect(T, 0:1e-4:1.0),nsweeps)

    @show res
    # em = IndependentDepolarizingError(p_vector,p_vector,p_vector)
    # ct = compile(TNMAP(), tanner, em)
    # tnres = decode(ct, syd)
    # marz = [0.7611243449261554, 0.23887565507384462]
    # marx = [0.8949023344449739, 0.1050976655550262]
    # kron(marz, marx) = [0.681131953077318, 0.07999239184883748, 0.21377038136765592, 0.02510527370618872]
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
    p = 0.05
    tanner = CSSTannerGraph(SurfaceCode(d,d))  
    p_vector = fill(p, d*d)
    Random.seed!(798899)
    error_qubits = random_error_qubits(d^2, FlipError(p))
    syd = syndrome_extraction(error_qubits, tanner.stgz)
    ct = compile(TNMAP(), tanner, [DepolarizingError(p_vector[i],0.0,0.0) for i in 1:d*d])
    css_syd = CSSSyndrome(zeros(Mod2,(d*d-1)÷2),syd.s)
    tnres = decode(ct, css_syd)
    @show tnres
#    Random.seed!(1234)  0.5914153505916218 0.15
#   Random.seed!(798899) 0.7758626199730365 0.15
#   Random.seed!(1234)  0.00 0.05
    # Random.seed!(798899)  0.288157226241484 0.05

    T = Float32
    lx,lz = TensorQEC.logical_operator(tanner)

    prob = SpinGlassSA(tanner.stgx.s2q, findall(i -> i.x, lx[1,:]), T.(p_vector), findall(i -> i.x, lz[1,:]))

    config = SpinConfig(TensorQEC._mixed_integer_programming_for_one_solution(tanner.stgz.H, syd.s))
    nsweeps = 1000
    res = anneal_singlerun!(config, prob, collect(T, 0:1e-5:1.0);num_trials=nsweeps)
    @show res
end 
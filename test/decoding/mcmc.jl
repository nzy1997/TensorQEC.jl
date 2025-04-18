using TensorQEC
using Test
using TensorQEC: generate_spin_glass_sa, flip!,anneal_run!, _vecvec2vecptr
using TensorQEC.TensorInference
using Random

@testset "vecvec2vecptr" begin
    vecvec = [[1,2,3],[4,5,6],[7,8,9]]
    vec, ptr = _vecvec2vecptr(vecvec)
    @test vec == [1,2,3,4,5,6,7,8,9]
    @test ptr == [1,4,7,10]
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
    nsweeps = 100
    prob = generate_spin_glass_sa(tanner, em, collect(T, 0:1e-3:1.0), nsweeps)
    res = anneal_run!(config, prob)

    @test sum(abs.(res - [0.681131953077318, 0.07999239184883748, 0.21377038136765592, 0.02510527370618872])) < 0.4
    # @show res

    # em = IndependentDepolarizingError(p_vector,p_vector,p_vector)
    # ct = compile(TNMAP(), tanner, em)
    # tnres = decode(ct, syd)
    # marz = [0.7611243449261554, 0.23887565507384462]
    # marx = [0.8949023344449739, 0.1050976655550262]
    # kron(marz, marx) = [0.681131953077318, 0.07999239184883748, 0.21377038136765592, 0.02510527370618872]
end

@testset "compile and decode" begin   
    d = 3
    n = 2*d^2
    tanner = CSSTannerGraph(ToricCode(d,d))
    em = iid_error(0.05,0.05,0.05,n)
    ct = compile(SimulatedAnnealing(collect(0:1e-3:1.0),100,false), tanner, em)

    Random.seed!(1234)
    eq = random_error_qubits(em)
    syd = syndrome_extraction(eq, tanner)
    res = decode(ct, syd)
    @test syd == syndrome_extraction(res.error_qubits, tanner)
    @test !check_logical_error(res.error_qubits, eq, ct.lx, ct.lz)
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

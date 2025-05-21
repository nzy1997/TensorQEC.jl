using CUDA
using TensorQEC
using Test
using TensorQEC:SpinGlassSA, SpinConfig, anneal_singlerun!

@testset "anneal - CUDA" begin
    d = 21
    tanner = CSSTannerGraph(SurfaceCode(d,d))
    em = FlipError(0.1)
    error_qubits = random_error_qubits(d^2, em)
    syd = syndrome_extraction(error_qubits, tanner.stgz)
    T = Float32
    p_vector = fill(T(0.1), d^2)

    lx,lz = TensorQEC.logical_operator(tanner)

    prob = SpinGlassSA(tanner.stgx.s2q, findall(lx[1,:]), p_vector, findall(lz[1,:]))
    hz = getfield.(tanner.stgz.H, :x)
    config = SpinConfig(TensorQEC._mixed_integer_programming_for_one_solution(hz, syd.s))
    num_trials = 1000
    @time CUDA.@sync res = anneal_singlerun!(SpinConfig(CUDA.CuVector(config.config)), prob, collect(T, 0:1e-6:1.0); num_trials)
    # @time res = anneal_singlerun!(SpinConfig(config.config), prob, collect(T, 0:1e-6:1.0);num_trials)
    @show res
end
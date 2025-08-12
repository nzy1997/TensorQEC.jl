using TensorQEC
using Test
using TensorQEC: generate_spin_glass_sa, flip!,anneal_run!, _vecvec2vecptr
using TensorQEC.TensorInference
using Random

@testset "vecvec2vecptr" begin
    vecvec = [[1,2,3],[4,5,6],[7,8,9]]
    vecptr = _vecvec2vecptr(vecvec, Int32,Int32)
    @test vecptr.vec == [1,2,3,4,5,6,7,8,9]
    @test vecptr.ptr == [1,4,7,10]
    @test typeof(vecptr.ptr) == Vector{Int32}
    @test typeof(vecptr.vec) == Vector{Int32}
end

@testset "anneal" begin
    T = Float32
    d = 3

    em = iid_error(T(0.1),T(0.1),T(0.1),d*d)
    tanner = CSSTannerGraph(SurfaceCode(d,d))
    Random.seed!(1234)
    error_qubits = random_error_qubits(em)
    syd = syndrome_extraction(error_qubits, tanner)

    config = CSSErrorPattern(TensorQEC._mixed_integer_programming_for_one_solution(tanner, syd)...)
    nsweeps = 100
    prob,_ = generate_spin_glass_sa(tanner, em, collect(T, 0:1e-3:1.0), nsweeps,false)
    res = anneal_run!(vcat(config.xerror,config.zerror), prob)

    @test sum(abs.(res - [0.681131953077318, 0.07999239184883748, 0.21377038136765592, 0.02510527370618872])) < 0.4
    # @show res

    # p_vector = fill(0.1, d*d)
    # em = IndependentDepolarizingError(p_vector,p_vector,p_vector)
    # ct = compile(TNMAP(), tanner, em)
    # tnres = decode(ct, syd)
    # marz = [0.7611243449261554, 0.23887565507384462]
    # marx = [0.8949023344449739, 0.1050976655550262]
    # kron(marz, marx) = [0.681131953077318, 0.07999239184883748, 0.21377038136765592, 0.02510527370618872]
end

@testset "compile and decode" begin   
    d = 3
    # tanner = CSSTannerGraph(ToricCode(d,d))
    tanner = CSSTannerGraph(SurfaceCode(d,d))
    n = tanner.stgx.nq
    em = iid_error(0.05,0.05,0.05,n)
    ct = compile(TensorQEC.SimulatedAnnealing(collect(0:1e-3:1.0),100,false), tanner, em)

    Random.seed!(1234)
    eq = random_error_qubits(em)
    syd = syndrome_extraction(eq, tanner)
    res = decode(ct, syd)
    @test syd == syndrome_extraction(res.error_qubits, tanner)
    lx,lz = logical_operator(tanner)
    @test !check_logical_error(res.error_qubits, eq, lx, lz)
end

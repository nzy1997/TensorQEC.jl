using Test
using TensorQEC
using TensorQEC.Yao
using TensorQEC.OMEinsum
using TensorQEC.TensorInference
using Random

@testset "parity_check_matrix" begin
    mat = TensorQEC.parity_check_matrix(4)
    @test mat[1,1,1,1,1] == 1
    @test mat[1,1,1,1,2] == 0
    @test mat[1,2,1,2,1] == 1
    @test mat[2,1,2,2,1] == 0
end

@testset "decode" begin
    Random.seed!(123)
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d, d))
    em = iid_error(0.05,d*d)
    ep = random_error_qubits(em)
    syn = syndrome_extraction(ep,tanner.stgz)

    decoder = TNMAP()
    @show decoder
    res = decode(decoder,tanner.stgz,syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner.stgz)
end

@testset "compile and decode" begin
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d, d))

    ct = compile(TNMAP(),tanner.stgz)

    Random.seed!(123)
    em = iid_error(0.05,d*d)
    ep = random_error_qubits(em)
    syn = syndrome_extraction(ep,tanner.stgz)
    
    res = decode(ct,syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner.stgz)
end

@testset "CSS compile and decode" begin
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d, d))

    ct = compile(TNMAP(),tanner)

    Random.seed!(123)
    em = iid_error(0.05,0.05,0.05,d*d)
    ep = random_error_qubits(em)
    syn = syndrome_extraction(ep,tanner)
    
    res = decode(ct,syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner)
end

@testset "TNMMAP decoder" begin
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d,d))
    error_qubits = Mod2[1, 0, 0, 0, 0, 0, 0, 0, 0]
    syd = syndrome_extraction(error_qubits, tanner.stgz)
    lx,lz = logical_operator(tanner)
    p_vector = fill(0.1, d*d)
    @show TNMMAP()
    ct = compile(TNMMAP(), tanner, IndependentDepolarizingError(p_vector,fill(0.0,d*d),fill(0.0,d*d)))
    css_syd = CSSSyndrome(zeros(Mod2,(d*d-1)÷2),syd.s)
    tnres = decode(ct, css_syd)

    @test !check_logical_error(tnres.error_qubits, TensorQEC.CSSErrorPattern(Mod2[1, 0, 0, 0, 0, 0, 0, 0, 0], Mod2[0, 0, 0, 0, 0, 0, 0, 0, 0]), lx, lz)
    @test css_syd == syndrome_extraction(tnres.error_qubits, tanner)

    p_vector = fill(0.1, d*d)
    p_vector[2] = 0.26
    p_vector[3] = 0.26
    ct = compile(TNMAP(), tanner, IndependentDepolarizingError(p_vector,fill(0.0,d*d),fill(0.0,d*d)))
    css_syd = CSSSyndrome(zeros(Mod2,(d*d-1)÷2),syd.s)
    tnres = decode(ct, css_syd)
    @test check_logical_error(tnres.error_qubits, TensorQEC.CSSErrorPattern(Mod2[1, 0, 0, 0, 0, 0, 0, 0, 0], Mod2[0, 0, 0, 0, 0, 0, 0, 0, 0]), lx, lz)
    @test css_syd == syndrome_extraction(tnres.error_qubits, tanner)
end

@testset "compile and decode for TNMMAP" begin   
    d = 3
    tanner = CSSTannerGraph(ToricCode(d,d))
    n = tanner.stgx.nq
    em = iid_error(0.05,0.05,0.05,n)
    ct = compile(TNMMAP(), tanner, em)

    Random.seed!(1234)
    eq = random_error_qubits(em)
    syd = syndrome_extraction(eq, tanner)
    res = decode(ct, syd)
    @test syd == syndrome_extraction(res.error_qubits, tanner)
    lx,lz = logical_operator(tanner)
    @test !check_logical_error(res.error_qubits, eq, lx, lz)
end

@testset "TNMMAP decoder marginal probability" begin
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d,d))
    error_qubits = Mod2[1, 0, 0, 0, 0, 0, 0, 0, 0]
    syd = syndrome_extraction(error_qubits, tanner.stgz)
    lx,lz = logical_operator(tanner)
    p_vector = fill(0.1, d*d)
    ct = compile(TNMMAP(), tanner, IndependentDepolarizingError(p_vector,fill(0.0,d*d),fill(0.0,d*d)))
    @test ct.optcode(ct.tensors...) ≈ [0.3972875040000002 0.004284496000000001; 0.0 0.0] atol=1e-10
end

@testset "dem TNMMAP" begin
    dem = TensorQEC.parse_dem_file(joinpath(@__DIR__, "..", "stim_parser", "test_circuits", "dem.dem"))
    ct = compile(TNMMAP(TreeSA(), true), dem)

    Random.seed!(12323)
    ep = random_error_qubits(IndependentFlipError(dem.error_rates))
    syd = syndrome_extraction(ep, ct.tanner)
    res = decode(ct, syd)
    @test syd == syndrome_extraction(res.error_qubits, ct.tanner)
end
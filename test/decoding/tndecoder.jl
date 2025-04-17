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
    em = FlipError(0.05)
    ep = random_error_qubits(d*d, em)
    syn = syndrome_extraction(ep,tanner.stgz)

    decoder = TNMAP()
    res = decode(decoder,tanner.stgz,syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner.stgz)
end

@testset "compile and decode" begin
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d, d))

    ct = compile(TNMAP(),tanner.stgz)

    Random.seed!(123)
    em = FlipError(0.05)
    ep = random_error_qubits(d*d, em)
    syn = syndrome_extraction(ep,tanner.stgz)
    
    res = decode(ct,syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner.stgz)
end

@testset "CSS compile and decode" begin
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d, d))

    ct = compile(TNMAP(),tanner)

    Random.seed!(123)
    em = DepolarizingError(0.05)
    ep = random_error_qubits(d*d, em)
    syn = syndrome_extraction(ep,tanner)
    
    res = decode(ct,syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner)
end

@testset "bitflip probability" begin
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d,d))
    error_qubits = Mod2[1, 0, 0, 0, 0, 0, 0, 0, 0]
    syd = syndrome_extraction(error_qubits, tanner.stgz)
    lx,lz = logical_operator(tanner)
    p_vector = fill(0.1, d*d)
    ct = compile(TNMAP(), tanner, IndependentDepolarizingError(p_vector,fill(0.0,d*d),fill(0.0,d*d)))
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
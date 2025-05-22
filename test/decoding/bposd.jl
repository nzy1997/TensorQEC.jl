using Test
using TensorQEC
using Random

@testset "belief_propagation1" begin
    sts = [[1, 2,3,4],[2,3,5,7],[3,4,5,6]]
    nq = 7
    tanner = SimpleTannerGraph(nq, sts)
    error_qubits = Mod2[1,0,0,0,0,0,0]
    syn = syndrome_extraction(error_qubits, tanner)
    ct = compile(BPDecoder(),tanner)
    bpres = decode(ct,syn)
    @test bpres.error_qubits == error_qubits
end

@testset "check_linear_indepent" begin
    H = Bool[1 1 1 0; 0 1 1 1]
    @test check_linear_indepent(H) == true
    H = Bool[1 1 1 0; 0 1 1 0; 0 0 0 1; 1 0 0 0]
    @test check_linear_indepent(H) == false
    H = Mod2[1 1 1 0; 0 1 1 1]
    @test check_linear_indepent(H) == true
    H = Mod2[1 1 1 0; 0 1 1 0; 0 0 0 1; 1 0 0 0]
    @test check_linear_indepent(H) == false
end

@testset "osd" begin
    sts = [[1, 2,3,4],[2,3,5,7],[3,4,5,6]]
    nq = 7
    tanner = SimpleTannerGraph(nq, sts)

    error_qubits = Mod2[0,0,0,1,0,0,0]
    syd = syndrome_extraction(error_qubits, tanner)
    order = [1, 2, 3, 4, 5, 6, 7]
    osd_error = osd(tanner, order,syd.s)

    @test syd == syndrome_extraction(osd_error, tanner)
end

@testset "decode" begin
    Random.seed!(123)
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d, d))
    em = iid_error(0.05,d*d)
    ep = random_error_qubits(em)
    syn = syndrome_extraction(ep,tanner.stgz)

    res = decode(BPDecoder(100,false),tanner.stgz,syn)
    @test !res.success_tag || (syn == syndrome_extraction(res.error_qubits, tanner.stgz))

    res = decode(BPDecoder(),tanner.stgz,syn)
    @test (syn == syndrome_extraction(res.error_qubits, tanner.stgz))
end

@testset "compile and decode" begin
    Random.seed!(123)
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d, d))
    em = iid_error(0.05,d*d)
    ep = random_error_qubits(em)
    syn = syndrome_extraction(ep,tanner.stgz)

    ct = compile(BPDecoder(100,false),tanner.stgz)
    res = decode(ct,syn)
    @test !res.success_tag || (syn == syndrome_extraction(res.error_qubits, tanner.stgz))

    ct = compile(BPDecoder(),tanner.stgz)
    res = decode(ct,syn)
    @test (syn == syndrome_extraction(res.error_qubits, tanner.stgz))
end

@testset "CSS compile and decode" begin
    Random.seed!(123)
    d = 3
    tanner = CSSTannerGraph(SurfaceCode(d, d))
    em = iid_error(0.05,0.05,0.05,d*d)
    ep = random_error_qubits(em)
    syn = syndrome_extraction(ep,tanner)

    ct = compile(BPDecoder(100,false),tanner)
    res = decode(ct,syn)
    @test !res.success_tag || (syn == syndrome_extraction(res.error_qubits, tanner))

    ct = compile(BPDecoder(),tanner)
    res = decode(ct,syn)
    @test (syn == syndrome_extraction(res.error_qubits, tanner))
end
using Test, TensorQEC
using TensorQEC.BitBasis
using Random
using TensorQEC.OMEinsum
using TensorQEC.Yao

@testset "make table" begin
    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    tb = make_table(tanner,4,TensorQEC.UniformError())
    INT =BitBasis.LongLongUInt{1}
    @test tb.table[zero(INT)] == (zero(INT),zero(INT))

    syn = TensorQEC.CSSSyndrome(Mod2[0, 0, 1, 1], Mod2[1, 0, 0, 1])
    @test TensorQEC.syndrome2lluint(syn,INT) == INT(156)

    ep = TensorQEC.lluint2error(tb.table[INT(156)],9)
    @test syn == syndrome_extraction(ep, tanner)
end

@testset "show table" begin
    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    tb = make_table(tanner,1,TensorQEC.UniformError())
    @show tb
end

@testset "decode" begin
    Random.seed!(123)
    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    em = iid_error(0.05,0.06,0.1,9)
    ep = random_error_qubits(em)
    syn = syndrome_extraction(ep,tanner)
    tb = make_table(tanner,4,TensorQEC.UniformError())
    ct = TensorQEC.CompiledTable(tb)
    res = decode(ct, syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner)
end

@testset "lluint" begin
    filename = "test_table.txt"
    Random.seed!(123)
    tanner = CSSTannerGraph(SurfaceCode(9, 9))
    em = iid_error(0.01,0.01,0.01,81)
    ep = random_error_qubits(em)
    syn = syndrome_extraction(ep,tanner)

    tb = make_table(tanner,1,TensorQEC.UniformError())
    save_table(tb, filename)
    tb2 = load_table(filename, 81, 80)
    @test tb.table == tb2.table
    @test tb.num_qubits == tb2.num_qubits
    @test tb.num_st == tb2.num_st
    ct = TensorQEC.CompiledTable(tb2)
    res = decode(ct, syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner)
    rm("test_table.txt")
end

@testset "save table and load" begin
    filename = "test_table.txt"
    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    tb = make_table(tanner,4,TensorQEC.UniformError())
    save_table(tb, filename)
    tb2 = load_table(filename, 9, 8)
    @test tb.table == tb2.table
    @test tb.num_qubits == tb2.num_qubits
    @test tb.num_st == tb2.num_st
    rm("test_table.txt")
end

@testset "DepolarizingDistribution" begin
    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    tb = make_table(tanner,4, TensorQEC.DistributionError(iid_error(0.3,0.3,0.3,9)))
    INT =BitBasis.LongLongUInt{1}
    @test tb.table[zero(INT)] != (zero(INT),zero(INT))

    syn = TensorQEC.CSSSyndrome(Mod2[0, 0, 1, 1], Mod2[1, 0, 0, 1])
    ep = TensorQEC.lluint2error(tb.table[INT(156)],9)
    @test syn == syndrome_extraction(ep, tanner)
end

@testset "compile" begin
    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    problem = TensorQEC.IndependentDepolarizingDecodingProblem(tanner,iid_error(0.05,0.05,0.05,9))
    decoder = TableDecoder(3)
    compiled = compile(decoder, problem)
    error_qubits = random_error_qubits(iid_error(0.05,0.05,0.05,9))
    syn = syndrome_extraction(error_qubits, tanner)
    res = decode(compiled, syn)
    @test syn == syndrome_extraction(res.error_qubits, tanner)
end
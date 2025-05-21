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
    em = DepolarizingError(0.05, 0.06, 0.1)
    ep = random_error_qubits(9, em)
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
    em = DepolarizingError(0.01, 0.01, 0.01)
    ep = random_error_qubits(81, em)
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
    tb = make_table(tanner,4,TensorQEC.DepolarizingDistribution(fill(DepolarizingError(0.3, 0.3, 0.3), 9)))
    INT =BitBasis.LongLongUInt{1}
    @test tb.table[zero(INT)] != (zero(INT),zero(INT))

    syn = TensorQEC.CSSSyndrome(Mod2[0, 0, 1, 1], Mod2[1, 0, 0, 1])
    ep = TensorQEC.lluint2error(tb.table[INT(156)],9)
    @test syn == syndrome_extraction(ep, tanner)
end

@testset "DepolarizingDistribution" begin
    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    tb = make_table(tanner,4,TensorQEC.DepolarizingDistribution(fill(DepolarizingError(0.3, 0.3, 0.3), 9)))
    INT =BitBasis.LongLongUInt{1}
    @test tb.table[zero(INT)] != (zero(INT),zero(INT))

    syn = TensorQEC.CSSSyndrome(Mod2[0, 0, 1, 1], Mod2[1, 0, 0, 1])
    ep = TensorQEC.lluint2error(tb.table[INT(156)],9)
    @test syn == syndrome_extraction(ep, tanner)
end

@testset "TNDistribution" begin
    num_qubits = 9
    code = DynamicEinCode([[[i,i+num_qubits,i+2*num_qubits] for i in [1,2,3,5,6,8,9]]...,vcat([[i,i+num_qubits,i+2*num_qubits] for i in [4,7]]...)],Int[])
    a = rand(2,2,2,2,2,2)
    a = a./sum(a)
    tensors= [[TensorQEC.single_qubit_tensor(0.01,0.01,0.01) for j in [1,2,3,5,6,8,9]]...,a]
    tn = TensorNetwork(code,tensors)
    td = TensorQEC.TNDistribution(tn,9)
    INT = BitBasis.LongLongUInt{1}

    @test TensorQEC.get_probability(td,(INT(72),INT(8))) ≈ a[1,2,1,2,1,1] * 0.97^7 atol = 1e-8

    tanner = CSSTannerGraph(SurfaceCode(3, 3))
    tb = make_table(tanner,4,TensorQEC.TNDistribution(tn,9))

    syn = TensorQEC.CSSSyndrome(Mod2[0, 0, 1, 1], Mod2[1, 0, 0, 1])
    ep = TensorQEC.lluint2error(tb.table[INT(156)],9)
    @test syn == syndrome_extraction(ep, tanner)
end
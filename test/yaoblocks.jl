using TensorQEC
using TensorQEC: NumberedMeasure
using Yao

@testset "NumberedMeasure" begin
    m = NumberedMeasure(Measure(5;locs = 2), 1)
    reg = rand_state(5)
    @test !isdefined(m.m, :results)
    copy(reg) |> m
    @test isdefined(m.m, :results)

    qc = chain(5,m)
    vizcircuit(qc)
end

@testset "ConditionBlock" begin
    m = NumberedMeasure(Measure(1), 1)
    reg = ArrayReg(ComplexF64[0,1])
    c = TensorQEC.condition(m, X, nothing)
    @show c
    @test_throws UndefRefError reg |> c
    copy(reg) |> m
    @test (measure(reg |> c; nshots=10) .== 0) |> all

    m = NumberedMeasure(Measure(1; locs=(1,)), 1)
    reg = ArrayReg(ComplexF64[0,1])
    c = TensorQEC.condition(m, X, nothing)

    copy(reg) |> m
    @test all(measure(reg |> c; nshots=10) .==0)

    m = NumberedMeasure(Measure(1; locs=(1,)), 1)
    c = TensorQEC.condition(m, nothing, Z)
    @test_throws ArgumentError mat(c)
end

@testset "DetectorBlock" begin
    m1 = NumberedMeasure(Measure(1), 1)
    m2 = NumberedMeasure(Measure(1), 2)
    c = TensorQEC.DetectorBlock{2}([m1, m2], 1, 0)
    @show c

    c2 = TensorQEC.LogicalDetectorBlock{2}([m1, m2], 1, 1)
    @show c2

    qc = chain(1,m1,m2,c,c2)
    @test qc isa ChainBlock
    vizcircuit(qc)
end
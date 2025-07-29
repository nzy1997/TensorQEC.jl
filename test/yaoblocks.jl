using TensorQEC
using TensorQEC: NumberedMeasure
using Yao

@testset "NumberedMeasure" begin
    m = NumberedMeasure(Measure(5;locs = 2), 1)
    reg = rand_state(5)
    @test !isdefined(m.m, :results)
    copy(reg) |> m
    @test isdefined(m.m, :results)
end

@testset "ConditionBlock" begin
    m = NumberedMeasure(Measure(1), 1)
    reg = ArrayReg(ComplexF64[0,1])
    c = TensorQEC.condition(m, X, nothing)
    @show c
    copy(reg) |> m
    @test (measure(reg |> c; nshots=10) .== 0) |> all

    m = NumberedMeasure(Measure(1; locs=(1,)), 1)
    reg = ArrayReg(ComplexF64[0,1])
    c = TensorQEC.condition(m, X, nothing)

    copy(reg) |> m
    @test all(measure(reg |> c; nshots=10) .==0)
end

@testset "DetectorBlock" begin
    m1 = NumberedMeasure(Measure(1), 1)
    m2 = NumberedMeasure(Measure(1), 2)
    c = TensorQEC.DetectorBlock{2}([m1, m2])
    qc = chain(1,m1,m2,c)
    @test qc isa ChainBlock
end
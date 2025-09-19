using TensorQEC
using TensorQEC: NumberedMeasure
using Yao
using Test

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

    c2 = TensorQEC.DetectorBlock{2}([m1, m2], 1, 1)
    @show c2

    qc = chain(1,m1,m2,c,c2)
    @test qc isa ChainBlock
    vizcircuit(qc)
end

@testset "AtomLossBlock" begin
    c = TensorQEC.AtomLossBlock{2}(0.1)
    @show c

    c2 = TensorQEC.AtomLossBlock{2}(0.2)
    @show c2

    qc = chain(1,c,c2)
    @test qc isa ChainBlock
    vizcircuit(qc)

    reg = rand_state(1)
    @test_throws ErrorException apply!(reg, c)
end

@testset "clifford_simulate with AtomLossBlock" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    qc = chain(put(5, 1 => H), put(5, 3 => TensorQEC.AtomLossBlock{2}(1.0)), control(5, 1, 2 => Z), control(5, 3, 4 => X), put(5, 2 => H), control(5, 5, 3 => X), put(5, 1 => X))
    ps = PauliString((z, y, i, y, x))

    res = clifford_simulate(ps, qc)
    pg2 = res.output

    qc_r = chain(put(5, 1 => H), control(5, 1, 2 => Z), put(5, 2 => H), put(5, 1 => X))
    ps_r = PauliString((z, y, i, y, x))
    @test mat(qc_r) * mat(ps_r) * mat(qc_r)' ≈ mat(pg2)
end
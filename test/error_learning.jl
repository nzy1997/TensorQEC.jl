using Test
using TensorQEC
using TensorQEC: qc_probability,TrainingChannel,probability_tn_channel,channel2mat,channel2tensor
using TensorQEC.Yao
using TensorQEC.OMEinsum
using Random

@testset "qc_probability" begin
    qc = chain(put(2, 1 => H),  put(2, 2 => H))
    @test qc_probability(qc, ComplexF64[1,0,0,0]) ≈ 0.25
    @test qc_probability(qc, ComplexF64[0,1,0,0]) ≈ 0.25
    @test qc_probability(qc, ComplexF64[1/2,1/2,1/2,1/2]) ≈ 1
end

@testset "qc_probability random test" begin
    Random.seed!(1234)
    qc = chain(put(3, 1 => H),  put(3, 2 => H), control(3, 2, 3 => H), put(3, 1 => X), control(3, 1, 2 => Z),)
    reg = rand_state(3)
    @test qc_probability(qc, vec(reg.state)) ≈ abs2(apply(reg, qc').state[1])
end

@testset "depolarizing_channel" begin
    unitary_channel = depolarizing_channel(2, fill(1/16,16))
    channel2mat(unitary_channel)

    qc = chain(put(2, 1 => H),  put(2, 2 => H),put(2,(1,2)=>unitary_channel))
    probability_tn_channel(qc, ComplexF64[1,0,0,0])
end

@testset "probability_tn_channel" begin
    Random.seed!(1234)
    umat = rand_unitary(4)
    unitary_channel =UnitaryChannel([matblock(umat),kron(X,X)],[1,0])
    qc = chain(put(3, 1 => H),  put(3, 2 => H), control(3, 2, 3 => H), put(3, 1 => X), control(3, 1, 2 => Z),put(3,(1,2)=>unitary_channel))
    
    qc2 = chain(put(3, 1 => H),  put(3, 2 => H), control(3, 2, 3 => H), put(3, 1 => X), control(3, 1, 2 => Z),put(3,(1,2)=>matblock(umat)))
    reg = rand_state(3)
    @test probability_tn_channel(qc, vec(reg.state)) ≈ abs2(apply(reg, qc2').state[1])
end

@testset "probability_tn_channel" begin
    unitary_channel = depolarizing_channel(2, fill(1/16,16))
    qc = chain(put(2,(1,2)=>unitary_channel))
    
    reg = rand_state(2)
    @show probability_tn_channel(qc, vec(reg.state))
end
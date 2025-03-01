using Test
using TensorQEC
using TensorQEC: qc_probability, TrainingChannel, probability_tn_channel, channel2mat, channel2tensor, probability_channel, get_grad
using TensorQEC.Yao
using TensorQEC.OMEinsum
using Random

@testset "qc_probability" begin
    qc = chain(put(2, 1 => H), put(2, 2 => H))
    @test qc_probability(qc, ComplexF64[1, 0, 0, 0]) ≈ 0.25
    @test qc_probability(qc, ComplexF64[0, 1, 0, 0]) ≈ 0.25
    @test qc_probability(qc, ComplexF64[1/2, 1/2, 1/2, 1/2]) ≈ 1
end

@testset "qc_probability random test" begin
    Random.seed!(1234)
    qc = chain(put(3, 1 => H), put(3, 2 => H), control(3, 2, 3 => H), put(3, 1 => X), control(3, 1, 2 => Z))
    reg = rand_state(3)
    @test qc_probability(qc, vec(reg.state)) ≈ abs2(apply(reg, qc').state[1])
end


@testset "probability_tn_channel" begin
    Random.seed!(1234)
    umat = rand_unitary(4)
    unitary_channel = UnitaryChannel([matblock(umat), kron(X, X)], [1, 0])
    qc = chain(put(3, 1 => H), put(3, 2 => H), control(3, 2, 3 => H), put(3, 1 => X), control(3, 1, 2 => Z), put(3, (1, 2) => unitary_channel))

    qc2 = chain(put(3, 1 => H), put(3, 2 => H), control(3, 2, 3 => H), put(3, 1 => X), control(3, 1, 2 => Z), put(3, (1, 2) => matblock(umat)))
    reg = rand_state(3)
    @test probability_channel(qc, vec(reg.state)) ≈ abs2(apply(reg, qc2').state[1])
end

@testset "probability_tn_channel" begin
    unitary_channel = depolarizing_channel(2, fill(1 / 16, 16))
    qc = chain(put(2, (1, 2) => unitary_channel))

    reg = rand_state(2)
    @test probability_channel(qc, vec(reg.state)) ≈ 0.25
end

@testset "error_learning" begin
    p2 = 0.02
    unitary_channel2 = depolarizing_channel(2, [1 - 15 * p2, fill(p2, 15)...])

    Random.seed!(1234)
    umat = rand_unitary(4)
    qc = chain(put(2, (1, 2) => matblock(umat)), put(2, (1, 2) => unitary_channel2))

    state = [
        ComplexF64[1, 0, 0, 0], ComplexF64[0, 1, 0, 0],
        ComplexF64[0, 0, 1, 0], ComplexF64[0, 0, 0, 1],
        ComplexF64[1/2, 1/2, 1/2, 1/2], ComplexF64[1/2, -1/2, 1/2, -1/2], ComplexF64[1/2, 1/2, -1/2, -1/2], ComplexF64[1/2, -1/2, -1/2, 1/2]]
    td = TrainningData([probability_channel(qc, s) for s in state],state)

    pvec0 = fill(1 / 16, 16)
    alpha = 0.01
    for i in 1:10000
        unitary_channel = depolarizing_channel(2, pvec0)
        qc_train = chain(put(2, (1, 2) => matblock(umat)), put(2, (1, 2) => unitary_channel))

        grad = get_grad(qc_train,td)[1]
        pvec0 = pvec0 - alpha * grad
        pvec0 = max.(0, pvec0)
        pvec0 = pvec0 / sum(pvec0)

        @show i
        @show pvec0
        @show norm(grad)
        if norm(grad) < 1e-2
            break
        end
    end
end

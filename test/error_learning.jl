using Test
using TensorQEC
using TensorQEC: qc_probability, TrainingChannel, probability_tn_channel, channel2mat, channel2tensor, probability_channel, get_grad,generate_new_tensor,loss_function
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

@testset "generate_new_tensor" begin
    p2 = 0.02
    unitary_channel2 = depolarizing_channel(2, [1 - 15 * p2, fill(p2, 15)...])

    Random.seed!(1234)
    umat = rand_unitary(4)
    qc = chain(put(2, (1, 2) => matblock(umat)), put(2, (1, 2) => unitary_channel2))
    optnet,p_pos = probability_tn_channel(qc, ComplexF64[1, 0, 0, 0])

    p3 = 0.01
    new_t = generate_new_tensor(optnet.tensors,p_pos,ComplexF64[0, 1, 0, 0],[[1 - 15 * p3, fill(p3, 15)...]])

    unitary_channel2 = depolarizing_channel(2, [1 - 15 * p3, fill(p3, 15)...])
    qc = chain(put(2, (1, 2) => matblock(umat)), put(2, (1, 2) => unitary_channel2))
    @test probability_channel(qc, ComplexF64[0, 1, 0, 0]) ≈  optnet.code(new_t...)[]
end 

@testset "get_grad" begin
    p2 = 0.02
    unitary_channel2 = depolarizing_channel(2, [1 - 15 * p2, fill(p2, 15)...])

    Random.seed!(1234)
    umat = rand_unitary(4)
    qc = chain(put(2, (1, 2) => matblock(umat)), put(2, (1, 2) => unitary_channel2))
    optnet,p_pos = probability_tn_channel(qc, ComplexF64[1, 0, 0, 0])
    td = TrainningData([0.25],[ComplexF64[0, 1, 0, 0]])

    p = fill(0.01,15)
    Δ = 1e-6
    new_t1 = generate_new_tensor(optnet.tensors,p_pos,ComplexF64[0, 1, 0, 0],[[1 - sum(p), p...]])
    fx = (optnet.code(new_t1...)[] -0.25)^2
    p2 = copy(p)
    p2[5] += Δ
    new_t2 = generate_new_tensor(optnet.tensors,p_pos,ComplexF64[0, 1, 0, 0],[[1 - sum(p2), p2...]])
    fxpΔx = (optnet.code(new_t2...)[]-0.25)^2
    @test (fxpΔx-fx) / Δ ≈ get_grad(optnet.code,optnet.tensors,p_pos,td,[p])[1][5] atol=1e-8
end

@testset "error_learning" begin
    p2 = collect(0.01:0.005:0.10)[1:15]
    unitary_channel2 = depolarizing_channel(2, [1 - sum(p2), p2...])

    Random.seed!(1234)
    umat = rand_unitary(4)
    qc = chain(put(2, (1, 2) => matblock(umat)), put(2, (1, 2) => unitary_channel2))

    state = [
        ComplexF64[1, 0, 0, 0], ComplexF64[0, 1, 0, 0],
        ComplexF64[0, 0, 1, 0], ComplexF64[0, 0, 0, 1],
        ComplexF64[1/2, 1/2, 1/2, 1/2], ComplexF64[1/2, -1/2, 1/2, -1/2],
        ComplexF64[1/2, 1/2, -1/2, -1/2], ComplexF64[1/2, -1/2, -1/2, 1/2],
        ComplexF64[1/2, 0.5im, 0.5im, -1/2], ComplexF64[1/2, 0.5im, -0.5im, 1/2],
        ComplexF64[1/2, -0.5im, 0.5im, 1/2], ComplexF64[1/2, -0.5im, -0.5im, -1/2]]
    td = TrainningData([probability_channel(qc, s) for s in state],state)

    optnet,p_pos = probability_tn_channel(qc, ComplexF64[1, 0, 0, 0])
    model= [fill(0.01,15)]
    res = error_learning(model,td,optnet,p_pos;iter = 10000)
end

@testset "error_learning" begin
    p2 = fill(0.1,3)
    unitary_channel1 = depolarizing_channel(1, [1 - sum(p2), p2...])

    Random.seed!(1234)
    umat = rand_unitary(4)
    qc = chain(put(2, (1, 2) => matblock(umat)), put(2, (1) => unitary_channel1), put(2, (2) => unitary_channel1))

    state = [
        ComplexF64[1, 0, 0, 0], ComplexF64[0, 1, 0, 0],
        ComplexF64[0, 0, 1, 0], ComplexF64[0, 0, 0, 1],
        ComplexF64[1/2, 1/2, 1/2, 1/2], ComplexF64[1/2, -1/2, 1/2, -1/2],
        ComplexF64[1/2, 1/2, -1/2, -1/2], ComplexF64[1/2, -1/2, -1/2, 1/2],
        ComplexF64[1/2, 0.5im, 0.5im, -1/2], ComplexF64[1/2, 0.5im, -0.5im, 1/2],
        ComplexF64[1/2, -0.5im, 0.5im, 1/2], ComplexF64[1/2, -0.5im, -0.5im, -1/2]]
    td = TrainningData([probability_channel(qc, s) for s in state],state)

    optnet,p_pos = probability_tn_channel(qc, ComplexF64[1, 0, 0, 0])
    model= [fill(0.01,3),fill(0.01,3)]
    res = error_learning(model,td,optnet,p_pos;iter = 10000)
    @show res
end
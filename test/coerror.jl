using Test
using TensorQEC
using TensorQEC.Yao
using TensorQEC.Yao.YaoBlocks.Optimise
using TensorQEC.LinearAlgebra
@testset "coherent_error_unitary" begin
    u = rand_unitary(16)
    inf = Float64[]
    u2 = coherent_error_unitary(u, 1e-2;cache = inf)
    @test u2'*u2 ≈ I
    @test inf[1] < 1e-2
    @test inf[1] > 0
    @show coherent_error_unitary(mat(X), 1e-2;cache = inf) 
end

@testset "replace_gates" begin
    qc = chain(3, put(3, 1 => X), put(3, 2 => Y), put(3, 3 => Z), control(3, 2, 3 => X), control(3, (1,3), 2 => Z), control(3, 1, 3 => Z))
    qcf, srs = ein_circ(qc, ConnectMap([], [], 3))
    qc2 = replace_block(x->toput(x), qcf)
    @test qc2.blocks[5].content == CCZ
end

@testset "error_quantum_circuit" begin
    qc = chain(3, put(3, 1 => X), put(3, 2 => H), put(3, 3 => Z), control(3, 2, 3 => X), control(3, (1,3), 2 => Z), control(3, 1, 3 => Z))
    qc, srs = ein_circ(qc, ConnectMap([], [], 3))
    qce,maxe = error_quantum_circuit(qc ,1e-5) 
    @show qce
    @test isapprox(1-abs(tr(mat(qc)'*mat(qce)))/2^6,0; atol=1e-5)
end

using Test
using TensorQEC
using TensorQEC.Yao
using TensorQEC.OMEinsum
using TensorQEC.LinearAlgebra
using Random
using TensorQEC.Yao.YaoBlocks.Optimise

@testset "ComplexConj" begin
    ccj = ComplexConj(ConstGate.X)
    @test ccj isa ComplexConj
    @test apply!(product_state(bit"1"), ccj) ≈ apply!(product_state(bit"1"), ConstGate.X)
    @test conj(ccj) == ConstGate.X
end

@testset "SymbolRecorder" begin
    reg = rand_state(1)
    sr = SymbolRecorder()
    @test apply!(reg, sr) == reg

    srs = [SymbolRecorder() for i in 1:3]
    qc = chain(3,put(3, 1 => srs[1]), put(3, 2 => X), put(3, 3 => Y), put(3, 1 => Z), put(3, 2 => srs[2]), put(3, 3 => srs[3]))
    ein_code = yao2einsum(qc)
    @test srs[1].symbol == 1
    @test srs[2].symbol == 4
    @test srs[3].symbol == 5
end

@testset "IdentityRecorder" begin
    reg = rand_state(1)
    sr = IdentityRecorder()
    @test apply!(reg, sr) == reg

    srs = [IdentityRecorder() for i in 1:3]
    qc = chain(3,put(3, 1 => srs[1]), put(3, 2 => X), put(3, 3 => Y), put(3, 1 => Z), put(3, 2 => srs[2]), put(3, 3 => srs[3]))
    tn = yao2einsum(qc)
    @test srs[1].symbol == 1
    @test srs[2].symbol == 5
    @test srs[3].symbol == 6
end

@testset "ein_circ" begin
    u1 = mat(X)
    toyqc = chain(2, put(2,1 => GeneralMatrixBlock(u1; nlevel=2, tag="X")))
    qcf, srs = ein_circ(toyqc, QCInfo([1],2))

    @test length(srs) == 6
    ein_code = yao2einsum(qcf;optimizer=nothing)
end

@testset "qc2enisum" begin
    st = [PauliString((1,4,4)),PauliString((4,1,4))]
    qcen, data_qubits, code = encode_stabilizers(st)
    qc = chain(qcen, put(3, 1=>X),put(3,2=>X),put(3,3=>X), qcen', put(3, 3=>X))
    qc_info = QCInfo(data_qubits, 3)
    qcf, srs = ein_circ(qc, qc_info)
    tn,input_inds,outpu_inds = qc2enisum(qcf, srs, qc_info)
    optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())
    matr = contract(optnet)
    @test size(matr) == (2,2,2,2)
    @test matr[1,1,1,1] == 1
    @test matr[2,2,2,2] == 1
    @test matr[1,2,1,2] == 1
    @test matr[2,1,2,1] == 1
    tn2,input_inds2,outpu_inds2= simulation_tensornetwork(qc, qc_info)
    optnet2 = optimize_code(tn2, TreeSA(), OMEinsum.MergeVectors())
    mat2 = contract(optnet2) 
    @test mat2 ≈ matr
    @test input_inds2 == input_inds
    @test outpu_inds2 == outpu_inds
end

function get_kraus(u::AbstractMatrix{T}, ndata::Int) where T
    return [u[i*2^ndata+1:(i+1)*2^ndata,1:2^ndata] for i in 0:2^(Yao.log2i(size(u, 1))-ndata)-1]
end

@testset "fidelity_tensornetwork" begin
    Random.seed!(214)
    u1 = rand_unitary(2)
    u2 = rand_unitary(4)
    toyqc = chain(2, put(2,1 => GeneralMatrixBlock(u1; nlevel=2, tag="U1")),put(2, (1,2) => GeneralMatrixBlock(u2; nlevel=2, tag="U2")))
    qc_info = QCInfo([1],[2],2)
    tn = fidelity_tensornetwork(toyqc, qc_info)
    optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())
    # @show contract(optnet)[1]

    u = kron(u1,u1')
    kraus = get_kraus(u2, 1)
    uapp = mapreduce(x -> kron(x,x'), +, kraus)
    @test real(tr(u * uapp))/4 ≈ real(contract(optnet)[1])
end

@testset "get_kraus" begin
    Random.seed!(214)
    u = rand_unitary(4)
    ub = matblock(u)
    reg1 = rand_state(1) 
    psi = join(zero_state(1),reg1)
    psi_f = apply(psi,ub)
    rho2 = partial_tr(density_matrix(psi_f),2)
    @test partial_tr(density_matrix(psi),2).state ≈ density_matrix(reg1).state

    kraus = get_kraus(u,1)
    rho_f = zeros(ComplexF64,2,2)
    for idx in 1:2
       rho_f += kraus[idx] * partial_tr(density_matrix(psi),2).state * kraus[idx]' 
    end
    @test rho_f ≈ rho2.state

    u = rand_unitary(16)
    ub = matblock(u)
    reg1 = rand_state(2) 
    psi = join(zero_state(2),reg1)
    psi_f = apply(psi,ub)
    rho2 = partial_tr(density_matrix(psi_f),[3,4])
    @test partial_tr(density_matrix(psi),[3,4]).state == density_matrix(reg1).state

    kraus = get_kraus(u, 2)
    rho_f = zeros(ComplexF64,4,4)
    for idx in 1:4
       rho_f += kraus[idx] * partial_tr(density_matrix(psi), [3,4]).state * kraus[idx]' 
    end
    @test rho_f ≈ rho2.state
end

@testset "coherent_error_unitary" begin
    u = rand_unitary(16)
    inf = Float64[]
    u2 = coherent_error_unitary(u, 1e-4;cache = inf)
    @test u2'*u2 ≈ I
    @test inf[1] < 1e-2
    @test inf[1] > 0
    @show coherent_error_unitary(mat(X), 1e-2;cache = inf) 
end

@testset "replace_gates" begin
    qc = chain(3, put(3, 1 => X), put(3, 2 => Y), put(3, 3 => Z), control(3, 2, 3 => X), control(3, (1,3), 2 => Z), control(3, 1, 3 => Z))
    qcf, srs = ein_circ(qc, QCInfo([], [], 3))
    qc2 = replace_block(x->toput(x), qcf)
    @test qc2.blocks[5].content == CCZ
end

@testset "error_quantum_circuit" begin
    qc = chain(3, put(3, 1 => X), put(3, 2 => H), put(3, 3 => Z), control(3, 2, 3 => X), control(3, (1,3), 2 => Z), control(3, 1, 3 => Z))
    qc, srs = ein_circ(qc, QCInfo([], [], 3))
    qce = error_quantum_circuit(qc ,1e-7) 
    @test isapprox(1-abs(tr(mat(qc)'*mat(qce)))/2^6,0; atol=1e-5)
end

@testset "error_quantum_circuit_pair_replace" begin
    qc = chain(3, put(3, 1 => X), put(3, 2 => H), put(3, 3 => Z), control(3, 2, 3 => X), control(3, (1,3), 2 => Z), control(3, 1, 3 => Z))
    qc, srs = ein_circ(qc, QCInfo([], [], 3))
    qce,vec = error_quantum_circuit_pair_replace(qc ,1e-7) 
    @test isapprox(1-abs(tr(mat(qc)'*mat(qce)))/2^6,0; atol=1e-5)
end

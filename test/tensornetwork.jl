using Test, TensorQEC
using TensorQEC.Yao, TensorQEC.TensorInference
using TensorQEC.TensorInference.OMEinsum

@testset "matrix2factor" begin
	g = cnot(2, 1, 2)
	f = TensorQEC.matrix2factor(g, [1, 2], [3, 4])
	@test f.vars == (3, 4, 1, 2)
end

@testset "projector" begin
	@test TensorQEC.projector(Bool, Bool[1,0,0,1]) == Bool[1 0 0 0; 0 0 0 1]
	@test TensorQEC.projector(Bool, Bool[0,1,1,0]) == Bool[0 1 0 0; 0 0 1 0]
end

@testset "convert to put" begin
    for g in [
        control(5, 3, 4=>X),
        control(5, 4, 2=>X),
        put(5, 2=>X),
        control(5, -3, 2=>X)
        ]
        pg = TensorQEC.convert_to_put(g)
        @test pg isa PutBlock
        @test mat(pg) ≈ mat(g)
    end
end

@testset "tensor network mapping - 1 gate" begin
	# create a circuit and convert it to the pauli basis
	yaoqc = chain(cnot(2, 1, 2), put(1=>T), put(2=>H), put(2=>rand_unitary(2)), cnot(2, 1, 2))
	yaopauli = pauli_mapping(mat(ComplexF64, yaoqc))

	# tensor network mapping of a quantum circuit
	for ci in CartesianIndices((fill(4, 2)...,))
		ps = [Yao.BitBasis._onehot(Float64, 4, ci.I[i]) for i in 1:2]
		tn = TensorQEC.simple_circuit2tensornetworks(yaoqc, ps)
        @test length(tn.vars) == 6
		p1 = probability(tn)
		p2 = yaopauli[ci.I..., :, :]
		@test p1 ≈ p2
	end
end


# |ψ> ---.--X------- p
#        |
# |0> ---X------.--- p
#               |
# |0> ----------X--- p
@testset "tensor network mapping" begin
	# create a circuit and convert it to the pauli basis
	yaoqc = chain(cnot(3, 1, 2), put(3, 1=>X), cnot(3, 3, 2))
	yaopauli = pauli_mapping(mat(ComplexF64, yaoqc))

	# tensor network mapping of a quantum circuit
	for ci in CartesianIndices((fill(4, 3)...,))
		ps = [Yao.BitBasis._onehot(Float64, 4, ci.I[i]) for i in 1:3]
		tn = TensorQEC.simple_circuit2tensornetworks(yaoqc, ps)
		p1 = probability(tn)
		p2 = yaopauli[ci.I..., :, :, :]
		@test p1 ≈ p2
	end
end


@testset "expect" begin
    # target circuit
    qc = chain(cnot(3, 1, 2), put(3, 1=>X), cnot(3, 3, 2))
    cl = clifford_network(qc)
	# step 1: pauli decomposition of rho0
	reg = rand_state(6)
	dm = density_matrix(reg, 1:3)
	sp = densitymatrix2sumofpaulis(dm)
    ps1, ps2 = sp.items[6].second, sp.items[2].second
    res1 = expect(ps1, cl, ps2)[]
    dm2 = DensityMatrix(Matrix{ComplexF64}(ps2))
    dm2f = apply(dm2, qc)
    res2 = Yao.expect(ps1, dm2f)
    @show res1, res2
    @test res1 ≈ res2
end
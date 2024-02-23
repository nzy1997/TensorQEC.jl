using Test, TensorQEC
using TensorQEC.Yao, TensorQEC.TensorInference
using TensorQEC.TensorInference.OMEinsum

@testset "matrix2factor" begin
	g = cnot(2, 1, 2)
	f = TensorQEC.matrix2factor(g, [1, 2], [3, 4])
	@test f.vars == (3, 4, 1, 2)
end

@testset "projector" begin
	@test TensorQEC.projector(Bool[1,0,0,1]) == Bool[1 0 0 0; 0 0 0 1]
	@test TensorQEC.projector(Bool[0,1,1,0]) == Bool[0 1 0 0; 0 0 1 0]
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
		tn = circuit2tensornetworks(yaoqc, ps)
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

@testset "tensor network mapping" begin
	# create a circuit and convert it to the pauli basis
	yaoqc = chain(cnot(3, 1, 2), put(3, 1=>X), cnot(3, 3, 2))
	yaopauli = pauli_mapping(mat(ComplexF64, yaoqc))

	# tensor network mapping of a quantum circuit
    ps = [Yao.BitBasis._onehot(Float64, 4, b) for b in [1,3,2,1]]
    extra = Dict([2=>UNITY4, 1=>PXY, 3=>PIZ_UNITY2])
    tn = circuit2tensornetworks(yaoqc, ps, extra)
    p1 = probability(tn)

    function tovector(::Type{T}, target) where T
        if target == UNITY4
            return T[1,1,1,1]
        elseif target == PXY_UNITY2
            return T[0,1,1,0]
        elseif target == PIZ_UNITY2
            return T[1,0,0,1]
        elseif target == PIZ
            return TensorQEC.projector(Bool[1,0,0,1])
        else
            return TensorQEC.projector(Bool[0,1,1,0])
        end
    end

    @show p1
    #p2 = yaopauli[ci.I..., :, :, :]
    #@test p1 ≈ p2
end

@testset "expect" begin
    # target circuit
    qc = chain(cnot(3, 1, 2), put(3, 1=>X), cnot(3, 3, 2))
    cl = clifford_network(qc)
	# step 1: pauli decomposition of rho0
	reg = rand_state(6)
	dm = density_matrix(reg, 1:3)
	sp = densitymatrix2sumofpaulis(dm)
    ps1, ps2 = sp.items[1].second, sp.items[2].second
    res1 = expect(ps1, cl, ps2)
    dm2 = apply(DensityMatrix(mat(ps2)), qc)
    res2 = Yao.expect(ps1, dm2)
end
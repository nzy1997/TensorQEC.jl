using TensorQEC.Yao, TensorQEC
using Test

@testset "paulistring" begin
    # constructor
    g = PauliString((1, 3, 4))
    @test g == PauliString(1, 3, 4)
    @test occupied_locs(g) == (2, 3)

    # properties (faster implementation)
    @test ishermitian(g) == ishermitian(mat(g)) == true
    @test isreflexive(g) == isreflexive(mat(g)) == true
    @test isunitary(g) == isunitary(mat(g)) == true

    # iterable iterfaces
    @test length(g) == 3
    @test g[1] == I2
    @test g[end] == Z
    @test [g...] == [I2, Y, Z] == [g[i] for i in eachindex(g)]
    @test collect(g) == [I2, Y, Z]

    # apply and mat
    r = rand_state(3)
    @test apply!(copy(r), g) ≈ apply!(copy(r), kron(I2, Y, Z))
	reg = rand_state(4)
	g = PauliString((1, 2, 3, 4))
	@test nqubits(g) == 4
	r1 = apply(reg, g)
	r2 = apply(reg, kron(I2, X, Y, Z))
	@test r1 ≈ r2
	@test mat(g) ≈ mat(kron(I2, X, Y, Z))
    g = PauliString(X, Y, I2, Z)
    @test mat(g) ≈ mat(kron(X, Y, I2, Z))
    @test cache_key(g) == hash(g.ids)

    # subblocks
    subblocks(g) == (X, Y, I2, Z)
    @test chsubblocks(g, (Z, Y, I2, Z)) == PauliString(Z, Y, I2, Z)

    # printing
    g = PauliString((1, 3, 4))
    print(g)
    print(chain(g, put(3, 3=>PauliString(2))))

    # others
    @test Yao.YaoBlocks.Optimise.to_basictypes(g) == chain([put(3, 2=>Y), put(3, 3=>Z)])
end

@testset "Sum of Paulis" begin
    reg = rand_state(6)
    p1 = PauliString((1, 2, 3))
    p2 = PauliString((4, 4, 3))
    sp = SumOfPaulis([0.6=>p1, 0.8=>p2])
    @test mat(sp) ≈ 0.6 * mat(p1) + 0.8 * mat(p2)

    p3 = chsubblocks(p1, (X, Y, Z))
    sp2 = chsubblocks(sp, [p3, p1])
    @test mat(sp2) ≈ 0.6 * mat(p3) + 0.8 * mat(p1)
    reg = rand_state(3)
    @test apply(reg, sp2) ≈ 0.6 * apply(reg, p3) + 0.8 * apply(reg, p1)

    # convert to sum of paulis
    reg = rand_state(6)
    dm = density_matrix(reg, 1:3)
    sp = densitymatrix2sumofpaulis(dm)
    @test dm.state ≈ mat(sp)
end
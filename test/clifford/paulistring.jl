using TensorQEC.Yao, TensorQEC
using Test

@testset "Pauli" begin
    I, X, Y, Z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    for i in 0:3
        @test mat(Pauli(i)) ≈ mat(yaoblock(Pauli(i)))
        for j in 0:3
            @info "Testing $(Pauli(i)) * $(Pauli(j)) -> $(Pauli(i) * Pauli(j))"
            @test mat(Pauli(i) * Pauli(j)) ≈ mat(Pauli(i)) * mat(Pauli(j))
        end
    end
    @test_throws AssertionError Pauli(4)
end

@testset "paulistring" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    # constructor
    g = PauliString(i, y, z)
    @test findfirst(x -> x == y, g) == 2
    @test g == PauliString((Pauli(0), Pauli(2), Pauli(3)))
    @test occupied_locs(yaoblock(g)) == (2, 3)
    @test g * g == SumOfPaulis([1=>PauliString(i, i, i)])
    @test g * g ≈ SumOfPaulis([1=>PauliString(i, i, i)])

    # properties (faster implementation)
    @test ishermitian(g) == ishermitian(mat(g)) == true
    @test isreflexive(g) == isreflexive(mat(g)) == true
    @test isunitary(g) == isunitary(mat(g)) == true

    # iterable iterfaces
    @test length(g) == 3
    @test yaoblock(g[1]) == I2
    @test yaoblock(g[end]) == Z
    @test [g...] == [g[i] for i in eachindex(g)]
    @test collect(g) == [i, y, z]
    @test collect(yaoblock.(g)) == [I2, Y, Z]

    # apply and mat
    r = rand_state(3)
    @test apply!(copy(r), yaoblock(g)) ≈ apply!(copy(r), kron(I2, Y, Z))
	reg = rand_state(4)
	g = PauliString(i, x, y, z)
	@test length(g) == 4
	r1 = apply(reg, yaoblock(g))
	r2 = apply(reg, kron(I2, X, Y, Z))
	@test r1 ≈ r2
	@test mat(g) ≈ mat(kron(I2, X, Y, Z))
    g = PauliString(x, y, i, z)
    @test mat(yaoblock(g)) ≈ mat(kron(X, Y, I2, Z))

    # printing
    g = PauliString(i, y, z)
    print(g)
    print(chain(g, put(3, 3=>yaoblock(PauliString(y)))))

    # iscommute and isanticommute
    g = PauliString(x, y, z, i)
    @test iscommute(g, PauliString(i, z, x, i))
    @test isanticommute(g, PauliString(i, i, x, i))
end

@testset "Sum of Paulis" begin
    reg = rand_state(6)
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    p1 = PauliString(i, x, y)
    p2 = PauliString(z, z, y)
    sp = SumOfPaulis([0.6=>p1, 0.8=>p2])
    @test mat(yaoblock(sp)) ≈ 0.6 * mat(yaoblock(p1)) + 0.8 * mat(yaoblock(p2))

    reg = rand_state(3)
    @test apply(reg, yaoblock(sp)) ≈ 0.6 * apply(reg, yaoblock(p1)) + 0.8 * apply(reg, yaoblock(p2))

    # convert to sum of paulis
    reg = rand_state(6)
    dm = density_matrix(reg, 1:3)
    sp = SumOfPaulis(dm)
    @test dm.state ≈ mat(yaoblock(sp))
end

@testset "pauli group" begin
    i2, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    g = PauliGroupElement(3, PauliString(x, y, z))
    h = PauliGroupElement(1, PauliString(y, i2, z))
    i = PauliGroupElement(2, PauliString(z, y, x))
    @test g * h == PauliGroupElement(1, PauliString(z, y, i2))
    @test g * g == PauliGroupElement(2, PauliString(i2, i2, i2))
    @test h * g == PauliGroupElement(3, PauliString(z, y, i2))
    @test isunitary(g)
    @test isunitary(h)
    @test isunitary(i)
    @test !ishermitian(g)
    @test !ishermitian(h)
    @test ishermitian(i)
    @test !isreflexive(g)
    @test !isreflexive(h)
    @test isreflexive(i)
    @test !iscommute(g, h)
    @test iscommute(g, g)
    @test iscommute(g, i)
    @test !iscommute(h, g)
    @test iscommute(h, i)
    @test isanticommute(h, g)
end

@testset "macro" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    @test P"IXYZ" == PauliString(i, x, y, z)
    @test P"IIII" == PauliString(i, i, i, i)
    @test P"XYZI" == PauliString(x, y, z, i)
    @test_throws ErrorException P"IXYZC"
end

@testset "algebra" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    ops = [x, PauliString(x, z, x), SumOfPaulis([0.6=>PauliString(x, z, x), 0.8=>PauliString(y, z, y)])]
    for a in ops
        @test mat(-a) ≈ -mat(a)
        @test mat(2 * a) ≈ 2 * mat(a)
        @test mat(a * 2) ≈ mat(a) * 2
        @test mat(a / 2) ≈ mat(a) / 2
        for b in ops
            if (a isa Pauli && b isa Pauli) || (!(a isa Pauli) && !(b isa Pauli))
                @test mat(a + b) ≈ mat(a) + mat(b)
                @test mat(a - b) ≈ mat(a) - mat(b)
                @test mat(a * b) ≈ mat(a) * mat(b)
            end
        end
    end
end
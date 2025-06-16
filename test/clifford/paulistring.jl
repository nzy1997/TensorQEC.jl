using TensorQEC.Yao, TensorQEC
using Test

@testset "Pauli" begin
    I, X, Y, Z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    @test string(I) == "I"
    @test string(X) == "X"
    @test string(Y) == "Y"
    @test string(Z) == "Z"
    @test TensorQEC.coeff_type(Pauli) == Int
    for i in 0:3
        @test isunitary(Pauli(i))
        @test ishermitian(Pauli(i))
        @test isreflexive(Pauli(i))
        @test mat(Pauli(i)) ≈ mat(yaoblock(Pauli(i)))
        for j in 0:3
            @info "Testing $(Pauli(i)) * $(Pauli(j)) -> $(Pauli(i) * Pauli(j))"
            @test mat(Pauli(i) * Pauli(j)) ≈ mat(Pauli(i)) * mat(Pauli(j))
        end
    end
    @test Pauli(Yao.I2) == Pauli(0)
    @test Pauli(Yao.X) == Pauli(1)
    @test Pauli(Yao.Y) == Pauli(2)
    @test Pauli(Yao.Z) == Pauli(3)
    @test_throws AssertionError Pauli(4)
    @test copy(Pauli(1)) == Pauli(1)
end

@testset "paulistring" begin
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    # constructor
    g = PauliString(i, y, z)
    @test TensorQEC.coeff_type(typeof(g)) == Int
    @test string(g) == "IYZ"
    @test isunitary(g)
    @test ishermitian(g)
    @test isreflexive(g)
    @test_throws AssertionError PauliString(4, (1, 2, 4)=>Pauli(1), 2=>Pauli(1))
    @test_throws AssertionError PauliString(4, (1, 2, 2)=>Pauli(1), 4=>Pauli(1))
    @test PauliString(Yao.I2, Yao.X) == PauliString(i, x)
    @test findfirst(x -> x == y, g) == 2
    @test g == PauliString((Pauli(0), Pauli(2), Pauli(3)))
    @test occupied_locs(yaoblock(g)) == (2, 3)
    @test g * g isa PauliGroupElement
    @test g * g == PauliGroupElement(0, PauliString(i, i, i))
    @test copy(g) == g
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
    @test string(sp) == "0.6 * IXY + 0.8 * ZZY"
    @test string(zero(sp)) == "𝟘"
    @test one(sp) == SumOfPaulis([1.0=>P"III"])
    @test one(sp) * sp == sp
    @test zero(sp) == SumOfPaulis(Pair{Float64, PauliString{3}}[])
    @test zero(sp) + sp == sp
    @test zero(sp) * sp == zero(sp)
    @test sp == SumOfPaulis([0.6=>p1]) + SumOfPaulis([0.8=>p2])
    @test sp ≈ SumOfPaulis([0.6=>p1, 0.8=>p2])
    @test mat(yaoblock(sp)) ≈ 0.6 * mat(yaoblock(p1)) + 0.8 * mat(yaoblock(p2))
    @test copy(sp) == sp

    reg = rand_state(3)
    @test apply(reg, yaoblock(sp)) ≈ 0.6 * apply(reg, yaoblock(p1)) + 0.8 * apply(reg, yaoblock(p2))

    # convert to sum of paulis
    reg = rand_state(6)
    dm = density_matrix(reg, 1:3)
    sp = pauli_decomposition(dm)
    @test dm.state ≈ mat(yaoblock(sp))

    # commute and anticommute
    @test iscommute(sp, sp)
    sp = 0.6 * P"IXY"
    sp2 = P"IXZ"
    @test isanticommute(sp, sp2)
end

@testset "pauli group" begin
    i2, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
    g = PauliGroupElement(3, PauliString(x, y, z))
    h = PauliGroupElement(1, PauliString(y, i2, z))
    i = PauliGroupElement(2, PauliString(z, y, x))
    @test length(g) == 3
    @test TensorQEC.coeff_type(typeof(g)) == Complex{Int}
    @test PauliGroupElement(P"XXX") == PauliGroupElement(0, PauliString(3, (1,2,3)=>x))
    @test one(g) == PauliGroupElement(0, PauliString(3, (1,2,3)=>i2))
    @test one(g) * g == g
    @test string(g) == "-i * XYZ"
    @test g * h == PauliGroupElement(1, PauliString(z, y, i2))
    @test g * g == PauliGroupElement(2, PauliString(i2, i2, i2))
    @test h * g == PauliGroupElement(3, PauliString(z, y, i2))
    @test copy(g) == g
    @test copy(h) == h
    @test copy(i) == i
    for g in [g, h, i]
        @test isunitary(g) == isunitary(yaoblock(g))
        @test ishermitian(g) == ishermitian(yaoblock(g))
        @test isreflexive(g) == isreflexive(yaoblock(g))
    end
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
    @test P"XX" > P"IX"
    @test P"XI" < P"IX"   # Note: big endian format!
    @test !(P"XI" < P"XI")
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
            @info "Testing $(a) and $(b)"
            if (a isa Pauli && b isa Pauli) || (!(a isa Pauli) && !(b isa Pauli))
                @test mat(a + b) ≈ mat(a) + mat(b)
                @test mat(a - b) ≈ mat(a) - mat(b)
                @test mat(a * b) ≈ mat(a) * mat(b)
            end
        end
    end
    p = (0.5 * P"XX" + 0.2 * P"XY")
    @test p^0 ≈ one(p)
    @test p^2 ≈ 0.29 * P"II"
    @test x^0 == PauliGroupElement(0, P"I")
    @test x^1 == PauliGroupElement(0, P"X")
    @test x^2 == PauliGroupElement(0, P"I")
end

@testset "promote" begin
    @test promote(Pauli(1), P"Z") == (P"X", P"Z")
    @test promote(Pauli(1), PauliGroupElement(P"Z")) == (PauliGroupElement(0, P"X"), PauliGroupElement(0, P"Z"))
    @test promote(Pauli(1), SumOfPaulis([0.6=>P"Z"])) == (SumOfPaulis([1.0=>P"X"]), SumOfPaulis([0.6=>P"Z"]))
    @test promote(P"ZX", PauliGroupElement(1, P"ZX")) == (PauliGroupElement(0, P"ZX"), PauliGroupElement(1, P"ZX"))
    @test promote(P"ZX", SumOfPaulis([0.6=>P"ZZ"])) == (SumOfPaulis([1.0=>P"ZX"]), SumOfPaulis([0.6=>P"ZZ"]))
    @test promote(PauliGroupElement(1, P"ZX"), SumOfPaulis([0.6=>P"ZZ"])) == (SumOfPaulis([1.0im=>P"ZX"]), SumOfPaulis([0.6=>P"ZZ"]))
end
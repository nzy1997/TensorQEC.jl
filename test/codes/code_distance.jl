using Test
using TensorQEC
using TensorQEC: row_echelon_form, null_space, logical_operator, same_qubit_order, verify_logical_action
using Random

@testset "classical_code_distance" begin
    H = [0 0 0 1 1 1 1;0 1 1 0 0 1 1; 1 0 1 0 1 0 1]
    @test code_distance(H) == 3

    H2 = (H .== 1)
    @test code_distance(H2) == 3

    H3 = Mod2.(H2)
    @test code_distance(H3) == 3
end

@testset "row_echelon_form" begin
    H = Bool[ 1  1  0  1  0  0  1  0  1  0;
    1  1  0  1  1  0  0  1  1  1;
    1  0  0  1  1  1  1  1  1  1;
    0  1  0  0  1  0  0  0  0  0;
    0  0  1  1  1  1  1  1  0  1;
    1  1  0  1  1  0  0  1  1  0;
    0  0  1  0  1  0  0  0  0  1;
    1  0  0  1  0  0  1  0  1  0;
    1  0  1  0  0  0  0  1  0  1;
    0  1  0  1  1  1  0  1  1  0]
    @test row_echelon_form(H) ==   Bool[ 1  0  0  0  0  0  0  0  1  0;
    0  1  0  0  0  0  0  0  0  0;
    0  0  1  0  0  0  0  0  0  0;
    0  0  0  1  0  0  0  0  1  0;
    0  0  0  0  1  0  0  0  0  0;
    0  0  0  0  0  1  0  0  1  0;
    0  0  0  0  0  0  1  0  1  0;
    0  0  0  0  0  0  0  1  1  0;
    0  0  0  0  0  0  0  0  0  1;
    0  0  0  0  0  0  0  0  0  0]
end

@testset "null_space" begin
    H = Bool[0 0 0 1 1 1 1;0 1 1 0 0 1 1; 1 0 1 0 1 0 1]
    kerH = null_space(H)
    @test size(kerH) == (4,7)
    for i in 1:4
        @test Mod2.(H) * Mod2.(kerH[i,:]) == zeros(Mod2, 3)
    end

    local x
    pivots = [x for i in 1:4 if begin x = findfirst(!iszero, kerH[i,:]); !isnothing(x) end]
    rank = length(pivots)
    @test rank == 4
end
@testset "logical_operator" begin
    tannerxz = CSSTannerGraph(SurfaceCode(3, 3))
    lx,lz = logical_operator(tannerxz)
    @test lx == Mod2[0 0 0 0 0 0 1 1 1]
    @test lz == Mod2[1 0 0 0 1 0 0 0 1]
end

@testset "verify_logical_action" begin
    st = stabilizers(SteaneCode())
    tanner = CSSTannerGraph(st)
    lx, lz = logical_operator(tanner)
    op = PauliString(7, findall(i -> i.x, lx[1, :]) => Pauli(1))

    result = verify_logical_action(st, lx, lz, op)

    @test result.preserves_stabilizers
    @test all(result.commutes_with_stabilizers)
    @test result.commutes_with_lx == [true]
    @test result.commutes_with_lz == [false]
end

@testset "verify_logical_action validation" begin
    st = stabilizers(SteaneCode())
    tanner = CSSTannerGraph(st)
    lx, lz = logical_operator(tanner)
    op = PauliString(7, 1 => Pauli(1))

    bad_lz = zeros(Mod2, size(lz, 1), size(lz, 2) - 1)

    @test_throws AssertionError verify_logical_action(st, lx, bad_lz, op)
end

@testset "verify_logical_action detects broken stabilizers" begin
    st = stabilizers(SteaneCode())
    tanner = CSSTannerGraph(st)
    lx, lz = logical_operator(tanner)
    op = PauliString(7, 1 => Pauli(1))

    result = verify_logical_action(st, lx, lz, op)

    @test !result.preserves_stabilizers
    @test any(.!result.commutes_with_stabilizers)
end

@testset "verify_logical_action supports multiple logical qubits" begin
    st = stabilizers(ToricCode(3, 3))
    tanner = CSSTannerGraph(ToricCode(3, 3))
    lx, lz = logical_operator(tanner)

    op1 = PauliString(size(lx, 2), findall(i -> i.x, lx[1, :]) => Pauli(1))
    result1 = verify_logical_action(st, lx, lz, op1)
    @test result1.preserves_stabilizers
    @test result1.commutes_with_lx == [true, true]
    @test result1.commutes_with_lz == [false, true]

    op2 = PauliString(size(lx, 2), findall(i -> i.x, lx[2, :]) => Pauli(1))
    result2 = verify_logical_action(st, lx, lz, op2)
    @test result2.preserves_stabilizers
    @test result2.commutes_with_lx == [true, true]
    @test result2.commutes_with_lz == [true, false]
end

@testset "code_distance" begin
    tannerxz = CSSTannerGraph(SurfaceCode(3, 3))
    lx,lz = logical_operator(tannerxz)
    @test code_distance(Int.(tannerxz.stgz.H),Int.(lz)) == 3
    @test code_distance(Int.(tannerxz.stgx.H),Int.(lx)) == 3

    tannerxz = CSSTannerGraph(SurfaceCode(5, 5))
    @test code_distance(tannerxz) == 5

    tannerxz = CSSTannerGraph(Code422())
    @test code_distance(tannerxz) == 2
end

@testset "same_qubit_order" begin
    tannerxz = CSSTannerGraph(ToricCode(3, 3))
    lx,lz = logical_operator(tannerxz)
    lz_new = zeros(Mod2, size(lx))
    lz_new[1,:] = lz[2,:]
    lz_new[2,:] = lz[1,:]

    lx,lz = same_qubit_order(lx,lz_new)
    @test sum(lx[1,:].*lz[1,:]).x
    @test sum(lx[2,:].*lz[2,:]).x
end

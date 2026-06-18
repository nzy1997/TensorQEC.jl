using Test
using TensorQEC
using TensorQEC: row_echelon_form, null_space, logical_operator, same_qubit_order, verify_logical_action, logical_pauli_coordinates
using TensorQEC.Yao
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

@testset "logical_pauli_coordinates" begin
    st = stabilizers(SteaneCode())
    tanner = CSSTannerGraph(st)
    lx, lz = logical_operator(tanner)
    opx = PauliString(7, findall(i -> i.x, lx[1, :]) => Pauli(1))
    opz = PauliString(7, findall(i -> i.x, lz[1, :]) => Pauli(3))

    xcoords = logical_pauli_coordinates(st, lx, lz, opx)
    @test xcoords.preserves_stabilizers
    @test xcoords.x_bits == [true]
    @test xcoords.z_bits == [false]

    zcoords = logical_pauli_coordinates(st, lx, lz, opz)
    @test zcoords.preserves_stabilizers
    @test zcoords.x_bits == [false]
    @test zcoords.z_bits == [true]
end

@testset "logical_pauli_coordinates is invariant on stabilizer cosets" begin
    st = stabilizers(SteaneCode())
    tanner = CSSTannerGraph(st)
    lx, lz = logical_operator(tanner)
    op = PauliString(7, findall(i -> i.x, lx[1, :]) => Pauli(1))
    shifted = (op * st[1]).ps

    coords = logical_pauli_coordinates(st, lx, lz, op)
    shifted_coords = logical_pauli_coordinates(st, lx, lz, shifted)

    @test coords.preserves_stabilizers
    @test shifted_coords.preserves_stabilizers
    @test shifted_coords.x_bits == coords.x_bits
    @test shifted_coords.z_bits == coords.z_bits
end

@testset "logical_pauli_coordinates supports mixed logical operators" begin
    st = stabilizers(SteaneCode())
    tanner = CSSTannerGraph(st)
    lx, lz = logical_operator(tanner)
    opx = PauliString(7, findall(i -> i.x, lx[1, :]) => Pauli(1))
    opz = PauliString(7, findall(i -> i.x, lz[1, :]) => Pauli(3))
    opy = (opx * opz).ps

    coords = logical_pauli_coordinates(st, lx, lz, opy)

    @test coords.preserves_stabilizers
    @test coords.x_bits == [true]
    @test coords.z_bits == [true]
end

@testset "logical_pauli_coordinates validation" begin
    st = stabilizers(SteaneCode())
    tanner = CSSTannerGraph(st)
    lx, lz = logical_operator(tanner)
    op = PauliString(7, 1 => Pauli(1))

    bad_lz = zeros(Mod2, size(lz, 1), size(lz, 2) - 1)

    @test_throws AssertionError logical_pauli_coordinates(st, lx, bad_lz, op)
end

@testset "logical_pauli_coordinates supports multiple logical qubits" begin
    st = stabilizers(ToricCode(3, 3))
    tanner = CSSTannerGraph(st)
    lx, lz = logical_operator(tanner)
    op = PauliString(size(lx, 2), findall(i -> i.x, lx[2, :]) => Pauli(1))

    coords = logical_pauli_coordinates(st, lx, lz, op)

    @test coords.preserves_stabilizers
    @test coords.x_bits == [false, true]
    @test coords.z_bits == [false, false]
end

@testset "logical_pauli_coordinates detects broken stabilizers" begin
    st = stabilizers(SteaneCode())
    tanner = CSSTannerGraph(st)
    lx, lz = logical_operator(tanner)
    op = PauliString(7, 1 => Pauli(1))

    coords = logical_pauli_coordinates(st, lx, lz, op)

    @test !coords.preserves_stabilizers
    @test coords.x_bits isa Vector{Bool}
    @test coords.z_bits isa Vector{Bool}
end

@testset "logical_clifford_action identity" begin
    st = stabilizers(SteaneCode())
    tanner = CSSTannerGraph(st)
    lx, lz = logical_operator(tanner)
    identity_action(op) = PauliGroupElement(op)

    result = logical_clifford_action(st, lx, lz, identity_action)

    @test result.preserves_code
    @test all(coords -> coords.preserves_stabilizers && !any(coords.x_bits) && !any(coords.z_bits), result.stabilizer_images)
    @test result.x_images[1].x_bits == [true]
    @test result.x_images[1].z_bits == [false]
    @test result.z_images[1].x_bits == [false]
    @test result.z_images[1].z_bits == [true]
end

@testset "logical_clifford_action rejects stabilizer images with logical support" begin
    st = stabilizers(SteaneCode())
    tanner = CSSTannerGraph(st)
    lx, lz = logical_operator(tanner)
    logical_x = PauliString(7, findall(i -> i.x, lx[1, :]) => Pauli(1))
    bad_action(op) = op == st[1] ? PauliGroupElement(logical_x) : PauliGroupElement(op)

    result = logical_clifford_action(st, lx, lz, bad_action)

    @test !result.preserves_code
    @test result.stabilizer_images[1].preserves_stabilizers
    @test result.stabilizer_images[1].x_bits == [true]
    @test result.stabilizer_images[1].z_bits == [false]
end

@testset "logical_clifford_action rejects phased stabilizer images" begin
    st = stabilizers(SteaneCode())
    tanner = CSSTannerGraph(st)
    lx, lz = logical_operator(tanner)
    phased_action(op) = op == st[1] ? PauliGroupElement(2, st[1]) : PauliGroupElement(op)

    result = logical_clifford_action(st, lx, lz, phased_action)

    @test !result.preserves_code
    @test result.stabilizer_images[1].preserves_stabilizers
    @test !any(result.stabilizer_images[1].x_bits)
    @test !any(result.stabilizer_images[1].z_bits)
end

@testset "logical_clifford_action rejects degenerate logical images" begin
    st = stabilizers(SteaneCode())
    tanner = CSSTannerGraph(st)
    lx, lz = logical_operator(tanner)
    logical_x = PauliString(7, findall(i -> i.x, lx[1, :]) => Pauli(1))
    id7 = PauliString(ntuple(_ -> Pauli(0), Val(7)))
    degenerate_action(op) = op == logical_x ? PauliGroupElement(id7) : PauliGroupElement(op)

    result = logical_clifford_action(st, lx, lz, degenerate_action)

    @test !result.preserves_code
    @test result.x_images[1].preserves_stabilizers
    @test !any(result.x_images[1].x_bits)
    @test !any(result.x_images[1].z_bits)
end

@testset "logical_clifford_action detects toric transversal cnot" begin
    st = stabilizers(ToricCode(3, 3))
    tanner = CSSTannerGraph(st)
    lx, lz = logical_operator(tanner)
    zero_block = zeros(Mod2, size(lx))
    n = size(lx, 2)
    id_block = PauliString(ntuple(_ -> Pauli(0), n))

    st_all = vcat([vcat(st_i, id_block) for st_i in st], [vcat(id_block, st_i) for st_i in st])
    lx_all = vcat(hcat(lx, zero_block), hcat(zero_block, lx))
    lz_all = vcat(hcat(lz, zero_block), hcat(zero_block, lz))

    function pairwise_cnot_action(op)
        gate = CliffordGate(ConstGate.CNOT)
        pg = PauliGroupElement(op)
        for i in 1:n
            pg = gate(pg, (i, n + i))
        end
        return pg
    end

    function assert_logical_image(coords, x_bits, z_bits)
        @test coords.preserves_stabilizers
        @test coords.x_bits == x_bits
        @test coords.z_bits == z_bits
    end

    result = logical_clifford_action(st_all, lx_all, lz_all, pairwise_cnot_action)

    @test result.preserves_code
    @test all(coords -> coords.preserves_stabilizers && !any(coords.x_bits) && !any(coords.z_bits), result.stabilizer_images)

    assert_logical_image(result.x_images[1], [true, false, true, false], [false, false, false, false])
    assert_logical_image(result.x_images[2], [false, true, false, true], [false, false, false, false])
    assert_logical_image(result.x_images[3], [false, false, true, false], [false, false, false, false])
    assert_logical_image(result.x_images[4], [false, false, false, true], [false, false, false, false])

    assert_logical_image(result.z_images[1], [false, false, false, false], [true, false, false, false])
    assert_logical_image(result.z_images[2], [false, false, false, false], [false, true, false, false])
    assert_logical_image(result.z_images[3], [false, false, false, false], [true, false, true, false])
    assert_logical_image(result.z_images[4], [false, false, false, false], [false, true, false, true])
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

using Test
using TensorQEC
using TensorQEC: row_echelon_form,null_space,logical_operator, same_qubit_order
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
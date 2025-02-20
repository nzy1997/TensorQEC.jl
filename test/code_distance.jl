using Test
using TensorQEC
using TensorQEC: row_echelon_form,null_space,logical_oprator

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
@testset "logical_oprator" begin
    st = stabilizers(SurfaceCode(3, 3))
    tannerxz = CSSTannerGraph(st)
    lx,lz = logical_oprator(tannerxz)
    @test lx == Bool[0 0 0 0 0 0 1 1 1]
    @test lz == Bool[1 0 0 0 1 0 0 0 1]
end

@testset "code_distance" begin
    st = stabilizers(SurfaceCode(3, 3))
    tannerxz = CSSTannerGraph(st)
    lx,lz = logical_oprator(tannerxz)
    @test code_distance(Int.(tannerxz.stgz.H),Int.(lz)) == 3
    @test code_distance(Int.(tannerxz.stgx.H),Int.(lx)) == 3

    st = stabilizers(SurfaceCode(5, 5))
    tannerxz = CSSTannerGraph(st)
    @test code_distance(tannerxz) == 5

    st = stabilizers(Code422())
    tannerxz = CSSTannerGraph(st)
    @test code_distance(tannerxz) == 2
end
using Test
using TensorQEC

@testset "SimpleTannerGraph" begin
    sts = [[1, 2], [2, 3], [3, 4], [4, 1]]
    nq = 4
    tanner = SimpleTannerGraph(nq, sts)
    @show tanner
end
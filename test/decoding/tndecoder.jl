using Test
using TensorQEC

@testset "ldpc2tensor" begin
    sts = [[1, 2,3,4],[2,3,4,5]]
    nq = 5
    tanner = SimpleTannerGraph(nq, sts)

    tn = TensorQEC.ldpc2tensor(tanner,0.1,Mod2[1,0])
end
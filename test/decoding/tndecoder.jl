using Test
using TensorQEC
using TensorQEC.Yao
using TensorQEC.OMEinsum

@testset "stg2tensornetwork" begin
    sts = [[1, 2,3,4],[2,3,4,5]]
    nq = 5
    tanner = SimpleTannerGraph(nq, sts)

    code = DynamicEinCode([[i] for i in 1:5],Int[])
    a = zeros(2,2,2,2,2,2)
    tensors= fill([0.1,0.9],5)
    ptn = TensorNetwork(code,tensors)

    tn = TensorQEC.stg2tensornetwork(tanner,ptn)
end
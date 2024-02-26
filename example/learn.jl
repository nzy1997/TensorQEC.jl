using TensorQEC, TensorQEC.Yao
using TensorInference
qc=chain(cnot(2, 1, 2))
cl=clifford_network(qc)
ps = Dict([1=>TensorQEC.BoundarySpec((0.0,1.0,0.0,0.0), false),2=>TensorQEC.BoundarySpec((1.0,0.0,0.0,0.0), false)])
qs = Dict([i=>TensorQEC.BoundarySpec((ones(Float64,4)...,), true) for i in 1:2])
tn=generate_tensor_network(cl,ps, qs)

marginals(tn)
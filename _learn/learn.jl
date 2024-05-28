include("functions.jl")

#Toric code
qc, data_qubits,num_qubits = toric_code_cir(3)
qcf,srs = ein_cir(qc, data_qubits, num_qubits)
tn = qc2enisum(qcf,srs,data_qubits,num_qubits)
optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())
contract(optnet)/2^8

# random example
Random.seed!(214)
u2 = rand_unitary(4)
u4 = rand_unitary(16)
toyqc = chain(4, put(4, (1,2) => GeneralMatrixBlock(u2; nlevel=2, tag="randu1")), put(4, (1,2,3,4) => GeneralMatrixBlock(u4; nlevel=2, tag="randu2")))
qcf,srs = ein_cir(toyqc,[1,2],4)
tn = qc2enisum(qcf,srs,[1,2],4)
optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())
contract(optnet)

u1 = kron(u2,u2')
kraus = get_kraus(u4, 2)
u1app = mapreduce(x->kron(x,x'),+,kraus)
tr(u1*u1app)


# x xcnot example
u1 = mat(X)
u2 = mat(kron(I2,X)) * mat(cnot(2,2,1))
toyqc = chain(2, put(2,1=>GeneralMatrixBlock(u1; nlevel=2, tag="X")),put(2, (1,2) => GeneralMatrixBlock(u2; nlevel=2, tag="XCNOT")))
qcf,srs = ein_cir(toyqc,[1],2)
tn = qc2enisum(qcf,srs,[1],2)
optnet = optimize_code(tn, TreeSA(), OMEinsum.MergeVectors())
contract(optnet)

u2 = kron(u1,u1')
kraus = get_kraus(u2, 1)
u2app = mapreduce(x -> kron(x,x'), +, kraus)
tr(u2 * u2app)

vizcircuit(qcf; filename = "_learn/toric_code.svg")
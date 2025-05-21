using Test, TensorQEC, TensorQEC.Yao

@testset "errorcorrect_circuit" begin
    t = ToricCode(3, 3)
	st = stabilizers(t)
    table = TensorQEC.correction_dict(st, 1;et = "XZ")
    qc1,st_pos = measure_circuit_fault_tol(st)
    qc2 = correction_circuit(table,9,8, st_pos,nqubits(qc1))
    @test chain(qc1, qc2) isa ChainBlock
end
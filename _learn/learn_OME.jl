using OMEinsum

a, b = randn(2, 2), randn(2, 2)

c = ein"ij,jk -> ik"(a,b)
@ein c[i,k] := a[i,j] * b[j,k]
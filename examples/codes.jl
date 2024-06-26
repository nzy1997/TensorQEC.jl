# # QEC Codes
# We provide a number of quantum error correction codes. 
using TensorQEC

# ## Toric Code
# The Toric code is a 2D topological code. We can define a toric code instance  by `ToricCode(m, n)`, where `m` and `n` are the number of rows and columns of the toric code.
tc = ToricCode(2, 3)

# The toric code has two types of stabilizers: X stabilizers and Z stabilizers. X stabilizers are plaquettes of the lattice, and Z stabilizers are vertices of the lattice. We can get the stabilizers of the toric code by
st = stabilizers(tc)

# ## Surface Code
# The surface code is a 2D topological code. Similarly to Toric code, we can define a surface code instance by `SurfaceCode(m, n)` and get the stabilizers of the surface code by `stabilizers(sc)`.
sc = SurfaceCode(3, 3)
st = stabilizers(sc)

# ## 
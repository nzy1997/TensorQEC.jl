# # QEC Codes
# We provide a number of quantum error correction codes. 
using TensorQEC

# ## Toric Code
# The Toric code is a 2D topological code[^Kitaev]. We can define a Toric code instance by [`ToricCode`](@ref).
tc = ToricCode(2, 3)
# Here is a schematic diagram of 2*3 Toric code:
# ![](../images/toric.svg)

# The Toric code has two types of stabilizers: X stabilizers and Z stabilizers. X stabilizers are plaquettes of the lattice, and Z stabilizers are vertices of the lattice. We can get the stabilizers of the toric code by
st = stabilizers(tc)

# Note the order of pauli strings is following the big-endian convention. For example, the Pauli string XYZ means $X_1Y_2Z_3$.

# ## Surface Code
# The surface code is a 2D topological code[^Kitaev]. Similarly to Toric code, we can define a surface code instance by [`SurfaceCode`](@ref) and get the stabilizers of the surface code by [`stabilizers`](@ref).
sc = SurfaceCode(3, 3)
st = stabilizers(sc)
# Here is a schematic diagram of 3*3 surface code:
# ![](../images/surface.svg)

# ## Shor Code
# The [[9,1,3]] Shor code[^Shor] functions by concatenating each qubit of a phase-flip with a bit-flip repetition code, allowing it to correct both types of errors at the same time. We can define a Shor code instance by [`ShorCode`](@ref).
shor = ShorCode()
st = stabilizers(shor)

# ## Steane Code
# The [[7,1,3]] Steane is constructed using the classical binary [7,4,3] Hamming code[^Steane]. We can define a Steane code instance by [`SteaneCode`](@ref).

# ![](../images/steane.svg)
steane = SteaneCode()
st = stabilizers(steane)

# ## [[8,3,2]] Code
# The [[8,3,2]] CSS code is the smallest non-trivial 3D color code[^Menendez][^Code832]. We can define a CSS [[8,3,2]] code instance by [`Code832`](@ref).

# ![](../images/code832.svg)
code832 = Code832()
st = stabilizers(code832)

# [^Kitaev]: Kitaev, A. Y. Fault-Tolerant Quantum Computation by Anyons. Annals of Physics 2003, 303 (1), 2–30. https://doi.org/10.1016/S0003-4916(02)00018-0.

# [^Menendez]: Menendez, D. H.; Ray, A.; Vasmer, M. Implementing Fault-Tolerant Non-Clifford Gates Using the [[8,3,2]] Color Code. arXiv September 15, 2023. http://arxiv.org/abs/2309.08663.

# [^Code832]: [THE SMALLEST INTERESTING COLOUR CODE](https://earltcampbell.com/2016/09/26/the-smallest-interesting-colour-code/)

# [^Shor]: Shor, P. W. (1995). Scheme for reducing decoherence in quantum computer memory. Physical Review A, 52(4), R2493–R2496. https://doi.org/10.1103/PhysRevA.52.R2493

# [^Steane]: Steane, A. (1997). Multiple-particle interference and quantum error correction. Proceedings of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences, 452(1954), 2551–2577. https://doi.org/10.1098/rspa.1996.0136

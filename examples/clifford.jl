# # Pauli Basis and Clifford group
# In this section, we introduce the definition of Pauli strings and basic operations on them. We also introduce the Clifford group and how to simulate a Clifford circuit on Pauli strings.

# ## Pauli Strings
# A pauli string is a tensor product of Pauli operators acting on different qubits. [`PauliString`](@ref) is a subtype of [`CompositeBlock`] with a field `ids` storing the Pauli operators. We can define pauli string with [`PauliString`](@ref) or [`paulistring`](@ref).
using TensorQEC, TensorQEC.Yao
PauliString(2,1, 4, 3) # X_1Z_3Y_4

paulistring(4, 2, (1, 2, 4)) # X_1X_2X_4

# We can use [`Yao.mat`] to get the matrix representation of a Pauli string.
mat(ComplexF64, PauliString(2,4)) # X_1Z_2

# ## Pauli Basis
# [`pauli_basis`](@ref) generates all the Pauli strings of a given length. Those Pauli strings are stored in a high-dimensional array.
pauli_basis(2)

# [`pauli_decomposition`](@ref) returns the coefficients of a matrix in the Pauli basis.
pauli_decomposition(mat(ConstGate.CNOT))

# That implies that $CNOT = \frac{1}{2} (I \otimes I + I \otimes X + Z \otimes I - Z \otimes X)$. We can check this by
0.5*(mat(kron(I2,I2)) + mat(kron(I2,X)) + mat(kron(Z,I2)) - mat(kron(Z,X))) == mat(ConstGate.CNOT)

# [`pauli_mapping`](@ref) converts a linear operator into the Pauli basis. For Hadamard gate H, we know that $HIH = I, HXH = Z, HYH = -Y, HZH = X$. We can convert $H$ into the Pauli basis.
pauli_mapping(mat(H))

# ## Clifford Group
# We know that the Clifford group is generated by Hadamard gate and CNOT gate. We can generate the Clifford group by [`clifford_group`](@ref).
clifford_group(1)

# Each element in the Clifford group acts on pauli basis as a permutation matrix. We can use [`to_perm_matrix`](@ref) to convert a matrix into a permutation matrix.
pm = to_perm_matrix(Int8, Int, H)
pm.perm, pm.vals

# With the permutation matrix, we can apply a Clifford gate to a Pauli string by [`perm_of_paulistring`](@ref). Here we apply the Hadamard gate to the second qubit of Pauli string $I_1X_2$ and get $I_1Z_2$ with a phase $1$.
ps1 = PauliString((1, 2))
ps2, val = perm_of_paulistring(pm, ps1, [2])
ps1, ps2, val

# Put those all together, we can apply a Clifford circuit to a Pauli string by [`clifford_simulate`](@ref).
qc = chain(put(5, 1 => H), control(5, 1, 2 => Z), control(5, 3, 4 => X), control(5, 5, 3 => X), put(5, 1 => X))
vizcircuit(qc)

# Apply the circuit to Pauli string $Z_1Y_2I_3Y_4X_5$, we get $Y_1X_2Y_3Y_4Y_5$ with a phase $1$.
ps = PauliString((4, 3, 1, 3, 2))
ps2, val = clifford_simulate(ps, qc)

# We can check the result by
val * mat(qc) * mat(ps) * mat(qc)' ≈ mat(ps2)

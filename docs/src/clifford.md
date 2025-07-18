# Pauli Basis and Clifford group
In this section, we introduce the definition of Pauli strings and basic operations on them. We also introduce the Clifford group and how to simulate a Clifford circuit applied on Pauli strings.

## Pauli Strings
A pauli string is a tensor product of Pauli operators acting on different qubits. [`PauliString`](@ref) is a subtype of `CompositeBlock` with a field `ids` storing the Pauli operators. We can define pauli string with [`PauliString`](@ref) or [`@P_str`](@ref) string literal.

```@example clifford
using TensorQEC, TensorQEC.Yao
PauliString(Pauli(1), Pauli(0), Pauli(3), Pauli(2)) # X_1Z_3Y_4
PauliString(4, (1, 2, 4)=>Pauli(1)) # X_1X_2X_4
P"XZZ"
```

Note that the printed Pauli string is in **big-endian** order, i.e. the first Pauli operator is the leftmost one.
We can check if two Pauli strings commute or anticommute with `iscommute` and [`isanticommute`](@ref).
````@example clifford
p1 = PauliString(5, (1,2)=>Pauli(1))
p2 = PauliString(5, (1,2)=>Pauli(3))
iscommute(p1, p2)
````

We can use `Yao.mat` to get the matrix representation of a Pauli string.

````@example clifford
mat(ComplexF64, P"XZ") # X_1Z_2
````

## Pauli Group Element
[`PauliGroupElement`](@ref) is a Pauli string with a phase factor. It is a subtype of `AbstractPauli`.

````@example clifford
PauliGroupElement(1, P"XZ") # +i * XZ
````

The product of two Pauli strings is a Pauli group element.

````@example clifford
P"XZ" * P"YI" # +i * ZZ
````

`iscommute` and [`isanticommute`](@ref) also work for Pauli group element.

````@example clifford
iscommute(P"XZ" * P"YI", PauliGroupElement(1, P"XZ"))
isanticommute(P"XZ" * P"YI",PauliGroupElement(1, P"XZ"))
````

We can also use `Yao.mat` to get the matrix representation of a Pauli group element.

````@example clifford
mat(ComplexF64, PauliGroupElement(1, P"XZ"))
````

## Linear Combination of Pauli strings
[`SumOfPaulis`](@ref) is a linear combination of Pauli strings, i.e. $c_1 P_1 + c_2 P_2 + \cdots + c_n P_n$. It is a subtype of `AbstractPauli`.

````@example clifford
sp = SumOfPaulis([0.6=>P"IXY", 0.8=>P"ZZY"])
````

The sum of two Pauli strings or two Pauli group elements is a [`SumOfPaulis`](@ref).

````@example clifford
P"IXY" + P"ZZY"
PauliGroupElement(1, P"XZ") + PauliGroupElement(2, P"YI")
````

We can also use `Yao.mat` to get the matrix representation of a [`SumOfPaulis`](@ref).

````@example clifford
mat(ComplexF64, sp)
````


## Pauli Basis
[`pauli_basis`](@ref) generates all the Pauli strings of a given length. Those Pauli strings are stored in a high-dimensional array.

````@example clifford
pauli_basis(2)
````

[`pauli_decomposition`](@ref) returns the coefficients of a matrix in the Pauli basis. The returned coefficients are stored in a [`SumOfPaulis`](@ref).

````@example clifford
pauli_decomposition(ConstGate.CNOT)
````

That implies that $\mathrm{CNOT} = \frac{1}{2} (I \otimes I + I \otimes X + Z \otimes I - Z \otimes X)$. We can check this by

````@example clifford
# Note: `Yao.kron` has an inversed order of the arguments compared to `LinearAlgebra.kron`.
0.5*(mat(kron(I2,I2) + kron(I2,X) + kron(Z,I2) - kron(Z,X))) ≈ mat(ConstGate.CNOT)
````

[`pauli_repr`](@ref) returns the matrix representation of a quantum gate in the Pauli basis. For Hadamard gate H, we know that $HIH = I, HXH = Z, HYH = -Y, HZH = X$. We can convert $H$ into the Pauli basis.

````@example clifford
pauli_repr(H)
````

## Clifford Group
Clifford group is the set of all permutations of Pauli group, i.e. the `pauli_repr` of its elements are "permutation matrices" with phases.
It can be generated by Hadamard gate, S gate and CNOT gate[^Bravyi2022].
We can use `TensorQEC.clifford_group` to generate the Clifford group.

````@example clifford
TensorQEC.clifford_group(1)
````

Each element in the Clifford group acts on pauli basis as a permutation matrix.
For $n= 1, 2$, and $3$, this group contains $24$, $11520$, and $92897280$ elements, respectively.
We can use [`CliffordGate`](@ref) to convert a Yao gate into a Clifford gate, which is characterized by a permutation (with phases) of Pauli basis.

````@example clifford
pm = CliffordGate(H)
````

With the permutation matrix representation, we can efficienlly simulate a Clifford circuit.
We first show how to apply a Clifford gate to a Pauli string.
Here we apply the Hadamard gate to the second qubit of Pauli string $I_1X_2$ and get $I_1Z_2$ with a phase $1$.

````@example clifford
ps1 = P"IX"  # same as: PauliString(Pauli(0), Pauli(1))
elem = pm(ps1, (2,))
````

Put those all together, we can apply a Clifford circuit to a Pauli string by [`clifford_simulate`](@ref).

````@example clifford
qc = chain(put(5, 1 => H), control(5, 1, 2 => Z), control(5, 3, 4 => X), control(5, 5, 3 => X), put(5, 1 => X))
vizcircuit(qc)
````

Apply the circuit to Pauli string $Z_1Y_2I_3Y_4X_5$, we get $Y_1X_2Y_3Y_4Y_5$ with a phase $1$.

````@example clifford
ps = P"ZYIYX"
res = clifford_simulate(ps, qc)
ps2 = res.output
````

where `res.output` is the Pauli string after the Clifford circuit and `res.phase` is the phase factor. It corresponds to the following quantum circuit.

````@example clifford
clifford_simulation_circuit = chain(qc', yaoblock(ps), qc)
CircuitStyles.barrier_for_chain[] = true  # setup barrier for better visualization
vizcircuit(clifford_simulation_circuit)
````
Here, we use [`yaoblock`](@ref) to convert the Pauli string to a Yao block.

We can check the result by

```@example clifford
CircuitStyles.barrier_for_chain[] = false  # disable barrier
mat(clifford_simulation_circuit) ≈ mat(yaoblock(ps2))
```

We can also visualize the history of Pauli strings by `TensorQEC.annotate_history`.

````@example clifford
TensorQEC.annotate_history(res)
````

[^Bravyi2022]: Bravyi, S., Latone, J.A., Maslov, D., 2022. 6-qubit optimal Clifford circuits. npj Quantum Inf 8, 1–12. https://doi.org/10.1038/s41534-022-00583-7

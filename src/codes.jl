"""
	ToricCode(m::Int, n::Int)

Construct a Toric code with `m` rows and `n` columns. 
"""
struct ToricCode
	m::Int
	n::Int
end

nsite(t::ToricCode) = t.m * t.n
Yao.nqubits(t::ToricCode) = 2 * nsite(t)
vertical_edges(t::ToricCode) = reshape(1:nsite(t), t.m, t.n)
horizontal_edges(t::ToricCode) = reshape(nsite(t)+1:2*nsite(t), t.m, t.n)

# Toric code (2*2)
# ∘---5---∘---7---∘
# |       |       |
# 1   ∘   3   ∘   1 
# |       |       |
# ∘---6---∘---8---∘
# |       |       | 
# 2   ∘   4   ∘   2
# |       |       |
# ∘---5---∘---7---∘


"""
	stabilizers(tc::ToricCode)
	stabilizers(sc::SurfaceCode)
	stabilizers(shor::ShorCode)
	stabilizers(steane::SteaneCode)
	stabilizers(code832::Code832)

Get the stabilizers of the code instances.
"""
function stabilizers(toric::ToricCode; linearly_independent::Bool = true)
	nq, m, n = nqubits(toric), toric.m, toric.n
	output = PauliString{nq}[]
	# numbering the qubits
	ve = vertical_edges(toric)
	he = horizontal_edges(toric)
	# X stabilizers
	for j in 1:n, i in 1:m
		i == m && j == n && linearly_independent && continue # not linearly independent
		push!(output, paulistring(nq, 2, (
			he[i, j], he[mod1(i + 1, m), j],
			ve[i, j], ve[i, mod1(j + 1, n)],
		)))
	end
	# Z stabilizers
	for j in 1:n, i in 1:m
		i == m && j == n && linearly_independent && continue # not linearly independent
		push!(output, paulistring(nq, 4, (
			ve[i, j], ve[mod1(i - 1, m), j],
			he[i, j], he[i, mod1(j - 1, n)],
		)))
	end
	return output
end


"""
	SurfaceCode(m::Int, n::Int)

Construct a surface code with `m` rows and `n` columns. 
"""
struct SurfaceCode	
    m::Int
	n::Int
end

# Surface code (3*3)
#       -
#     / z \
#     1---2---3 -\
#     | x | z | x |
#  /- 4---5- -6 -/
# | x | z | x |
#  \- 7---8---9
#		  \ z /
#			-
# 8 stableizers:
# X type: 36, 1245, 5689, 47
# Z type: 12, 2356, 4578, 89


function stabilizers(sc::SurfaceCode)
    m, n = sc.m, sc.n
	qubit_config = reshape(1:m*n, n, m)' 
	pauli_string = PauliString{m*n}[]
	for i in 1:m-1, j in 1:n-1
		if mod(i+j, 2) == 0
			push!(pauli_string, paulistring(m*n, 2, (qubit_config[i, j], qubit_config[i+1, j], qubit_config[i, j+1], qubit_config[i+1, j+1])))
		end
	end
	for i in 1:m÷2
		push!(pauli_string, paulistring(m*n, 2, (qubit_config[2*i-1+mod(n+1,2), n], qubit_config[2*i+mod(n+1,2), n])))
		if 2*i+1 <= m
			push!(pauli_string, paulistring(m*n, 2, (qubit_config[2*i, 1], qubit_config[2*i+1, 1])))
		end
	end
	for i in 1:m-1, j in 1:n-1
		if mod(i+j, 2) == 1
			push!(pauli_string, paulistring(m*n, 4, (qubit_config[i, j], qubit_config[i+1, j], qubit_config[i, j+1], qubit_config[i+1, j+1])))
		end
	end
	for j in 1:n÷2
		push!(pauli_string, paulistring(m*n, 4, (qubit_config[1, 2*j-1], qubit_config[1, 2*j])))
		if 2*j+1 <= n
			push!(pauli_string, paulistring(m*n, 4, (qubit_config[m, 2*j-1+mod(m,2)], qubit_config[m, 2*j+mod(m,2)])))
		end
	end
	return pauli_string
end

"""
	ShorCode

Construct a Shor code instance.
"""
struct ShorCode end

function stabilizers(::ShorCode; linearly_independent::Bool = true)
	nq = 9
	pauli_string = PauliString{nq}[]
	push!(pauli_string, paulistring(nq, 2, (1,2,3,4,5,6)))
	push!(pauli_string, paulistring(nq, 2, (1, 2, 3, 7,8,9)))
	linearly_independent || push!(pauli_string,paulistring(nq, 2, (4, 5, 6, 7, 8, 9)))
	push!(pauli_string, paulistring(nq, 4, (1, 2)))
	push!(pauli_string, paulistring(nq, 4, (1, 3)))
	linearly_independent || push!(pauli_string, paulistring(nq, 4, (2, 3)))
	push!(pauli_string, paulistring(nq, 4, (4, 5)))
	push!(pauli_string, paulistring(nq, 4, (4,6)))
	linearly_independent || push!(pauli_string, paulistring(nq, 4, (5, 6)))
	push!(pauli_string, paulistring(nq, 4, (7,8)))
	push!(pauli_string, paulistring(nq, 4, (7, 9)))
	linearly_independent || push!(pauli_string, paulistring(nq, 4, (8, 9)))
	return pauli_string
end

"""
	SteaneCode

Construct a Steane code instance.
"""
struct SteaneCode end

function stabilizers(::SteaneCode)
	nq = 7
	pauli_string = PauliString{nq}[]
	push!(pauli_string, paulistring(nq, 2, (1,3,5,7)))
	push!(pauli_string, paulistring(nq, 2, (2,3,6,7)))
	push!(pauli_string, paulistring(nq, 2, (4,5,6,7)))
	push!(pauli_string, paulistring(nq, 4, (1,3,5,7)))
	push!(pauli_string, paulistring(nq, 4, (2,3,6,7)))
	push!(pauli_string, paulistring(nq, 4, (4,5,6,7)))
	return pauli_string
end

"""
	Code832

Construct a [[8,3,2]] CSS code instance.
"""
struct Code832 end

function stabilizers(::Code832)
	nq = 8
	pauli_string = PauliString{nq}[]
	push!(pauli_string, PauliString(fill(2, 8)...))
	push!(pauli_string, PauliString(fill(4, 8)...))
	push!(pauli_string, paulistring(nq, 4, (1,3,5,7)))
	push!(pauli_string, paulistring(nq, 4, (1,2,3,4)))
	push!(pauli_string, paulistring(nq, 4, (1,2,5,6)))
	return pauli_string
end

"""
	Code422

Construct a [[4,2,2]] CSS code instance.
"""
struct Code422 end

function stabilizers(::Code422)
	nq = 4
	pauli_string = PauliString{nq}[]
	push!(pauli_string, PauliString(fill(2, 4)...))
	push!(pauli_string, PauliString(fill(4, 4)...))
	return pauli_string
end

"""
	Code1573

Construct a [[15,7,3]] CSS code instance.
"""
struct Code1573 end

function stabilizers(::Code1573)
	nq = 15
	pauli_string = PauliString{nq}[]
	push!(pauli_string, paulistring(nq, 2, 8:15))
	push!(pauli_string, paulistring(nq, 2, (4:7) ∪ (12:15)))
	push!(pauli_string, paulistring(nq, 2, (2,3,6,7,10,11,14,15)))
	push!(pauli_string, paulistring(nq, 2, 1:2:15))
	push!(pauli_string, paulistring(nq, 4, 8:15))
	push!(pauli_string, paulistring(nq, 4, (4:7) ∪ (12:15)))
	push!(pauli_string, paulistring(nq, 4, (2,3,6,7,10,11,14,15)))
	push!(pauli_string, paulistring(nq, 4, 1:2:15))
	return pauli_string
end
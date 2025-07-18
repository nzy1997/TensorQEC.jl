abstract type QuantumCode end
abstract type CSSQuantumCode <: QuantumCode end

"""
	SurfaceCode(m::Int, n::Int)

Construct a rotating surface code with `m` rows and `n` columns.
"""
struct SurfaceCode <: CSSQuantumCode
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

"""
	stabilizers(tc::ToricCode)
	stabilizers(sc::SurfaceCode)
	stabilizers(shor::ShorCode)
	stabilizers(steane::SteaneCode)
	stabilizers(code832::Code832)

Get the stabilizers of the code instances.
"""
function stabilizers(sc::SurfaceCode)
    m, n = sc.m, sc.n
	qubit_config = reshape(1:m*n, n, m)' 
	pauli_string = PauliString{m*n}[]
	for i in 1:m-1, j in 1:n-1
		if mod(i+j, 2) == 0
			push!(pauli_string, PauliString(m*n, (qubit_config[i, j], qubit_config[i+1, j], qubit_config[i, j+1], qubit_config[i+1, j+1])=>Pauli(1)))
		end
	end
	for i in 1:m÷2
		if 2*i+mod(n+1,2) <= m
			push!(pauli_string, PauliString(m*n, (qubit_config[2*i-1+mod(n+1,2), n], qubit_config[2*i+mod(n+1,2), n])=>Pauli(1)))
		end
		if 2*i+1 <= m
			push!(pauli_string, PauliString(m*n, (qubit_config[2*i, 1], qubit_config[2*i+1, 1])=>Pauli(1)))
		end
	end
	for i in 1:m-1, j in 1:n-1
		if mod(i+j, 2) == 1
			push!(pauli_string, PauliString(m*n, (qubit_config[i, j], qubit_config[i+1, j], qubit_config[i, j+1], qubit_config[i+1, j+1])=>Pauli(3)))
		end
	end
	for j in 1:n÷2
		push!(pauli_string, PauliString(m*n, (qubit_config[1, 2*j-1], qubit_config[1, 2*j])=>Pauli(3)))
		if 2*j + mod(m,2) <= n
			push!(pauli_string, PauliString(m*n, (qubit_config[m, 2*j-1+mod(m,2)], qubit_config[m, 2*j+mod(m,2)])=>Pauli(3)))
		end
	end
	return pauli_string
end

"""
	ShorCode

Construct a Shor code instance.
"""
struct ShorCode  <: CSSQuantumCode end

function stabilizers(::ShorCode; linearly_independent::Bool = true)
	nq = 9
	pauli_string = PauliString{nq}[]
	push!(pauli_string, PauliString(nq, (1,2,3,4,5,6)=>Pauli(1)))
	push!(pauli_string, PauliString(nq, (1, 2, 3, 7,8,9)=>Pauli(1)))
	linearly_independent || push!(pauli_string, PauliString(nq, (4, 5, 6, 7, 8, 9)=>Pauli(1)))
	push!(pauli_string, PauliString(nq, (1, 2)=>Pauli(3)))
	push!(pauli_string, PauliString(nq, (1, 3)=>Pauli(3)))
	linearly_independent || push!(pauli_string, PauliString(nq, (2, 3)=>Pauli(3)))
	push!(pauli_string, PauliString(nq, (4, 5)=>Pauli(3)))
	push!(pauli_string, PauliString(nq, (4,6)=>Pauli(3)))
	linearly_independent || push!(pauli_string, PauliString(nq, (5, 6)=>Pauli(3)))
	push!(pauli_string, PauliString(nq, (7,8)=>Pauli(3)))
	push!(pauli_string, PauliString(nq, (7, 9)=>Pauli(3)))
	linearly_independent || push!(pauli_string, PauliString(nq, (8, 9)=>Pauli(3)))
	return pauli_string
end

"""
	SteaneCode

Construct a Steane code instance.
"""
struct SteaneCode   <: CSSQuantumCode end

function stabilizers(::SteaneCode)
	nq = 7
	pauli_string = PauliString{nq}[]
	push!(pauli_string, PauliString(nq, (1,3,5,7) => Pauli(1)))
	push!(pauli_string, PauliString(nq, (2,3,6,7) => Pauli(1)))
	push!(pauli_string, PauliString(nq, (4,5,6,7) => Pauli(1)))
	push!(pauli_string, PauliString(nq, (1,3,5,7) => Pauli(3)))
	push!(pauli_string, PauliString(nq, (2,3,6,7) => Pauli(3)))
	push!(pauli_string, PauliString(nq, (4,5,6,7) => Pauli(3)))
	return pauli_string
end

"""
	Code832

Construct a [[8,3,2]] CSS code instance.
"""
struct Code832  <: CSSQuantumCode end

function stabilizers(::Code832)
	nq = 8
	pauli_string = PauliString{nq}[]
	push!(pauli_string, PauliString(fill(Pauli(1), 8)...))
	push!(pauli_string, PauliString(fill(Pauli(3), 8)...))
	push!(pauli_string, PauliString(nq, (1,3,5,7) => Pauli(3)))
	push!(pauli_string, PauliString(nq, (1,2,3,4) => Pauli(3)))
	push!(pauli_string, PauliString(nq, (1,2,5,6) => Pauli(3)))
	return pauli_string
end

"""
	Code422

Construct a [[4,2,2]] CSS code instance.
"""
struct Code422  <: CSSQuantumCode end

function stabilizers(::Code422)
	nq = 4
	pauli_string = PauliString{nq}[]
	push!(pauli_string, PauliString(fill(Pauli(1), 4)...))
	push!(pauli_string, PauliString(fill(Pauli(3), 4)...))
	return pauli_string
end

"""
	Code1573

Construct a [[15,7,3]] CSS code instance.
"""
struct Code1573  <: CSSQuantumCode end

function stabilizers(::Code1573)
	nq = 15
	pauli_string = PauliString{nq}[]
	push!(pauli_string, PauliString(nq, 8:15 => Pauli(1)))
	push!(pauli_string, PauliString(nq, (4:7) ∪ (12:15) => Pauli(1)))
	push!(pauli_string, PauliString(nq, (2,3,6,7,10,11,14,15) => Pauli(1)))
	push!(pauli_string, PauliString(nq, 1:2:15 => Pauli(1)))
	push!(pauli_string, PauliString(nq, 8:15 => Pauli(3)))
	push!(pauli_string, PauliString(nq, (4:7) ∪ (12:15) => Pauli(3)))
	push!(pauli_string, PauliString(nq, (2,3,6,7,10,11,14,15) => Pauli(3)))
	push!(pauli_string, PauliString(nq, 1:2:15 => Pauli(3)))
	return pauli_string
end

"""
	Code513

Construct a [[5,1,3]] code instance.
"""	
struct Code513  <: QuantumCode end

function stabilizers(::Code513)
	nq = 5
    i, x, y, z = Pauli(0), Pauli(1), Pauli(2), Pauli(3)
	pauli_string = PauliString{nq}[]
	push!(pauli_string, PauliString((x,z,z,x,i)))
	push!(pauli_string, PauliString((i,x,z,z,x)))
	push!(pauli_string, PauliString((x,i,x,z,z)))
	push!(pauli_string, PauliString((z,x,i,x,z)))
	return pauli_string
end

struct BivariateBicycleCode{N} <: CSSQuantumCode
    m::Int
    n::Int
	vc::NTuple{N,Tuple{Int,Int}}
	hd::NTuple{N,Tuple{Int,Int}}
end
nsite(bb::BivariateBicycleCode) = bb.m * bb.n
Yao.nqubits(bb::BivariateBicycleCode) = 2 * nsite(bb)
vertical_edges(bb::BivariateBicycleCode) = reshape(1:nsite(bb), bb.m, bb.n)
horizontal_edges(bb::BivariateBicycleCode) = reshape(nsite(bb)+1:2*nsite(bb), bb.m, bb.n)

function stabilizers(bb::BivariateBicycleCode{N};rm_linear_dependency::Bool = true) where N
	nq, m, n = nqubits(bb), bb.m, bb.n
	output = PauliString{nq}[]
	# numbering the qubits
	ve = vertical_edges(bb)
	he = horizontal_edges(bb)
	# X stabilizers
	for j in 1:n, i in 1:m
		st = Int[]
		for (s,t) in bb.hd
			push!(st, ve[mod1(i+s, m), mod1(j+t, n)])
		end
		for (s,t) in bb.vc
			push!(st, he[mod1(i+s, m), mod1(j+t, n)])
		end
		push!(output, PauliString(nq, st => Pauli(1)))
	end
	# Z stabilizers
	for j in 1:n, i in 1:m	
		st = Int[]
		for (s,t) in bb.vc
			push!(st, ve[mod1(i-s, m), mod1(j-t, n)])
		end
		for (s,t) in bb.hd
			push!(st, he[mod1(i-s, m), mod1(j-t, n)])
		end
		push!(output, PauliString(nq, st => Pauli(3)))
	end
	if rm_linear_dependency
		output = remove_linear_dependency(output)
	end
	return output
end

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
	ToricCode(m::Int, n::Int)

Construct a Toric code with `m` rows and `n` columns. 
"""
ToricCode(m::Int,n::Int) = BivariateBicycleCode(m,n,((1,0),(0,0)), ((0,1),(0,0)))

abstract type TriangularColorCode <: CSSQuantumCode end
struct Color488 <: TriangularColorCode
	d::Int
end

struct Color666 <: TriangularColorCode
	d::Int
end

# The following two functions are copied from "https://github.com/QuantumSavory/QuantumClifford.jl/pull/361" a pull request to QuantumClifford.jl
function _colorcode_get_check_matrix(c::Color488)
    n = (c.d^2 + 2*c.d - 1) ÷ 2
    num_checks = (n-1)÷2
    num_layers = (c.d-1)÷2
    checks = zeros(Bool, num_checks, n)
    
    i = 1
    checks_written = 0
    for layer in 1:num_layers
        # Convert half 8-gons from previous layer into full 8-gons
        num_8_gons = layer-1
        checks_written -= num_8_gons
        for j in 1:(num_8_gons)
            checks[checks_written+1,i+2*layer+1+(j-1)*2] = 1
            checks[checks_written+1,i+2*layer+j*2] = 1

            offset = 0
            if layer == num_layers && num_layers%2==0
                offset = 1
            end

            checks[checks_written+1,(2*layer+1)+i+2*layer+1+(j-1)*2 - offset] = 1
            checks[checks_written+1,(2*layer+1)+i+2*layer+j*2 - offset] = 1

            checks_written += 1
        end

        # red 4-gons
        for j in 0:(layer-1)
            checks[checks_written+1,i+j*2] = 1
            checks[checks_written+1,i+1+j*2] = 1
            checks[checks_written+1,i+2*layer+j*2] = 1
            checks[checks_written+1,i+2*layer+1+j*2] = 1
            checks_written += 1
        end

        if layer%2 == 1
            # green half 8-gons on left side
            checks[checks_written+1,i] = 1
            checks[checks_written+1,i+2*layer] = 1
            checks[checks_written+1,i+4*layer] = 1
            checks[checks_written+1,i+4*layer+1] = 1
            checks_written += 1
        else
            # blue half 8-gon on right side
            checks[checks_written+1,i+1+(layer-1)*2] = 1
            checks[checks_written+1,i+2*layer+1+(layer-1)*2] = 1

            # when d=5,9,13,... the final row qubits is indexed slightly differently.
            offset = 0
            if layer == num_layers
                offset = 1
            end

            checks[checks_written+1,(i+4*layer)+layer*2-offset] = 1
            checks[checks_written+1,(i+4*layer)+1+layer*2-offset] = 1
            checks_written += 1
        end

        # blue/green half 8-gons on the bottom
        for j in 0:(layer-1)
            checks[checks_written+1,i+2*layer+j*2] = 1
            checks[checks_written+1,i+2*layer+1+j*2] = 1

            offset = 0
            if layer == num_layers && num_layers%2==0
                offset = 1
            end

            checks[checks_written+1,(2*layer+1)+i+2*layer+j*2 - offset] = 1
            checks[checks_written+1,(2*layer+1)+i+2*layer+1+j*2 - offset] = 1

            checks_written += 1
        end

        i += 4*layer  
    end
    return checks
end

function _colorcode_get_check_matrix(c::Color666) 
    n = (3*c.d^2 + 1) ÷ 4
    num_checks = (n-1)÷2
    num_layers = (c.d-1)÷2
    checks = zeros(Bool, num_checks, n)

    i = 1
    checks_written = 0
    for layer in 1:num_layers
        # extend half 6-gons from last iteration to full hexagons
        num_6_gons = layer-1
        checks_written -= num_6_gons
        for j in 1:(num_6_gons)
            init_pos = i + 2*(j-1)
            checks[checks_written+1,init_pos+2*(layer-1)+1] = 1
            checks[checks_written+1,init_pos+2*(layer-1)+2] = 1
            checks_written+=1
        end

        # red trapezoid
        checks[checks_written+1,i] = 1
        checks[checks_written+1,i+(layer-1)*2+1] = 1
        checks[checks_written+1,i+2+4*(layer-1)] = 1
        checks[checks_written+1,i+2+4*(layer-1)+1] = 1
        checks_written += 1

        # blue trapezoid
        checks[checks_written+1,i+1+4*(layer-1)] = 1
        checks[checks_written+1,i+1+4*(layer-1)+layer*2] = 1
        checks[checks_written+1,i+5+8*(layer-1)] = 1
        checks[checks_written+1,i+5+8*(layer-1)+1] = 1
        checks_written += 1

        # red hexagons
        for j in 1:(layer-1)
            checks[checks_written+1,i+(j-1)*2+1] = 1
            checks[checks_written+1,i+(j-1)*2+2] = 1
            checks[checks_written+1,i+(j-1)*2+2+2*(layer-1)] = 1
            checks[checks_written+1,i+(j-1)*2+2+2*(layer-1)+1] = 1
            checks[checks_written+1,i+(j-1)*2+2+2*(layer-1)+layer*2] = 1
            checks[checks_written+1,i+(j-1)*2+2+2*(layer-1)+layer*2+1] = 1

            checks_written += 1
        end

        # blue hexagons
        for j in 1:(layer-1)
            init_pos = i+(j-1)*2+(layer-1)*2+1
            checks[checks_written+1, init_pos] = 1
            checks[checks_written+1, init_pos+1] = 1 
            checks[checks_written+1, init_pos+2*layer] = 1 
            checks[checks_written+1, init_pos+2*layer+1] = 1 
            checks[checks_written+1, init_pos+4*layer] = 1 
            checks[checks_written+1, init_pos+4*layer+1] = 1 

            checks_written += 1
        end

        # green half 6gons
        for j in 0:(layer-1)
            init_pos = i+2*j+2+4*(layer-1)
            checks[checks_written+1,init_pos] = 1
            checks[checks_written+1,init_pos+1] = 1
            checks[checks_written+1,init_pos+2*layer] = 1
            checks[checks_written+1,init_pos+2*layer+1] = 1
            checks_written += 1
        end

        i += 4+6*(layer-1)
    end

    return checks
end

function stabilizers(c::TriangularColorCode)
	checks = _colorcode_get_check_matrix(c)
	nq = size(checks,2)
	pauli_string = PauliString{nq}[]
	for i in axes(checks,1)
		push!(pauli_string, PauliString(nq, findall(checks[i,:]) => Pauli(1)))
	end
	for i in axes(checks,1)
		push!(pauli_string, PauliString(nq, findall(checks[i,:]) => Pauli(3)))
	end
	return pauli_string
end

# codes from QECCore
function stabilizers(c::AbstractQECC)
	pm = parity_matrix(c)
	nq = size(pm, 2) ÷ 2
	pauli_string = PauliString{nq}[]

	for i in axes(pm, 1)
		xs = findall(pm[i,1:nq])
		zs = findall(pm[i,nq+1:end])
		ys = xs ∩ zs
		xs = setdiff(xs, ys)
		zs = setdiff(zs, ys)
		push!(pauli_string, PauliString(nq, xs => Pauli(1), ys => Pauli(2), zs => Pauli(3)))
	end
	return pauli_string
end
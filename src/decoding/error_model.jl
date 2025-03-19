abstract type AbstractErrorModel end
abstract type AbstractClassicalErrorModel <: AbstractErrorModel end
abstract type AbstractQuantumErrorModel <: AbstractErrorModel end

"""
    FlipError(p::Float64) <: AbstractClassicalErrorModel

A classical error model that flips a qubit with probability `p`.
"""
struct FlipError <: AbstractClassicalErrorModel
    p::Float64
end

"""
    DepolarizingError <: AbstractQuantumErrorModel
    DepolarizingError(p::Float64)
    DepolarizingError(px::Float64, py::Float64, pz::Float64)
    
A quantum error model that flips a qubit by a Pauli operator σ with probability `pσ`.
Fields:
- `px::Float64`: the probability of an X error
- `py::Float64`: the probability of a Y error
- `pz::Float64`: the probability of a Z error
"""
struct DepolarizingError <: AbstractQuantumErrorModel
    px::Float64
    py::Float64
    pz::Float64
end
DepolarizingError(p::Float64) = DepolarizingError(p, p, p)
 
"""
    random_error_qubits(qubit_number::Int, em::AbstractErrorModel)

Generate a random error pattern for a given number of qubits and an error model.
"""
function random_error_qubits(qubit_number::Int, em::FlipError)
    return Mod2.([rand() < em.p for _ in 1:qubit_number])
end

"""
    CSSErrorPattern(xerror::Vector{Mod2}, zerror::Vector{Mod2})

A CSS error pattern with X and Z errors.
Fields:
- `xerror::Vector{Mod2}`: the X errors
- `zerror::Vector{Mod2}`: the Z errors
"""
struct CSSErrorPattern
    xerror::Vector{Mod2}
    zerror::Vector{Mod2}
end

Base.show(io::IO, ::MIME"text/plain", cep::CSSErrorPattern) = show(io, cep)
function Base.show(io::IO, cep::CSSErrorPattern)
    println(io, "X error:", findall(v->v.x, cep.xerror))
    println(io, "Z error:", findall(v->v.x,cep.zerror))
end
function random_error_qubits(qubit_number::Int, em::DepolarizingError)
    return random_error_qubits(fill(em, qubit_number))
end

function random_error_qubits(ems::Vector{DepolarizingError})
    xerror =  Mod2[]
    zerror =  Mod2[]

    for em in ems
        randnum = rand()
        if randnum < em.py
            push!(xerror, Mod2(true))
            push!(zerror, Mod2(true))
        elseif randnum < em.px + em.py
            push!(xerror, Mod2(true))
            push!(zerror, Mod2(false))
        elseif randnum < em.px + em.py + em.pz
            push!(xerror, Mod2(false))
            push!(zerror, Mod2(true))
        else
            push!(xerror, Mod2(false))
            push!(zerror, Mod2(false))
        end
    end
    return CSSErrorPattern(xerror, zerror)
end
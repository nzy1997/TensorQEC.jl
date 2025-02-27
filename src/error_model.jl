abstract type AbstractErrorModel end
abstract type AbstractClassicalErrorModel <: AbstractErrorModel end
abstract type AbstractQuantumErrorModel <: AbstractErrorModel end

struct FlipError <: AbstractClassicalErrorModel
    p::Float64
end

struct DepolarizingError <: AbstractQuantumErrorModel
    px::Float64
    py::Float64
    pz::Float64
end
DepolarizingError(p::Float64) = DepolarizingError(p, p, p)
 
function random_error_qubits(qubit_number::Int, em::FlipError)
    return Mod2.([rand() < em.p for _ in 1:qubit_number])
end

struct CSSErrorPattern
    xerror::Vector{Mod2}
    zerror::Vector{Mod2}
end

function random_error_qubits(qubit_number::Int, em::DepolarizingError)
    xerror =  Mod2[]
    zerror =  Mod2[]

    for i in 1:qubit_number
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
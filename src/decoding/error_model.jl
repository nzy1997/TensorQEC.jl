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

uniform_error_vector(p::Float64,tanner::SimpleTannerGraph) = fill(p,tanner.nq)
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
uniform_error_vector(p::Float64,tanner::CSSTannerGraph) = fill(DepolarizingError(p),nq(tanner))

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
    xe = findall(v->v.x, cep.xerror)
    ze = findall(v->v.x, cep.zerror)
    n = length(cep.xerror)
    psx = paulistring(n,2,xe)
    psz = paulistring(n,4,ze)
    println(io, (PauliGroup(1,psx)*PauliGroup(1,psz)).ps)
    return
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

struct SimpleSyndrome <: AbstractSyndrome
    s::Vector{Mod2}
end
Base.:(==)(s1::SimpleSyndrome, s2::SimpleSyndrome) = s1.s == s2.s

"""
    syndrome_extraction(errored_qubits::Vector{Mod2}, H::Matrix{Mod2})
    syndrome_extraction(errored_qubits::Vector{Mod2}, tanner::SimpleTannerGraph)
    syndrome_extraction(error_patterns::CSSErrorPattern, tanner::CSSTannerGraph)

Extract the syndrome from the error pattern.
"""
function syndrome_extraction(errored_qubits::Vector{Mod2}, H::Matrix{Mod2})
    return SimpleSyndrome(H * errored_qubits)
end
function syndrome_extraction(errored_qubits::Vector{Mod2}, tanner::SimpleTannerGraph)
    return syndrome_extraction(errored_qubits, tanner.H)
end

struct CSSSyndrome <: AbstractSyndrome
    sx::Vector{Mod2}
    sz::Vector{Mod2}
end
Base.:(==)(s1::CSSSyndrome, s2::CSSSyndrome) = s1.sx == s2.sx && s1.sz == s2.sz

function syndrome_extraction(error_patterns::CSSErrorPattern, tanner::CSSTannerGraph)
    return CSSSyndrome(syndrome_extraction(error_patterns.zerror, tanner.stgx).s, syndrome_extraction(error_patterns.xerror, tanner.stgz).s)
end

function check_logical_error(errored_qubits1::Vector{Mod2}, errored_qubits2::Vector{Mod2}, lz::Matrix{Mod2})
    return any(i->sum(lz[i,:].*(errored_qubits1-errored_qubits2)).x, 1:size(lz,1))
end

function check_logical_error(errored_qubits1::CSSErrorPattern, errored_qubits2::CSSErrorPattern, lx::Matrix{Mod2}, lz::Matrix{Mod2})
    return check_logical_error(errored_qubits1.zerror, errored_qubits2.zerror, lx) && check_logical_error(errored_qubits1.xerror, errored_qubits2.xerror, lz)
end

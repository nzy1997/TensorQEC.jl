abstract type AbstractErrorModel end
abstract type AbstractClassicalErrorModel <: AbstractErrorModel end
abstract type AbstractQuantumErrorModel <: AbstractErrorModel end

struct TNDistribution <:AbstractQuantumErrorModel 
	ptn::TensorNetwork # probability distributions
end

struct IndependentFlipError{T} <: AbstractClassicalErrorModel
    p::Vector{T}
end
iid_error(p::T,n::Int) where T <: Real = IndependentFlipError(fill(p,n))
iid_error(p,tanner::SimpleTannerGraph) = iid_error(p,nq(tanner))

struct IndependentDepolarizingError{T} <: AbstractQuantumErrorModel
    px::Vector{T}
    py::Vector{T}
    pz::Vector{T}
end
iid_error(px::T,py::T,pz::T,n::Int) where T <: Real = IndependentDepolarizingError(fill(px,n),fill(py,n),fill(pz,n))
iid_error(p::T, tanner::CSSTannerGraph) where T <: Real = iid_error(p,p,p,nq(tanner))

"""
    random_error_qubits(qubit_number::Int, em::AbstractErrorModel)

Generate a random error pattern for a given number of qubits and an error model.
"""
function random_error_qubits(em::IndependentFlipError)
    return Mod2.([rand() < em.p[i] for i in 1:length(em.p)])
end

"""
    CSSErrorPattern(xerror::Vector{Mod2}, zerror::Vector{Mod2})

A CSS error pattern with X and Z errors.
Fields:
- `xerror::Vector{Mod2}`: the X errors
- `zerror::Vector{Mod2}`: the Z errors
"""
struct CSSErrorPattern{VM <:AbstractVector{Mod2}}
    xerror::VM
    zerror::VM
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

function random_error_qubits(em::IndependentDepolarizingError)
    xerror =  Mod2[]
    zerror =  Mod2[]
    for i in 1:length(em.px)
        randnum = rand()
        if randnum < em.py[i]
            push!(xerror, Mod2(true))
            push!(zerror, Mod2(true))
        elseif randnum < em.px[i] + em.py[i]
            push!(xerror, Mod2(true))
            push!(zerror, Mod2(false))
        elseif randnum < em.px[i] + em.py[i] + em.pz[i]
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
    return check_logical_error(errored_qubits1.zerror, errored_qubits2.zerror, lx) || check_logical_error(errored_qubits1.xerror, errored_qubits2.xerror, lz)
end

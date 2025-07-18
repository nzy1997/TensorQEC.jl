struct Tableau{N}
    tabx::Vector{PauliGroupElement{N}}
    tabz::Vector{PauliGroupElement{N}}
end

function new_tableau(n::Int)
    return Tableau([PauliGroupElement(0,PauliString(n, i => Pauli(1))) for i in 1:n], [PauliGroupElement(0,PauliString(n, i => Pauli(3))) for i in 1:n])
end

Base.show(io::IO, ::MIME"text/plain", tab::Tableau) = show(io, tab)
function Base.show(io::IO, tab::Tableau{N}) where {N}
    for i in 1:N
        print(io, "X$(i): ", tab.tabx[i], "\n")
    end
    for i in 1:N
        print(io, "Z$(i): ", tab.tabz[i], "\n")
    end
end


function tableau_simulate(tab::Tableau{N}, operation::Pair{NTuple{M,Int}, <:CliffordGate}) where {M, N}
    return Tableau([operation.second(tab.tabx[i], operation.first) for i in 1:N], [operation.second(tab.tabz[i], operation.first) for i in 1:N])
end

function tableau_simulate(tab::Tableau{N}, qc::ChainBlock) where N
    qc = simplify(qc; rules=[to_basictypes, Optimise.eliminate_nested])
    gatedict=Dict{UInt64, CliffordGate}()
    for _gate in qc
        gate = toput(_gate)
        key = hash(gate.content)
        if haskey(gatedict, key) 
            tab = tableau_simulate(tab, gate.locs=>gatedict[key])
        else 
            pm = CliffordGate(gate.content)
            push!(gatedict, key => pm)
            tab = tableau_simulate(tab, gate.locs=>pm)
        end
    end
    return tab
end
function tableau_simulate(qc::ChainBlock)
    tab = new_tableau(nqubits(qc))
    return tableau_simulate(tab, qc)
end
function tableau_simulate(ps::PauliString{N}, qc::ChainBlock) where N
    tab = tableau_simulate(qc)
    res = PauliGroupElement(0, PauliString(N, 1 => Pauli(0)))
    count = 0
    for i in 1:N
        if ps[i] == Pauli(1)
            res *= tab.tabx[i]
        elseif ps[i] == Pauli(3)
            res *= tab.tabz[i]
        elseif ps[i] == Pauli(2)
            res *= tab.tabx[i] * tab.tabz[i]
            count += 1
        end
    end
    return PauliGroupElement(_add_phase(res.phase, count), res.ps)
end
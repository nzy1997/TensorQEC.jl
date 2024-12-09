struct Tableau{N}
    tabx::Vector{PauliGroup{N}}
    tabz::Vector{PauliGroup{N}}
end

function new_tableau(n::Int)
    return Tableau([PauliGroup(0,paulistring(n,2,i)) for i in 1:n], [PauliGroup(0,paulistring(n,4,i)) for i in 1:n])
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


function tableau_simulate(tab::Tableau{N}, operation::Pair{Vector{Int}, <:PermMatrix}) where N
    return Tableau([perm_of_pauligroup(tab.tabx[i], operation) for i in 1:N], [perm_of_pauligroup(tab.tabz[i], operation) for i in 1:N])
end

function tableau_simulate(tab::Tableau{N}, qc::ChainBlock) where N
    qc = simplify(qc; rules=[to_basictypes, Optimise.eliminate_nested])
    gatedict=Dict{UInt64, PermMatrix}()
    for _gate in qc
        gate = toput(_gate)
        key = hash(gate.content)
        if haskey(gatedict, key) 
            tab = tableau_simulate(tab, collect(gate.locs)=>gatedict[key])
        else 
            pm = to_perm_matrix(Int8, UInt8, pauli_repr(mat(gate.content)))
            push!(gatedict, key => pm)
            tab = tableau_simulate(tab, collect(gate.locs)=>pm)
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
    res = PauliGroup{N}(0, paulistring(N, 1,1))
    count = 0
    for i in 1:N
        if ps.ids[i] == 2
            res *= tab.tabx[i]
        elseif ps.ids[i] == 4
            res *= tab.tabz[i]
        elseif ps.ids[i] == 3
            res *= tab.tabx[i] * tab.tabz[i]
            count += 1
        end
    end
    return PauliGroup{N}(_mul_coeff(res.coeff,count), res.ps)
end
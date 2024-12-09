struct Tableau{N}
    xfactor::Vector{ComplexF64}
    zfactor::Vector{ComplexF64}
    tabx::Vector{PauliString{N}}
    tabz::Vector{PauliString{N}}
end

function new_tableau(n::Int)
    return Tableau(ones(ComplexF64,n), ones(ComplexF64,n), [paulistring(n,2,i) for i in 1:n], [paulistring(n,4,i) for i in 1:n])
end

Base.show(io::IO, ::MIME"text/plain", tab::Tableau) = show(io, tab)
function Base.show(io::IO, tab::Tableau{N}) where {N}
    for i in 1:N
        print(io,"X_$i:")
        print(io, tab.xfactor[i])
        print(io, " ")
        println(io, tab.tabx[i])
    end
    for i in 1:N
        print(io,"Z_$i:")
        print(io, tab.zfactor[i])
        print(io, " ")
        println(io, tab.tabz[i])
    end
end


function tableau_simulate(tab::Tableau{N}, operation::Pair{Vector{Int}, <:PermMatrix}) where N
    resx = [perm_of_paulistring(tab.tabx[i], operation) for i in 1:N]
    resz = [perm_of_paulistring(tab.tabz[i], operation) for i in 1:N]
    return Tableau([resx[i][2] * tab.xfactor[i] for i in 1:N], [resz[i][2] * tab.zfactor[i] for i in 1:N], [resx[i][1] for i in 1:N], [resz[i][1] for i in 1:N])
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

function tableau_simulate(ps::PauliString{N}, qc::ChainBlock) where N
    tab = new_tableau(N)
    tab = tableau_simulate(tab, qc)
    val = 1 + 0im
    psout = paulistring(N, 1,1)
    res = PauliGroup{N}(0, psout)
    for i in 1:N
        if ps.ids[i] == 2
            val *= tab.xfactor[i]
            psout = psout * tab.tabx[i]
        elseif ps.ids[i] == 4
            val *= tab.zfactor[i]
            psout = psout * tab.tabz[i]
        elseif ps.ids[i] == 3
            val *= im*tab.xfactor[i] * tab.zfactor[i]
            psout = psout * tab.tabx[i] * tab.tabz[i]
        end
    end
    return psout, val
end
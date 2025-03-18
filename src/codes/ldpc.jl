abstract type AbstractTannerGraph end
abstract type AbstractSyndrome end
"""
    SimpleTannerGraph(nq::Int, ns::Int, q2s::Vector{Vector{Int}}, s2q::Vector{Vector{Int}}, H::Matrix{Mod2})

Tanner graph for a classical linear code.
Fields:
    nq: number of qubits
    ns: number of stabilizers
    q2s: a list of lists, q2s[i] is the list of stabilizers that contain qubit i
    s2q: a list of lists, s2q[i] is the list of qubits that stabilizer i contains
    H: the parity check matrix
"""
struct SimpleTannerGraph <: AbstractTannerGraph
    nq::Int
    ns::Int
    q2s::Vector{Vector{Int}}
    s2q::Vector{Vector{Int}}
    H::Matrix{Mod2}
end

"""
    SimpleTannerGraph(nq::Int, sts::Vector{Vector{Int}})

Construct a Tanner graph from a list of stabilizers.
Input:
    nq: number of qubits
    sts: a list of parity checks, each parity check is a list of bits.
"""
function SimpleTannerGraph(nq::Int, sts::Vector{Vector{Int}})
    ns = length(sts)
    q2s = [findall(x-> i ∈ x , sts) for i in 1:nq]
    H = zeros(Mod2, ns, nq)
    for i in 1:ns, j in sts[i]
        H[i, j] = 1
    end
    return SimpleTannerGraph(nq, ns, q2s, sts, H)
end

"""
    CSSCSSTannerGraph(stgx::SimpleTannerGraph, stgz::SimpleTannerGraph)
    CSSTannerGraph(nq::Int, stxs::Vector{Vector{Int}}, stzs::Vector{Vector{Int}})
    CSSTannerGraph(sts::Vector{PauliString{N}}) where N
    CSSTannerGraph(cqc::CSSQuantumCode)

Two tanner graph for a CSS code, one for X stabilizers and one for Z stabilizers.
Fields:
    stgx: Tanner graph for X stabilizers
    stgz: Tanner graph for Z stabilizers
"""
struct CSSTannerGraph <: AbstractTannerGraph
    stgx::SimpleTannerGraph
    stgz::SimpleTannerGraph
end
nq(stg::SimpleTannerGraph) = stg.nq
nq(stg::CSSTannerGraph) = stg.stgx.nq
function dual_graph(tanner::SimpleTannerGraph)
    return SimpleTannerGraph( tanner.ns,  tanner.nq, tanner.s2q, tanner.q2s, transpose(tanner.H))
end

"""
    get_graph(tanner::SimpleTannerGraph)
    get_graph(ctg::CSSTannerGraph)

Get the simple graph of a simple tanner graph or a CSS tanner graph.
"""
function get_graph(stg::SimpleTannerGraph)
    sg = SimpleGraph([zeros(Bool,(stg.nq,stg.nq)) getproperty.(stg.H,:x)';
       getproperty.(stg.H,:x) zeros(Bool,(stg.ns,stg.ns))])
    return sg
end
function get_graph(ctg::CSSTannerGraph)
    sg = SimpleGraph([zeros(Bool,(ctg.stgx.nq,ctg.stgx.nq)) getproperty.(ctg.stgx.H,:x)' getproperty.(ctg.stgz.H,:x)';
       [getproperty.(ctg.stgx.H,:x);  getproperty.(ctg.stgz.H,:x)] zeros(Bool,(ctg.stgz.ns+ctg.stgx.ns,ctg.stgz.ns+ctg.stgx.ns))])
    return sg
end

function CSSTannerGraph(nq::Int, stxs::Vector{Vector{Int}}, stzs::Vector{Vector{Int}})
    return CSSTannerGraph(SimpleTannerGraph(nq, stxs), SimpleTannerGraph(nq, stzs))
end

function CSSTannerGraph(sts::Vector{PauliString{N}}) where N
    xlabel = findall([st.ids[findfirst(!=(1),st.ids)] == 2 for st in sts])
    zlabel = findall([st.ids[findfirst(!=(1),st.ids)] == 4 for st in sts])
    stxs = [findall(!=(1),st.ids) for st in sts[xlabel]]
    stzs = [findall(!=(1),st.ids) for st in sts[zlabel]]
    return CSSTannerGraph(N, stxs, stzs)
end

function CSSTannerGraph(cqc::CSSQuantumCode)
    return CSSTannerGraph(stabilizers(cqc))
end

"""
    syndrome_extraction(errored_qubits::Vector{Mod2}, H::Matrix{Mod2})
    syndrome_extraction(errored_qubits::Vector{Mod2}, tanner::SimpleTannerGraph)
    syndrome_extraction(error_patterns::CSSErrorPattern, tanner::CSSTannerGraph)

Extract the syndrome from the error pattern.
"""
function syndrome_extraction(errored_qubits::Vector{Mod2}, H::Matrix{Mod2})
    return H * errored_qubits
end
function syndrome_extraction(errored_qubits::Vector{Mod2}, tanner::SimpleTannerGraph)
    return syndrome_extraction(errored_qubits, tanner.H)
end
struct CSSSyndrome <: AbstractSyndrome
    sx::Vector{Mod2}
    sz::Vector{Mod2}
end

function syndrome_extraction(error_patterns::CSSErrorPattern, tanner::CSSTannerGraph)
    return CSSSyndrome(syndrome_extraction(error_patterns.zerror, tanner.stgx), syndrome_extraction(error_patterns.xerror, tanner.stgz))
end

function coordinate_transform(i::Int, j::Int, nq2::Int)
    return (i-1)*nq2+j
end

function coordinate_transform(i::Int, j::Int,nq12::Int,ns2::Int)
    return (i-1)*ns2+j+nq12
end

function product_graph(tanner1::SimpleTannerGraph, tanner2::SimpleTannerGraph)
    nq = tanner1.nq*tanner2.nq+tanner1.ns*tanner2.ns
    stxs = [[[[coordinate_transform(i, k, tanner2.nq) for k in tanner2.s2q[j]]..., [coordinate_transform(k, j,  tanner1.nq*tanner2.nq, tanner2.ns) for k in tanner1.q2s[i]]...]  for i in 1:tanner1.nq, j in 1:tanner2.ns]...]
    stzs = [[[[coordinate_transform(i, k,tanner1.nq*tanner2.nq, tanner2.ns) for k in tanner2.q2s[j]]..., [coordinate_transform(k, j,   tanner2.nq) for k in tanner1.s2q[i]]...]  for i in 1:tanner1.ns, j in 1:tanner2.nq]...]
    return CSSTannerGraph(nq, stxs, stzs)
end

function random_ldpc(n1::Int,n2::Int,nq::Int)
    ns = nq*n2/n1
    qcount = zeros(nq)
    sts = Vector{Vector{Int}}()
    qvec = collect(1:nq)
    for _ in 1:ns
        stsi=Int[]
        for _ in 1:n1
            q = rand(setdiff(qvec, stsi))   
            push!(stsi, q)
            qcount[q] += 1
        end
        push!(sts, stsi)
        qvec = setdiff(qvec, findall(x -> x==n2,qcount))
    end
    return SimpleTannerGraph(nq, sts)
end

function check_decode(error_qubits::Vector{Mod2}, syd::Vector{Mod2}, H::Matrix{Mod2})
    return syd == syndrome_extraction(error_qubits, H)
end

function check_decode(error_qubits::Vector{Mod2}, syd::Vector{Mod2}, tanner::SimpleTannerGraph)
    return check_decode(error_qubits, syd, tanner.H)
end

function check_logical_error(errored_qubits1::Vector{Mod2}, errored_qubits2::Vector{Mod2}, H)
    return !check_linear_indepent([a.x for a in [H;transpose(errored_qubits1+errored_qubits2)]])
end

"""
    SimpleTannerGraph(nq::Int, sts::Vector{Vector{Int}})

Construct a simple tanner graph from a list of stabilizers.
Input:
    nq: number of qubits
    sts: a list of stabilizers, each stabilizer is a list of qubits
"""
struct SimpleTannerGraph
    nq::Int
    ns::Int
    q2s::Vector{Vector{Int}}
    s2q::Vector{Vector{Int}}
    H::Matrix{Mod2}
end

struct BPResult
    success_tag::Bool
    error_qubits::Vector{Mod2}
    iter::Int
    error_perm::Vector{Int}
end

function SimpleTannerGraph(nq::Int, sts::Vector{Vector{Int}})
    ns = length(sts)
    q2s = [findall(x-> i ∈ x , sts) for i in 1:nq]
    H = zeros(Mod2, ns, nq)
    for i in 1:ns, j in sts[i]
        H[i, j] = 1
    end
    SimpleTannerGraph(nq, ns, q2s, sts, H)
end

struct CSSTannerGraph
    stgx::SimpleTannerGraph
    stgz::SimpleTannerGraph
end

function dual_graph(tanner::SimpleTannerGraph)
    return SimpleTannerGraph( tanner.ns,  tanner.nq, tanner.s2q, tanner.q2s, transpose(tanner.H))
end

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
function plot_graph(stg::SimpleTannerGraph)
    sg = get_graph(stg)
    zlocs = [fill(1, stg.nq)..., fill(300, stg.ns)...]
    return _plot_graph(sg, [[string("$i") for i=1:stg.nq]...,[string("s$i") for i=1:stg.ns]...], zlocs)
end

function plot_graph(ctg::CSSTannerGraph)
    sg = get_graph(ctg)
    zlocs = [fill(300, ctg.stgx.nq)..., fill(0, ctg.stgx.ns)..., fill(600, ctg.stgz.ns)...]
    return _plot_graph(sg, [[string("$i") for i=1:ctg.stgx.nq]...,[string("x$i") for i=1:ctg.stgx.ns]...,[string("z$i") for i=1:ctg.stgz.ns]...], zlocs)
end

function _plot_graph(sg::SimpleGraph,texts,zlocs)
    optimal_distance = 40.0
    layout = Layouts.LayeredStressLayout(; zlocs, optimal_distance)
    return show_graph(sg,Layouts.render_locs(sg, layout), texts=texts)
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

function sydrome_extraction(errored_qubits::Vector{Mod2}, H::Matrix{Mod2})
    return H * errored_qubits
end
function sydrome_extraction(errored_qubits::Vector{Mod2}, tanner::SimpleTannerGraph)
    return sydrome_extraction(errored_qubits, tanner.H)
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

function belief_propagation(sydrome::Vector{Mod2}, tanner::SimpleTannerGraph, p::Float64;max_iter=100)
    mu = [log((1-p)/p) for _ in 1:tanner.nq]
    mq2s =[[mu[i] for _ in 1:length(tanner.q2s[i])] for i in 1:tanner.nq]
    ms2q = [[0.0 for _ in 1:length(tanner.s2q[s])] for s in 1:tanner.ns]
    # mq2s =[[rand() for _ in v] for v in tanner.q2s]
    errored_qubits = Mod2[]
    success_tag = false
    iter = 0
    for iter in 1:max_iter
        ms2q = [[messages2q(message_list(mq2s,s,tanner.q2s,tanner.s2q;exampt_qubit = q),sydrome[s]) for q in tanner.s2q[s]] for s in 1:tanner.ns]
        mq2s = [[messageq2s(message_list(ms2q,qubit,tanner.s2q,tanner.q2s;exampt_qubit = s),mu[qubit]) for s in tanner.q2s[qubit]] for qubit in 1:tanner.nq]
        errored_qubits = error_qubits(ms2q,tanner,mu)
        if sydrome_extraction(errored_qubits, tanner) == sydrome
            success_tag = true
            break
        end
    end
    return BPResult(success_tag, errored_qubits, iter, sortperm(messageq(ms2q,tanner,mu)))
end

function message_list(mq2s,s,tq2s,ts2q;exampt_qubit = 0)
    return [mq2s[q][findfirst(==(s),tq2s[q])] for q in ts2q[s] if q != exampt_qubit]
end

function messageq(ms2q,tanner,mu)
    [messageq2s(message_list(ms2q,qubit,tanner.s2q,tanner.q2s),mu[qubit])  for qubit in 1:tanner.nq]
end

function error_qubits(ms2q,tanner,mu)
    Mod2.(messageq(ms2q,tanner,mu) .< 0 )
end

function messages2q(mlist,sydrome)
    pro = 1.0
    for m in mlist
        pro *= tanh(m)
    end
    return (-1)^sydrome.x*atanh(pro)
end

function messageq2s(mlist,muq)
    return max(min(sum(mlist) + muq,10),-10)
end

function random_ldpc(n1::Int,n2::Int,nq::Int)
    ns = nq*n2/n1
    qcount = zeros(nq)
    sts = Vector{Vector{Int}}()
    qvec = collect(1:nq)
    for _ in 1:ns
        stsi=Int[]
        for _ in 1:n1
            # @show qvec,stsi,qcount[20],qcount[26]
            q = rand(setdiff(qvec, stsi))   
            push!(stsi, q)
            qcount[q] += 1
        end
        push!(sts, stsi)
        qvec = setdiff(qvec, findall(x -> x==n2,qcount))
    end
    return SimpleTannerGraph(nq, sts)
end

function random_error_qubits(qubit_number,p)
    return Mod2.([rand() < p for _ in 1:qubit_number])
end

function check_decode(error_qubits::Vector{Mod2}, syd::Vector{Mod2}, H::Matrix{Mod2})
    return syd == sydrome_extraction(error_qubits, H)
end

function check_decode(error_qubits::Vector{Mod2}, syd::Vector{Mod2}, tanner::SimpleTannerGraph)
    return check_decode(error_qubits, syd, tanner.H)
end

function check_decode(errored_qubits1::Vector{Mod2}, errored_qubits2::Vector{Mod2}, H)
    return !check_linear_indepent([a.x for a in [H;transpose(errored_qubits1+errored_qubits2)]])
end

function osd(tanner::SimpleTannerGraph,order::Vector{Int},syndrome::Vector{Mod2})
    H = tanner.H[:,order[1:1]]
    hinv = 0
    qubit_list = [order[1]]
    for i in 1:length(order)
        if check_linear_indepent(Matrix([H  tanner.H[:,order[i:i]]]'))
            H = [H  tanner.H[:,order[i:i]]]
            push!(qubit_list,order[i])
        end
        if size(H,2) == tanner.ns
            break
        end
    end

    hinv = mod2matrix_inverse(H)

    error = hinv * syndrome
    return [(i ∈ qubit_list) ? (error[findfirst(==(i),qubit_list)]) : Mod2(0)  for i in 1:tanner.nq]
end

function mod2matrix_inverse(H::Matrix{Bool})
    bm = SimpleBimatrix(copy(H),Matrix{Mod2}(I, size(H,1), size(H,1)),collect(1:size(H,2)))
    gaussian_elimination!(bm, 1:size(bm.matrix,1), 0, 0;allow_col_operation = false)
    return bm.Q
end

function mod2matrix_inverse(H::Matrix{Mod2})
    return mod2matrix_inverse([a.x for a in H])
end
function _check_linear_indepent(H::Matrix{Bool})
    bm = SimpleBimatrix(H,Matrix{Mod2}(I, size(H,1), size(H,1)),collect(1:size(H,2)))
    gaussian_elimination!(bm, 1:size(bm.matrix,1), 0, 0)
    return bm,!(bm.matrix[end,:] == fill(Mod2(0),size(H,2)))
end

function check_linear_indepent(H::Matrix{Bool})
    _ , res= _check_linear_indepent(H)
    return res
end

function check_linear_indepent(H::Matrix{Mod2})
    return check_linear_indepent([a.x for a in H])
end

function ldpc2tensor(tanner::SimpleTannerGraph, p::Float64, syndrome::Vector{Mod2})
    nvars = tanner.nq + tanner.ns 
    cards = fill(2, nvars)
    factors = [Factor(((tanner.s2q[s] ∪ (s+tanner.nq))...,), parity_check_matrix(length(tanner.s2q[s]))) for s in 1:(tanner.ns)]
    pfac = [Factor((q,), [1-p, p]) for q in 1:tanner.nq]
    sfac = [Factor((s+tanner.nq,), syndrome[s].x ? [1.0, 0.0] : [0.0, 1.0 ]) for s in 1:tanner.ns]
    return TensorNetworkModel(
		1:nvars,
		cards,
		factors ∪ pfac ∪ sfac;
		openvars = [],
		mars = [[i] for i in 1:tanner.nq],
	)
end

function parity_check_matrix(qubit_num::Int)
    return reshape([Float64(count_ones(i-1)%2) for i in 1:2^(qubit_num+1)], (fill(2,qubit_num+1)...))
end

function tensor_infer(tanner::SimpleTannerGraph, p::Float64, syndrome::Vector{Mod2})
    tn = ldpc2tensor(tanner, p, syndrome)
    mp = marginals(tn)
    return [ mp[[k]][2] for k in 1:tanner.nq]
end

function bp_osd(sydrome::Vector{Mod2}, tanner::SimpleTannerGraph, p::Float64;max_iter=100)
    bp_res = belief_propagation(sydrome, tanner, p;max_iter=max_iter)
    if bp_res.success_tag
        return bp_res.error_qubits
    else
        return osd(tanner, bp_res.error_perm, sydrome)
    end
end
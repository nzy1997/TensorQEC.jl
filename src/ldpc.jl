struct SimpleTannerGraph
    nq::Int
    ns::Int
    q2s::Vector{Vector{Int}}
    s2q::Vector{Vector{Int}}
end
function SimpleTannerGraph(nq::Int, sts::Vector{Vector{Int}})
    ns = length(sts)
    q2s = [findall(x-> i ∈ x , sts) for i in 1:nq]
    SimpleTannerGraph(nq, ns, q2s, sts)
end

function sydrome_extraction(errored_qubits::Vector{Int}, tanner::SimpleTannerGraph)
    syndrome = zeros(Int, length(tanner.sts))
    for (i, st) in enumerate(tanner.sts)
        for q in st
            if q in errored_qubits
                syndrome[i] = mod1(syndrome[i] + 1)
            end
        end
    end
    return syndrome
end
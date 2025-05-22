"""
	TruthTable

The truth table for error correction.

### Fields
- `table::Dict{Int,Int}`: The truth table for error correction.
- `num_qubits::Int`: The number of qubits.
- `num_st::Int`: The number of stabilizers.
- `d::Int64`: The maximum number of errors.
"""
struct TruthTable{INTs <: Integer,INT <: Integer}
	table::Dict{INTs, Tuple{INT,INT}} # syndrome -> (xerror,zerror)
	num_qubits::Int
	num_st::Int
end

# visualization
Base.show(io::IO, ::MIME"text/plain", tb::TruthTable) = show(io, tb)
function Base.show(io::IO, tb::TruthTable)
	es = Vector{Vector{Int64}}()
    eq = Vector{CSSErrorPattern}()
	for (k, v) in tb.table
		push!(es, find1pos(k, tb.num_st))
        push!(eq, lluint2error(v, tb.num_qubits))
	end
	header = ["syndrome", "Error"]
	pt = pretty_table(io, hcat(es, eq); header)
	return nothing
end


function find1pos(v::INT,max_pos::Int) where INT <: Integer
	return findall(x-> !iszero(readbit(v,x)),1:max_pos)
end

abstract type AbstractSyndromeConflict end
struct UniformError <:AbstractSyndromeConflict end

function make_table(tanner::CSSTannerGraph,d::Int, sc::AbstractSyndromeConflict)
	N = nq(tanner)
	num_st = ns(tanner)
	INT = BitBasis.longinttype(N, 2)
	INTs = BitBasis.longinttype(num_st,2)
	error_vec = 1:N
	count = 0
	table = Dict{INTs, Tuple{INT, INT}}()

	table[zero(INTs)] = (zero(INT), zero(INT))
	for error_num in 1:d
		error_patternx = [[(i÷(3^(j-1)) % 3 == 0) || (i÷(3^(j-1)) % 3 == 1)  for j in 1:error_num] for i in 0:3^error_num-1]
		error_patternz = [[(i÷(3^(j-1)) % 3 == 1) || (i÷(3^(j-1)) % 3 == 2) for j in 1:error_num] for i in 0:3^error_num-1]
		for combo in combinations(error_vec, error_num)
			for (epx,epz) in zip(error_patternx, error_patternz)
				xerror = zeros(Mod2, N)
				zerror = zeros(Mod2, N)
				xerror[combo] = epx
				zerror[combo] = epz
				cep = CSSErrorPattern(xerror, zerror)
				syn = syndrome_extraction(cep, tanner)
				syn = syndrome2lluint(syn, INTs)
				xerror = _vec2int(INT, getfield.(xerror,:x))
				zerror = _vec2int(INT, getfield.(zerror,:x))
				if !haskey(table, syn)
					table[syn] = (xerror,zerror)
				else
					conflict_syndrome(table[syn], (xerror,zerror),sc) && (table[syn] = (xerror,zerror))
				end
				count += 1
			end

		end
	end
	return TruthTable(table, N, num_st)
end

function get_probability(sc::UniformError, cep::Tuple{INT, INT}) where {INT}
	return 0.0
end

function syndrome2lluint(syn::CSSSyndrome,::Type{T}) where T <: Integer
	return _vec2int(T, [getfield.(syn.sx,:x)..., getfield.(syn.sz,:x)...])
end

function _vec2int(::Type{T}, v::AbstractVector) where T <: Integer
    res = zero(T)
    for i in 1:length(v)
        res |= T(v[i]) << (i-1)
    end
    return res
end

function lluint2error(ep::Tuple{INT, INT}, num_qubits::Int) where INT <: Integer
	xerror = zeros(Mod2, num_qubits)
	zerror = zeros(Mod2, num_qubits)
	for i in 1:num_qubits
		iszero(readbit(ep[1], i)) || (xerror[i] = 1)
		iszero(readbit(ep[2], i)) || (zerror[i] = 1)
	end
	return CSSErrorPattern(xerror, zerror)
end

struct CompiledTable{INTs <: Integer,INT <: Integer} <: CompiledDecoder
   table::TruthTable{INTs,INT}
end

function decode(ct::CompiledTable{INTs,INT}, syndrome::CSSSyndrome) where {INTs,INT}
	syn = syndrome2lluint(syndrome, INTs)
	if haskey(ct.table.table, syn)
		return DecodingResult(true,lluint2error(ct.table.table[syn], ct.table.num_qubits))
	else
		return DecodingResult(false, CSSErrorPattern(zeros(Mod2, ct.table.num_qubits), zeros(Mod2, ct.table.num_qubits)))
	end
end

function save_table(tb::TruthTable, filename::String)
	writedlm(filename,vcat.(broadcast(x->collect(x.content) ,collect(keys(tb.table))),broadcast(x->[x[1].content...,x[2].content...] ,collect(values(tb.table)))))
end

function load_table(filename::String, num_qubits::Int, num_st::Int)
	INT = BitBasis.longinttype(num_qubits, 2)
	INTs = BitBasis.longinttype(num_st,2)
	C = INT.parameters[1]
	Cs = INTs.parameters[1]
	data = readdlm(filename,UInt)
	table = Dict{INTs, Tuple{INT, INT}}()
	for i in 1:size(data, 1)
		table[INTs((data[i, 1:Cs]...,))] = (INT((data[i, Cs+1:Cs+C]...,)), INT((data[i, Cs+1+C:end]...,)))
	end
	return TruthTable(table, num_qubits, num_st)
end

# p1 is the old one
function conflict_syndrome(cep1::Tuple{INT, INT}, cep2::Tuple{INT, INT},sc::AbstractSyndromeConflict) where {INT}
	p1 = get_probability(sc, cep1)
	p2 = get_probability(sc, cep2)
	return p1 < p2
end

struct DistributionError{DT} <:AbstractSyndromeConflict 
	dis::DT
end

get_probability(de::DistributionError, cep::Tuple) = get_probability(de.dis, cep)
function get_probability(pm::IndependentDepolarizingError, cep::Tuple{INT, INT}) where {INT}
	p = 1.0
	qubit_num = length(cep[1])
	for i in 1:qubit_num
		if iszero(readbit(cep[1], i)) 
			# no x error
			p = p * (iszero(readbit(cep[2], i)) ? (1-pm.px[i]-pm.py[i]-pm.pz[i]) : pm.pz[i])
		else
			# x error
			p = p * (iszero(readbit(cep[2], i)) ? pm.px[i] : pm.py[i])
		end
	end
	return p
end
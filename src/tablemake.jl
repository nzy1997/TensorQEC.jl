function combo2bivec(combo::AbstractVector{Int64}, qubit_num::Int64)
	bivec = zeros(Bool, 2 * qubit_num)
	for i in combo
		bivec[mod1(i,2*qubit_num)] = true
		if i > 2 * qubit_num
			bivec[i-qubit_num] = true
		end
	end
	return bivec
end

function make_table(mat::Matrix{Bool}, d::Int64)
	qubit_num = size(mat, 2) ÷ 2
    num_st=size(mat,1)
    table = Dict{Int,Int}()
	for i in 1:d
		all_combinations = combinations(1:2*qubit_num, i)
		for combo in all_combinations
            bivec = combo2bivec(combo, qubit_num)

            # sydrome = 0(false) means the measure outcome is 1
            # sydrome = 1(true) means the measure outcome is -1
            sydrome = 0

            for j in 1:num_st
                reduce(xor, bivec[findall(mat[j,:])]) && (sydrome |= 1<< (j-1))
            end
            table[sydrome] = Yao.BitBasis.bit_literal(Int.(bivec)...).buf
		end
	end
    return table
end

function save_table(table::Dict{Int,Int}, filename::String)
    writedlm(filename, hcat(collect(keys(table)), collect(values(table))))
end
function read_table(filename::String)
    data = readdlm(filename)
    return Dict{Int,Int}(zip(data[:,1], data[:,2]))
end
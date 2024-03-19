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

function make_table(mat::Matrix{Bool}, d::Int64) where N
	qubit_num = size(mat, 2) ÷ 2
	for i in 1:d
		all_combinations = combinations(1:3*qubit_num, i)
		for combo in all_combinations
            bivec = combo2bivec(combo, qubit_num)
            # println(bivec)

		end
	end
end

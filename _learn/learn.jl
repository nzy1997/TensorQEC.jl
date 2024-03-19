using Combinatorics

n = 10
k = 2

values = 1:n
all_combinations = combinations(values, k)

for combo in all_combinations
    println(combo[1])
end
